#!/usr/bin/env python3
"""
End-to-end MACE-style forward pass benchmark.

Compares custom "batteries" CUDA kernels against stock e3nn/MACE implementation.

Model config (matching MACE-MP-0 small):
- L_max = 3
- hidden_channels = 32
- 2 interaction layers
- cutoff = 5.0 Angstroms
"""

import torch
import torch.nn as nn
import numpy as np
import ctypes
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# Third-party imports
from e3nn import o3
import torch_scatter
import torch_cluster

# Import batteries CUDA kernels
import _batteries

# Import autograd-compatible wrappers for backward pass support
from python.autograd import (
    spherical_harmonics as batteries_spherical_harmonics,
    tensor_product as batteries_tensor_product,
)

# Load neighbor list library via ctypes
lib_path = Path(__file__).parent.parent / "libbatteries_neighbor.so"
if not lib_path.exists():
    raise RuntimeError(f"Library not found: {lib_path}")

_neighbor_lib = ctypes.CDLL(str(lib_path))


class NeighborListResult(ctypes.Structure):
    _fields_ = [
        ("edge_index", ctypes.c_void_p),
        ("edge_vectors", ctypes.c_void_p),
        ("num_edges", ctypes.c_int64),
        ("d_edge_index", ctypes.c_void_p),
        ("d_edge_vectors", ctypes.c_void_p),
    ]


_neighbor_lib.neighbor_list_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ctypes.c_float, ctypes.c_bool, ctypes.POINTER(NeighborListResult), ctypes.c_void_p
]
_neighbor_lib.neighbor_list_cuda.restype = ctypes.c_int
_neighbor_lib.neighbor_list_free.argtypes = [ctypes.POINTER(NeighborListResult)]
_neighbor_lib.neighbor_list_free.restype = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MACEConfig:
    """MACE model configuration (matching MACE-MP-0 small)."""
    L_max: int = 3
    hidden_channels: int = 32
    num_layers: int = 2
    cutoff: float = 5.0
    num_elements: int = 89  # Periodic table


# =============================================================================
# Test System Generation
# =============================================================================

def generate_amorphous_system(N: int, density: float = 0.05, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random positions with specified density.

    Args:
        N: Number of atoms
        density: Atoms per cubic Angstrom (0.05 typical for liquids)
        seed: Random seed

    Returns:
        positions: [N, 3] atom positions on GPU
        cell: [3, 3] cell matrix on GPU
        atomic_numbers: [N] atomic numbers on GPU
    """
    torch.manual_seed(seed)

    # Calculate box size from density
    volume = N / density
    box_size = volume ** (1/3)

    # Random positions
    positions = torch.rand(N, 3, device="cuda", dtype=torch.float32) * box_size

    # Orthorhombic cell
    cell = torch.zeros(3, 3, device="cuda", dtype=torch.float32)
    cell[0, 0] = box_size
    cell[1, 1] = box_size
    cell[2, 2] = box_size

    # Random atomic numbers (mostly light elements for realistic test)
    # Mix of H, C, N, O, S, P (common in organic materials)
    element_pool = torch.tensor([1, 6, 7, 8, 15, 16], device="cuda", dtype=torch.long)
    atomic_numbers = element_pool[torch.randint(0, len(element_pool), (N,), device="cuda")]

    return positions, cell, atomic_numbers


def replicate_li3ps4_like(target_N: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a Li3PS4-like system by replicating a small unit.

    Li3PS4 has ~8 formula units in unit cell, so ~32 atoms.
    We replicate to reach target_N.
    """
    torch.manual_seed(seed)

    # Approximate Li3PS4 unit cell (scaled for testing)
    # Real Li3PS4: a=12.82, b=8.22, c=6.12 Angstroms
    base_cell = torch.tensor([
        [6.41, 0.0, 0.0],
        [0.0, 4.11, 0.0],
        [0.0, 0.0, 3.06]
    ], device="cuda", dtype=torch.float32)

    # Approximate positions for one formula unit (Li3PS4)
    # Li: 3 atoms, P: 1 atom, S: 4 atoms = 8 atoms per formula unit
    base_positions = torch.tensor([
        # Li atoms (fractional coordinates)
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
        # P atom
        [0.5, 0.5, 0.5],
        # S atoms
        [0.1, 0.5, 0.5],
        [0.9, 0.5, 0.5],
        [0.5, 0.1, 0.5],
        [0.5, 0.9, 0.5],
    ], device="cuda", dtype=torch.float32)

    base_atomic_numbers = torch.tensor([3, 3, 3, 15, 16, 16, 16, 16], device="cuda", dtype=torch.long)
    n_base = len(base_positions)

    # Convert to Cartesian
    base_positions = base_positions @ base_cell

    # Calculate replication factors
    n_replicas = int(np.ceil(target_N / n_base))
    rep_factor = int(np.ceil(n_replicas ** (1/3)))

    # Replicate
    all_positions = []
    all_atomic_numbers = []

    for ix in range(rep_factor):
        for iy in range(rep_factor):
            for iz in range(rep_factor):
                shift = torch.tensor([ix, iy, iz], device="cuda", dtype=torch.float32) @ base_cell
                all_positions.append(base_positions + shift)
                all_atomic_numbers.append(base_atomic_numbers)

    positions = torch.cat(all_positions, dim=0)[:target_N]
    atomic_numbers = torch.cat(all_atomic_numbers, dim=0)[:target_N]

    # Scale cell
    cell = base_cell * rep_factor

    return positions, cell, atomic_numbers


# =============================================================================
# Batteries Implementation (Custom CUDA kernels)
# =============================================================================

class BatteriesNeighborList:
    """Neighbor list using batteries CUDA kernel with PBC."""

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def __call__(self, positions: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute neighbor list with PBC.

        Returns:
            edge_index: [2, num_edges] source and destination indices
            edge_vectors: [num_edges, 3] displacement vectors (j - i wrapped)
        """
        N = positions.shape[0]

        result = NeighborListResult()
        err = _neighbor_lib.neighbor_list_cuda(
            ctypes.c_void_p(positions.data_ptr()),
            ctypes.c_void_p(cell.data_ptr()),
            ctypes.c_int64(N),
            ctypes.c_float(self.cutoff),
            ctypes.c_bool(False),  # no self-loops
            ctypes.byref(result),
            None  # default stream
        )
        torch.cuda.synchronize()

        if err != 0:
            raise RuntimeError(f"CUDA error {err}")

        num_edges = result.num_edges

        if num_edges == 0:
            _neighbor_lib.neighbor_list_free(ctypes.byref(result))
            return (
                torch.zeros(2, 0, device="cuda", dtype=torch.long),
                torch.zeros(0, 3, device="cuda", dtype=torch.float32)
            )

        # Create tensors from device pointers
        # edge_index is stored as [src..., dst...] contiguous
        edge_index = torch.zeros(2, num_edges, device="cuda", dtype=torch.int32)
        edge_vectors = torch.zeros(num_edges, 3, device="cuda", dtype=torch.float32)

        # Copy from device memory
        edge_index_ptr = ctypes.cast(result.d_edge_index, ctypes.c_void_p).value
        edge_vectors_ptr = ctypes.cast(result.d_edge_vectors, ctypes.c_void_p).value

        # Use torch's from_dlpack or manual copy
        src_tensor = torch.frombuffer(
            (ctypes.c_int32 * (2 * num_edges)).from_address(edge_index_ptr),
            dtype=torch.int32
        ).cuda()

        # Actually, we need to use cudaMemcpy. Let's use a simpler approach:
        # Create tensors that share memory with the result
        _edge_src = torch.empty(num_edges, device="cuda", dtype=torch.int32)
        _edge_dst = torch.empty(num_edges, device="cuda", dtype=torch.int32)

        # The result stores edge_index as [src0, src1, ..., dst0, dst1, ...]
        # We need to copy this data
        import torch.cuda as cuda_module

        # Use ctypes to call cudaMemcpy
        cuda_rt = ctypes.CDLL("libcudart.so")
        cuda_rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

        # cudaMemcpyDeviceToDevice = 3
        cuda_rt.cudaMemcpy(
            ctypes.c_void_p(_edge_src.data_ptr()),
            ctypes.c_void_p(edge_index_ptr),
            num_edges * 4,  # int32 = 4 bytes
            3
        )
        cuda_rt.cudaMemcpy(
            ctypes.c_void_p(_edge_dst.data_ptr()),
            ctypes.c_void_p(edge_index_ptr + num_edges * 4),
            num_edges * 4,
            3
        )
        cuda_rt.cudaMemcpy(
            ctypes.c_void_p(edge_vectors.data_ptr()),
            ctypes.c_void_p(edge_vectors_ptr),
            num_edges * 3 * 4,  # float32 * 3
            3
        )

        edge_index = torch.stack([_edge_src.long(), _edge_dst.long()], dim=0)

        _neighbor_lib.neighbor_list_free(ctypes.byref(result))

        return edge_index, edge_vectors


class BatteriesInteractionBlock(nn.Module):
    """
    MACE-style interaction block using batteries CUDA kernels.

    Forward pass:
    1. Compute edge spherical harmonics Y_lm from edge vectors
    2. Tensor product of node features with Y_lm (message)
    3. Scatter-add messages to nodes
    4. Linear layer to mix channels
    """

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.hidden = config.hidden_channels
        self.num_irreps = (config.L_max + 1) ** 2  # 16 for L_max=3

        num_paths = _batteries.get_tp_num_paths()

        # Learnable weights for tensor product
        self.tp_weights = nn.Parameter(
            torch.randn(num_paths, self.hidden, self.hidden) * 0.01
        )

        # Output linear layer (channel mixing)
        self.linear = nn.Linear(self.hidden * self.num_irreps, self.hidden * self.num_irreps)

    def forward(
        self,
        node_features: torch.Tensor,  # [N, hidden, 16]
        edge_index: torch.Tensor,     # [2, E]
        edge_vectors: torch.Tensor,   # [E, 3]
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, hidden, 16] node features with irreps
            edge_index: [2, E] source and destination indices
            edge_vectors: [E, 3] edge displacement vectors

        Returns:
            new_node_features: [N, hidden, 16]
        """
        src, dst = edge_index[0], edge_index[1]
        E = edge_vectors.shape[0]
        N = node_features.shape[0]

        # 1. Compute spherical harmonics for edges
        # Normalize edge vectors
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths

        # Y_lm: [E, 16]
        Y_lm = batteries_spherical_harmonics(edge_unit, self.config.L_max)

        # 2. Gather source node features for each edge
        # node_src: [E, hidden, 16]
        node_src = node_features[src]

        # 3. Tensor product: node_src x Y_lm -> messages
        # messages: [E, hidden, 16]
        messages = batteries_tensor_product(node_src, Y_lm, self.tp_weights,
                                            c_in=self.hidden, c_out=self.hidden)

        # 4. Scatter-add messages to destination nodes
        # Reshape for scatter: [E, hidden*16]
        messages_flat = messages.view(E, -1)

        # Scatter to nodes
        out_flat = torch.zeros(N, self.hidden * self.num_irreps, device=node_features.device)
        torch_scatter.scatter_add(messages_flat, dst, dim=0, out=out_flat)

        # 5. Linear layer
        out_flat = self.linear(out_flat)

        # Reshape back
        return out_flat.view(N, self.hidden, self.num_irreps)


class BatteriesMACE(nn.Module):
    """
    Simplified MACE model using batteries CUDA kernels.

    Architecture:
    - Embedding layer (atomic numbers -> features)
    - N interaction layers
    - Output head (sum over atoms)
    """

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.num_irreps = (config.L_max + 1) ** 2

        # Atom embedding
        self.embedding = nn.Embedding(config.num_elements, config.hidden_channels * self.num_irreps)

        # Interaction layers
        self.interactions = nn.ModuleList([
            BatteriesInteractionBlock(config) for _ in range(config.num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(config.hidden_channels * self.num_irreps, 1)

        # Neighbor list
        self.neighbor_list = BatteriesNeighborList(config.cutoff)

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            positions: [N, 3] atom positions
            cell: [3, 3] cell matrix
            atomic_numbers: [N] atomic numbers

        Returns:
            energy: scalar energy prediction
        """
        N = positions.shape[0]

        # Compute neighbor list (indices only, edge_vectors are not gradient-connected)
        edge_index, _ = self.neighbor_list(positions.detach(), cell)

        # Recompute edge_vectors from positions to preserve autograd graph
        # This is needed for force computation via backward pass
        src, dst = edge_index[0], edge_index[1]
        edge_vectors = positions[dst] - positions[src]  # [E, 3]

        # Apply minimum image convention for PBC
        # cell is [3, 3], for orthorhombic: diag = [cell[0,0], cell[1,1], cell[2,2]]
        cell_diag = torch.diagonal(cell)  # [3]
        inv_cell_diag = 1.0 / cell_diag
        edge_vectors = edge_vectors - torch.round(edge_vectors * inv_cell_diag) * cell_diag

        # Embed atoms
        # [N, hidden * 16] -> [N, hidden, 16]
        node_features = self.embedding(atomic_numbers).view(N, self.config.hidden_channels, self.num_irreps)

        # Interaction layers
        for interaction in self.interactions:
            node_features = node_features + interaction(node_features, edge_index, edge_vectors)

        # Output
        node_features_flat = node_features.view(N, -1)
        node_energies = self.output_head(node_features_flat)  # [N, 1]
        total_energy = node_energies.sum()

        return total_energy


# =============================================================================
# e3nn/MACE Baseline Implementation
# =============================================================================

class E3nnInteractionBlock(nn.Module):
    """
    MACE-style interaction block using e3nn.
    """

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.hidden = config.hidden_channels

        # Define irreps
        self.irreps_node = o3.Irreps(f"{config.hidden_channels}x0e + {config.hidden_channels}x1o + {config.hidden_channels}x2e + {config.hidden_channels}x3o")
        self.irreps_edge = o3.Irreps("0e + 1o + 2e + 3o")

        # Spherical harmonics
        self.sh = o3.SphericalHarmonics(
            irreps_out=self.irreps_edge,
            normalize=True,
            normalization="component"
        )

        # Tensor product
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_edge,
            self.irreps_node,
            shared_weights=False
        )

        # Weight MLP (simplified - just a linear layer)
        self.weight_net = nn.Linear(1, self.tp.weight_numel)

    def forward(
        self,
        node_features: torch.Tensor,  # [N, irreps_node.dim]
        edge_index: torch.Tensor,     # [2, E]
        edge_vectors: torch.Tensor,   # [E, 3]
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        E = edge_vectors.shape[0]
        N = node_features.shape[0]

        # Normalize edge vectors
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths

        # Spherical harmonics
        Y_lm = self.sh(edge_unit)  # [E, 16]

        # Generate weights from edge lengths
        weights = self.weight_net(edge_lengths)  # [E, weight_numel]

        # Gather source features
        node_src = node_features[src]  # [E, irreps_dim]

        # Tensor product
        messages = self.tp(node_src, Y_lm, weights)  # [E, irreps_dim]

        # Scatter
        out = torch.zeros(N, node_features.shape[1], device=node_features.device)
        torch_scatter.scatter_add(messages, dst, dim=0, out=out)

        return out


class E3nnMACE(nn.Module):
    """
    Simplified MACE model using e3nn.
    """

    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config

        # Define irreps
        irreps_dim = config.hidden_channels * 16  # 1 + 3 + 5 + 7 = 16 for L_max=3
        self.irreps_node = o3.Irreps(f"{config.hidden_channels}x0e + {config.hidden_channels}x1o + {config.hidden_channels}x2e + {config.hidden_channels}x3o")

        # Embedding
        self.embedding = nn.Embedding(config.num_elements, self.irreps_node.dim)

        # Interaction layers
        self.interactions = nn.ModuleList([
            E3nnInteractionBlock(config) for _ in range(config.num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(self.irreps_node.dim, 1)

    def forward(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        N = positions.shape[0]

        # Compute neighbor list using torch_cluster (no PBC)
        edge_index = torch_cluster.radius_graph(
            positions, r=self.config.cutoff,
            loop=False, max_num_neighbors=128
        )

        # Compute edge vectors
        src, dst = edge_index[0], edge_index[1]
        edge_vectors = positions[dst] - positions[src]

        # Embed atoms
        node_features = self.embedding(atomic_numbers)  # [N, irreps_dim]

        # Interaction layers
        for interaction in self.interactions:
            node_features = node_features + interaction(node_features, edge_index, edge_vectors)

        # Output
        node_energies = self.output_head(node_features)
        total_energy = node_energies.sum()

        return total_energy


# =============================================================================
# Timing Utilities
# =============================================================================

class CUDATimer:
    """GPU event-based timer for accurate CUDA timing."""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


@dataclass
class TimingResult:
    """Stores timing results for a single benchmark run."""
    total_ms: float
    forward_ms: Optional[float] = None
    backward_ms: Optional[float] = None
    neighbor_list_ms: Optional[float] = None
    spherical_harmonics_ms: Optional[float] = None
    tensor_product_ms: Optional[float] = None
    scatter_ms: Optional[float] = None
    linear_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    num_edges: Optional[int] = None


def measure_peak_memory() -> float:
    """Returns peak GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_batteries_forward(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark batteries implementation with detailed timing breakdown.
    """
    model = BatteriesMACE(config).cuda()
    model.eval()

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(positions, cell, atomic_numbers)
    torch.cuda.synchronize()

    # Benchmark total forward pass
    timer = CUDATimer()
    times = []

    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            _ = model(positions, cell, atomic_numbers)
        elapsed = timer.stop()
        times.append(elapsed)

    avg_ms = np.mean(times)
    memory_mb = measure_peak_memory()

    # Get edge count from last run
    edge_index, _ = model.neighbor_list(positions, cell)
    num_edges = edge_index.shape[1]

    return TimingResult(
        total_ms=avg_ms,
        memory_mb=memory_mb,
        num_edges=num_edges
    )


def benchmark_batteries_breakdown(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark batteries implementation with per-operation timing.
    """
    N = positions.shape[0]
    hidden = config.hidden_channels
    num_irreps = (config.L_max + 1) ** 2
    num_paths = _batteries.get_tp_num_paths()

    # Setup
    neighbor_list = BatteriesNeighborList(config.cutoff)
    embedding = nn.Embedding(config.num_elements, hidden * num_irreps).cuda()
    tp_weights = torch.randn(num_paths, hidden, hidden, device="cuda") * 0.01
    linear = nn.Linear(hidden * num_irreps, hidden * num_irreps).cuda()

    # Warmup
    for _ in range(warmup):
        edge_index, edge_vectors = neighbor_list(positions, cell)
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths
        Y_lm = _batteries.spherical_harmonics(edge_unit, config.L_max)
        node_features = embedding(atomic_numbers).view(N, hidden, num_irreps)
        node_src = node_features[edge_index[0]]
        messages = _batteries.tensor_product(node_src, Y_lm, tp_weights)
        messages_flat = messages.view(-1, hidden * num_irreps)
        out = torch.zeros(N, hidden * num_irreps, device="cuda")
        torch_scatter.scatter_add(messages_flat, edge_index[1], dim=0, out=out)
        out = linear(out)
    torch.cuda.synchronize()

    timer = CUDATimer()

    # Time each operation
    nl_times, sh_times, tp_times, scatter_times, linear_times = [], [], [], [], []

    for _ in range(iterations):
        # Neighbor list
        timer.start()
        edge_index, edge_vectors = neighbor_list(positions, cell)
        nl_times.append(timer.stop())

        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths

        # Spherical harmonics
        timer.start()
        Y_lm = _batteries.spherical_harmonics(edge_unit, config.L_max)
        sh_times.append(timer.stop())

        node_features = embedding(atomic_numbers).view(N, hidden, num_irreps)
        node_src = node_features[edge_index[0]]

        # Tensor product
        timer.start()
        messages = _batteries.tensor_product(node_src, Y_lm, tp_weights)
        tp_times.append(timer.stop())

        messages_flat = messages.view(-1, hidden * num_irreps)

        # Scatter
        timer.start()
        out = torch.zeros(N, hidden * num_irreps, device="cuda")
        torch_scatter.scatter_add(messages_flat, edge_index[1], dim=0, out=out)
        scatter_times.append(timer.stop())

        # Linear
        timer.start()
        out = linear(out)
        linear_times.append(timer.stop())

    return TimingResult(
        total_ms=np.mean(nl_times) + np.mean(sh_times) + np.mean(tp_times) + np.mean(scatter_times) + np.mean(linear_times),
        neighbor_list_ms=np.mean(nl_times),
        spherical_harmonics_ms=np.mean(sh_times),
        tensor_product_ms=np.mean(tp_times),
        scatter_ms=np.mean(scatter_times),
        linear_ms=np.mean(linear_times),
        num_edges=edge_index.shape[1],
        memory_mb=measure_peak_memory()
    )


def benchmark_e3nn_forward(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark e3nn/MACE implementation.
    """
    model = E3nnMACE(config).cuda()
    model.eval()

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(positions, cell, atomic_numbers)
    torch.cuda.synchronize()

    # Benchmark
    timer = CUDATimer()
    times = []

    for _ in range(iterations):
        timer.start()
        with torch.no_grad():
            _ = model(positions, cell, atomic_numbers)
        elapsed = timer.stop()
        times.append(elapsed)

    avg_ms = np.mean(times)
    memory_mb = measure_peak_memory()

    # Get edge count
    edge_index = torch_cluster.radius_graph(positions, r=config.cutoff, loop=False, max_num_neighbors=128)
    num_edges = edge_index.shape[1]

    return TimingResult(
        total_ms=avg_ms,
        memory_mb=memory_mb,
        num_edges=num_edges
    )


def benchmark_e3nn_breakdown(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark e3nn implementation with per-operation timing.
    """
    N = positions.shape[0]
    hidden = config.hidden_channels

    # Setup
    irreps_node = o3.Irreps(f"{hidden}x0e + {hidden}x1o + {hidden}x2e + {hidden}x3o")
    irreps_edge = o3.Irreps("0e + 1o + 2e + 3o")

    embedding = nn.Embedding(config.num_elements, irreps_node.dim).cuda()
    sh = o3.SphericalHarmonics(irreps_out=irreps_edge, normalize=True, normalization="component")
    tp = o3.FullyConnectedTensorProduct(irreps_node, irreps_edge, irreps_node, shared_weights=False).cuda()
    weight_net = nn.Linear(1, tp.weight_numel).cuda()
    linear = nn.Linear(irreps_node.dim, irreps_node.dim).cuda()

    # Warmup
    for _ in range(warmup):
        edge_index = torch_cluster.radius_graph(positions, r=config.cutoff, loop=False, max_num_neighbors=128)
        src, dst = edge_index[0], edge_index[1]
        edge_vectors = positions[dst] - positions[src]
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths
        Y_lm = sh(edge_unit)
        weights = weight_net(edge_lengths)
        node_features = embedding(atomic_numbers)
        node_src = node_features[src]
        messages = tp(node_src, Y_lm, weights)
        out = torch.zeros(N, irreps_node.dim, device="cuda")
        torch_scatter.scatter_add(messages, dst, dim=0, out=out)
        out = linear(out)
    torch.cuda.synchronize()

    timer = CUDATimer()
    nl_times, sh_times, tp_times, scatter_times, linear_times = [], [], [], [], []

    for _ in range(iterations):
        # Neighbor list (torch_cluster)
        timer.start()
        edge_index = torch_cluster.radius_graph(positions, r=config.cutoff, loop=False, max_num_neighbors=128)
        nl_times.append(timer.stop())

        src, dst = edge_index[0], edge_index[1]
        edge_vectors = positions[dst] - positions[src]
        edge_lengths = torch.norm(edge_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
        edge_unit = edge_vectors / edge_lengths

        # Spherical harmonics (e3nn)
        timer.start()
        Y_lm = sh(edge_unit)
        sh_times.append(timer.stop())

        weights = weight_net(edge_lengths)
        node_features = embedding(atomic_numbers)
        node_src = node_features[src]

        # Tensor product (e3nn)
        timer.start()
        messages = tp(node_src, Y_lm, weights)
        tp_times.append(timer.stop())

        # Scatter
        timer.start()
        out = torch.zeros(N, irreps_node.dim, device="cuda")
        torch_scatter.scatter_add(messages, dst, dim=0, out=out)
        scatter_times.append(timer.stop())

        # Linear
        timer.start()
        out = linear(out)
        linear_times.append(timer.stop())

    return TimingResult(
        total_ms=np.mean(nl_times) + np.mean(sh_times) + np.mean(tp_times) + np.mean(scatter_times) + np.mean(linear_times),
        neighbor_list_ms=np.mean(nl_times),
        spherical_harmonics_ms=np.mean(sh_times),
        tensor_product_ms=np.mean(tp_times),
        scatter_ms=np.mean(scatter_times),
        linear_ms=np.mean(linear_times),
        num_edges=edge_index.shape[1],
        memory_mb=measure_peak_memory()
    )


# =============================================================================
# Force Computation Benchmarks (Forward + Backward)
# =============================================================================

def benchmark_batteries_forces(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark batteries implementation with force computation.
    Forces = -dE/dR via autograd backward pass.
    """
    model = BatteriesMACE(config).cuda()
    model.eval()

    # Enable gradient tracking on positions
    positions = positions.clone().detach().requires_grad_(True)

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        if positions.grad is not None:
            positions.grad.zero_()
        energy = model(positions, cell, atomic_numbers)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
    torch.cuda.synchronize()

    timer = CUDATimer()
    forward_times = []
    backward_times = []

    for _ in range(iterations):
        if positions.grad is not None:
            positions.grad.zero_()

        # Forward pass
        timer.start()
        energy = model(positions, cell, atomic_numbers)
        fwd_time = timer.stop()
        forward_times.append(fwd_time)

        # Backward pass (force computation)
        timer.start()
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        bwd_time = timer.stop()
        backward_times.append(bwd_time)

    avg_fwd = np.mean(forward_times)
    avg_bwd = np.mean(backward_times)
    memory_mb = measure_peak_memory()

    # Get edge count
    edge_index, _ = model.neighbor_list(positions.detach(), cell)
    num_edges = edge_index.shape[1]

    return TimingResult(
        total_ms=avg_fwd + avg_bwd,
        forward_ms=avg_fwd,
        backward_ms=avg_bwd,
        memory_mb=memory_mb,
        num_edges=num_edges
    )


def benchmark_e3nn_forces(
    positions: torch.Tensor,
    cell: torch.Tensor,
    atomic_numbers: torch.Tensor,
    config: MACEConfig,
    warmup: int = 5,
    iterations: int = 20,
) -> TimingResult:
    """
    Benchmark e3nn implementation with force computation.
    Forces = -dE/dR via autograd backward pass.
    """
    model = E3nnMACE(config).cuda()
    model.eval()

    # Enable gradient tracking on positions
    positions = positions.clone().detach().requires_grad_(True)

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        if positions.grad is not None:
            positions.grad.zero_()
        energy = model(positions, cell, atomic_numbers)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
    torch.cuda.synchronize()

    timer = CUDATimer()
    forward_times = []
    backward_times = []

    for _ in range(iterations):
        if positions.grad is not None:
            positions.grad.zero_()

        # Forward pass
        timer.start()
        energy = model(positions, cell, atomic_numbers)
        fwd_time = timer.stop()
        forward_times.append(fwd_time)

        # Backward pass (force computation)
        timer.start()
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        bwd_time = timer.stop()
        backward_times.append(bwd_time)

    avg_fwd = np.mean(forward_times)
    avg_bwd = np.mean(backward_times)
    memory_mb = measure_peak_memory()

    # Get edge count
    edge_index = torch_cluster.radius_graph(
        positions.detach(), r=config.cutoff, loop=False, max_num_neighbors=128
    )
    num_edges = edge_index.shape[1]

    return TimingResult(
        total_ms=avg_fwd + avg_bwd,
        forward_ms=avg_fwd,
        backward_ms=avg_bwd,
        memory_mb=memory_mb,
        num_edges=num_edges
    )


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    print("=" * 70)
    print("End-to-End MACE Benchmark: batteries vs e3nn")
    print("=" * 70)
    print()

    # Configuration
    config = MACEConfig(
        L_max=3,
        hidden_channels=32,
        num_layers=2,
        cutoff=5.0,
    )

    print(f"Model Configuration:")
    print(f"  L_max: {config.L_max}")
    print(f"  hidden_channels: {config.hidden_channels}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  cutoff: {config.cutoff} A")
    print(f"  num_paths (tensor product): {_batteries.get_tp_num_paths()}")
    print()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Test sizes
    test_sizes = [200, 500, 1000, 2000]

    forward_results = []
    force_results = []

    for N in test_sizes:
        print(f"{'='*70}")
        print(f"N = {N:,} atoms")
        print(f"{'='*70}")

        # Generate test system
        positions, cell, atomic_numbers = generate_amorphous_system(N, density=0.05)
        box_size = cell[0, 0].item()
        print(f"Box size: {box_size:.2f} A, density: 0.05 atoms/A^3")
        print()

        # =====================================================================
        # Forward-only benchmarks (inference)
        # =====================================================================
        print("--- Forward Pass (Energy Only) ---")
        print()

        # Benchmark batteries
        print("batteries (CUDA kernels):")
        try:
            bat_result = benchmark_batteries_breakdown(positions, cell, atomic_numbers, config)
            print(f"  Total forward:        {bat_result.total_ms:>8.3f} ms")
            print(f"    - neighbor_list:    {bat_result.neighbor_list_ms:>8.3f} ms")
            print(f"    - spherical_harm:   {bat_result.spherical_harmonics_ms:>8.3f} ms")
            print(f"    - tensor_product:   {bat_result.tensor_product_ms:>8.3f} ms")
            print(f"    - scatter_add:      {bat_result.scatter_ms:>8.3f} ms")
            print(f"    - linear:           {bat_result.linear_ms:>8.3f} ms")
            print(f"  Edges: {bat_result.num_edges:,}, Memory: {bat_result.memory_mb:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            bat_result = None
        print()

        # Benchmark e3nn
        print("e3nn/MACE baseline:")
        try:
            e3nn_result = benchmark_e3nn_breakdown(positions, cell, atomic_numbers, config)
            print(f"  Total forward:        {e3nn_result.total_ms:>8.3f} ms")
            print(f"    - neighbor_list:    {e3nn_result.neighbor_list_ms:>8.3f} ms  (torch_cluster, no PBC)")
            print(f"    - spherical_harm:   {e3nn_result.spherical_harmonics_ms:>8.3f} ms")
            print(f"    - tensor_product:   {e3nn_result.tensor_product_ms:>8.3f} ms")
            print(f"    - scatter_add:      {e3nn_result.scatter_ms:>8.3f} ms")
            print(f"    - linear:           {e3nn_result.linear_ms:>8.3f} ms")
            print(f"  Edges: {e3nn_result.num_edges:,}, Memory: {e3nn_result.memory_mb:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            e3nn_result = None
        print()

        # Forward speedup comparison
        if bat_result and e3nn_result:
            speedup = e3nn_result.total_ms / bat_result.total_ms
            print(f"Forward Speedup: {speedup:.2f}x faster")

            if bat_result.tensor_product_ms and e3nn_result.tensor_product_ms:
                tp_speedup = e3nn_result.tensor_product_ms / bat_result.tensor_product_ms
                print(f"Tensor Product Speedup: {tp_speedup:.2f}x faster")

            forward_results.append({
                "N": N,
                "batteries_ms": bat_result.total_ms,
                "e3nn_ms": e3nn_result.total_ms,
                "speedup": speedup,
                "bat_edges": bat_result.num_edges,
                "e3nn_edges": e3nn_result.num_edges,
            })
        print()

        # =====================================================================
        # Force computation benchmarks (forward + backward)
        # =====================================================================
        print("--- Force Computation (Forward + Backward) ---")
        print()

        # Benchmark batteries forces
        print("batteries (CUDA kernels):")
        try:
            bat_force = benchmark_batteries_forces(positions, cell, atomic_numbers, config)
            bwd_fwd_ratio = bat_force.backward_ms / bat_force.forward_ms
            print(f"  Forward:              {bat_force.forward_ms:>8.3f} ms")
            print(f"  Backward (forces):    {bat_force.backward_ms:>8.3f} ms")
            print(f"  Total:                {bat_force.total_ms:>8.3f} ms")
            print(f"  Backward/Forward:     {bwd_fwd_ratio:>8.2f}x")
            print(f"  Memory: {bat_force.memory_mb:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            bat_force = None
        print()

        # Benchmark e3nn forces
        print("e3nn/MACE baseline:")
        try:
            e3nn_force = benchmark_e3nn_forces(positions, cell, atomic_numbers, config)
            bwd_fwd_ratio = e3nn_force.backward_ms / e3nn_force.forward_ms
            print(f"  Forward:              {e3nn_force.forward_ms:>8.3f} ms")
            print(f"  Backward (forces):    {e3nn_force.backward_ms:>8.3f} ms")
            print(f"  Total:                {e3nn_force.total_ms:>8.3f} ms")
            print(f"  Backward/Forward:     {bwd_fwd_ratio:>8.2f}x")
            print(f"  Memory: {e3nn_force.memory_mb:.1f} MB")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            e3nn_force = None
        print()

        # Force speedup comparison
        if bat_force and e3nn_force:
            fwd_speedup = e3nn_force.forward_ms / bat_force.forward_ms
            bwd_speedup = e3nn_force.backward_ms / bat_force.backward_ms
            total_speedup = e3nn_force.total_ms / bat_force.total_ms
            print(f"Speedup (forward):  {fwd_speedup:.2f}x faster")
            print(f"Speedup (backward): {bwd_speedup:.2f}x faster")
            print(f"Speedup (total):    {total_speedup:.2f}x faster")

            force_results.append({
                "N": N,
                "bat_fwd": bat_force.forward_ms,
                "bat_bwd": bat_force.backward_ms,
                "bat_total": bat_force.total_ms,
                "bat_ratio": bat_force.backward_ms / bat_force.forward_ms,
                "e3nn_fwd": e3nn_force.forward_ms,
                "e3nn_bwd": e3nn_force.backward_ms,
                "e3nn_total": e3nn_force.total_ms,
                "e3nn_ratio": e3nn_force.backward_ms / e3nn_force.forward_ms,
                "fwd_speedup": fwd_speedup,
                "bwd_speedup": bwd_speedup,
                "total_speedup": total_speedup,
            })
        print()

    # =========================================================================
    # Summary Tables
    # =========================================================================

    # Forward-only summary
    if forward_results:
        print("=" * 70)
        print("Summary: Forward Pass (Energy Only)")
        print("=" * 70)
        print(f"{'N':>8} | {'batteries':>12} | {'e3nn':>12} | {'Speedup':>10}")
        print("-" * 50)
        for r in forward_results:
            print(f"{r['N']:>8,} | {r['batteries_ms']:>9.3f} ms | {r['e3nn_ms']:>9.3f} ms | {r['speedup']:>9.2f}x")
        print()

    # Force computation summary
    if force_results:
        print("=" * 70)
        print("Summary: Force Computation (Forward + Backward)")
        print("=" * 70)
        print(f"{'N':>6} | {'bat_fwd':>8} | {'bat_bwd':>8} | {'bat_B/F':>7} | {'e3nn_fwd':>9} | {'e3nn_bwd':>9} | {'e3nn_B/F':>8} | {'Speedup':>8}")
        print("-" * 90)
        for r in force_results:
            print(f"{r['N']:>6,} | {r['bat_fwd']:>5.2f} ms | {r['bat_bwd']:>5.2f} ms | {r['bat_ratio']:>6.2f}x | {r['e3nn_fwd']:>6.2f} ms | {r['e3nn_bwd']:>6.2f} ms | {r['e3nn_ratio']:>7.2f}x | {r['total_speedup']:>7.2f}x")

        print()
        print("Detailed Force Speedups:")
        print(f"{'N':>8} | {'Forward':>12} | {'Backward':>12} | {'Total':>12}")
        print("-" * 50)
        for r in force_results:
            print(f"{r['N']:>8,} | {r['fwd_speedup']:>10.2f}x | {r['bwd_speedup']:>10.2f}x | {r['total_speedup']:>10.2f}x")

    print()
    print("Notes:")
    print("  - batteries uses periodic boundary conditions (PBC)")
    print("  - e3nn/torch_cluster does NOT use PBC (hence fewer edges)")
    print("  - Backward/Forward ratio: typically 2-3x for e3nn")
    print("  - Forces computed via: forces = -torch.autograd.grad(energy.sum(), positions)")


if __name__ == "__main__":
    main()
