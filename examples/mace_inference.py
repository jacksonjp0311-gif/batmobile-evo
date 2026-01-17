#!/usr/bin/env python3
"""
MACE-style inference example using Batteries kernels.

Demonstrates:
1. Loading a molecular structure with ASE
2. Building neighbor list
3. Running equivariant message passing forward pass
4. Computing forces via autograd

This is a simplified MACE layer - real MACE has more components
(radial basis, ACE-style symmetric contractions, etc.)
"""

import torch
import numpy as np

# Optional: ASE for loading structures
try:
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("ASE not installed. Using synthetic data.")

import batteries
from batteries.autograd import SphericalHarmonicsFunction


def create_water_molecule():
    """Create a simple water molecule for testing."""
    if HAS_ASE:
        # H2O geometry (Angstroms)
        positions = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.757, 0.586, 0.0],  # H
            [-0.757, 0.586, 0.0], # H
        ])
        atoms = Atoms('OH2', positions=positions)
        return atoms
    else:
        return None


def create_bulk_silicon(n_repeat=2):
    """Create bulk silicon for larger-scale testing."""
    if HAS_ASE:
        from ase.build import bulk
        si = bulk('Si', 'diamond', a=5.43)
        si = si.repeat(n_repeat)
        return si
    else:
        return None


def build_edge_index(atoms, cutoff=5.0):
    """Build edge index from ASE Atoms using neighbor list."""
    if HAS_ASE:
        # Get neighbor list
        i, j, d = neighbor_list('ijd', atoms, cutoff)

        # Edge index: [2, N_edges] with [sources, targets]
        edge_index = np.stack([j, i], axis=0)  # j -> i (source -> target)

        # Edge vectors (target - source, normalized)
        positions = atoms.get_positions()
        edge_vectors = positions[i] - positions[j]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
        edge_lengths = np.clip(edge_lengths, 1e-10, None)
        edge_vectors = edge_vectors / edge_lengths

        return edge_index, edge_vectors, edge_lengths.squeeze()
    else:
        return None, None, None


def create_synthetic_graph(n_atoms=100, n_edges_per_atom=20):
    """Create synthetic graph data for testing without ASE."""
    n_edges = n_atoms * n_edges_per_atom

    # Random edge index
    sources = np.random.randint(0, n_atoms, n_edges)
    targets = np.random.randint(0, n_atoms, n_edges)
    # Remove self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]

    edge_index = np.stack([sources, targets], axis=0)
    n_edges = edge_index.shape[1]

    # Random unit edge vectors
    edge_vectors = np.random.randn(n_edges, 3)
    edge_vectors = edge_vectors / np.linalg.norm(edge_vectors, axis=1, keepdims=True)

    # Random edge lengths
    edge_lengths = np.random.uniform(1.0, 5.0, n_edges)

    return edge_index, edge_vectors, edge_lengths


class SimpleMACELayer(torch.nn.Module):
    """
    Simplified MACE-style message passing layer using Batteries kernels.

    Real MACE has additional components:
    - Radial basis functions
    - ACE-style symmetric contractions
    - Multiple interaction blocks
    - Equivariant MLPs

    This demo shows the core SH + TP + scatter pattern.
    """

    def __init__(self, hidden_channels=32, out_channels=32, lmax=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.num_irreps = (lmax + 1) ** 2  # 16 for lmax=3

        # Number of CG tensor product paths
        self.num_paths = batteries.get_tp_num_paths()  # 34 for lmax=3

        # Learnable weights for tensor product
        self.tp_weights = torch.nn.Parameter(
            torch.randn(self.num_paths, hidden_channels, out_channels) / (hidden_channels ** 0.5)
        )

        # Simple linear for output (in real MACE this would be equivariant)
        self.output_linear = torch.nn.Linear(out_channels * self.num_irreps, out_channels)

    def forward(self, node_features, edge_vectors, edge_index):
        """
        Forward pass of MACE-style message passing.

        Args:
            node_features: [N_atoms, hidden_channels, 16] - node embeddings
            edge_vectors: [N_edges, 3] - unit edge direction vectors
            edge_index: [2, N_edges] - [source, target] pairs

        Returns:
            node_output: [N_atoms, out_channels] - updated node embeddings
        """
        n_atoms = node_features.size(0)
        n_edges = edge_vectors.size(0)

        sources = edge_index[0]
        targets = edge_index[1]

        # 1. Compute spherical harmonics for edge directions
        # Using autograd-enabled version for force computation
        Y_lm = SphericalHarmonicsFunction.apply(edge_vectors, self.lmax, False)

        # 2. Gather source node features
        src_features = node_features[sources]  # [N_edges, C_in, 16]

        # 3. Compute tensor product with learnable weights
        messages = batteries.tensor_product(src_features, Y_lm, self.tp_weights)
        # messages: [N_edges, C_out, 16]

        # 4. Scatter add messages to target nodes
        node_output = torch.zeros(
            n_atoms, self.out_channels, self.num_irreps,
            device=node_features.device, dtype=node_features.dtype
        )
        node_output.index_add_(0, targets, messages)

        # 5. Simple readout (flatten and linear)
        # In real MACE this would preserve equivariance
        node_output = node_output.reshape(n_atoms, -1)
        node_output = self.output_linear(node_output)

        return node_output


class SimpleMACELayerFused(torch.nn.Module):
    """
    Same as SimpleMACELayer but using fused SH+TP kernel.

    This eliminates the intermediate Y_lm tensor from global memory.
    """

    def __init__(self, hidden_channels=32, out_channels=32, lmax=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lmax = lmax
        self.num_irreps = (lmax + 1) ** 2

        # Simple linear for output
        self.output_linear = torch.nn.Linear(hidden_channels * self.num_irreps, out_channels)

    def forward(self, node_features, edge_vectors, edge_index):
        """
        Forward pass using fused SH+TP kernel.

        Note: fused_sh_tp_simple doesn't support learnable weights yet,
        so this is a simpler version without the weighted tensor product.
        """
        n_atoms = node_features.size(0)

        sources = edge_index[0]
        targets = edge_index[1]

        # 1. Fused SH + TP (no weights)
        messages = batteries.fused_sh_tp_simple(edge_vectors, node_features, sources)
        # messages: [N_edges, C, 16]

        # 2. Scatter add messages to target nodes
        node_output = torch.zeros(
            n_atoms, self.hidden_channels, self.num_irreps,
            device=node_features.device, dtype=node_features.dtype
        )
        node_output.index_add_(0, targets, messages)

        # 3. Simple readout
        node_output = node_output.reshape(n_atoms, -1)
        node_output = self.output_linear(node_output)

        return node_output


def compute_forces(model, node_features, edge_vectors, edge_index):
    """
    Compute forces as negative gradient of energy w.r.t. positions.

    Since edge_vectors = (pos[target] - pos[source]) / |...|,
    we get forces via the chain rule through the spherical harmonics.
    """
    # Enable gradients on edge vectors
    edge_vectors = edge_vectors.clone().requires_grad_(True)

    # Forward pass
    output = model(node_features, edge_vectors, edge_index)

    # Simple "energy" - sum of outputs (real MACE has proper readout)
    energy = output.sum()

    # Backward pass
    energy.backward()

    # Gradient w.r.t. edge vectors
    # This would need to be converted back to atom forces via the neighbor list
    grad_edge_vectors = edge_vectors.grad

    return energy.item(), grad_edge_vectors


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Create test structure
    if HAS_ASE:
        print("Creating bulk silicon structure...")
        atoms = create_bulk_silicon(n_repeat=2)
        n_atoms = len(atoms)
        edge_index, edge_vectors, edge_lengths = build_edge_index(atoms, cutoff=4.0)
        print(f"  N_atoms: {n_atoms}")
        print(f"  N_edges: {edge_index.shape[1]}")
    else:
        print("Creating synthetic graph (ASE not available)...")
        n_atoms = 64
        edge_index, edge_vectors, edge_lengths = create_synthetic_graph(n_atoms, 20)
        print(f"  N_atoms: {n_atoms}")
        print(f"  N_edges: {edge_index.shape[1]}")

    # Convert to torch tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_vectors = torch.tensor(edge_vectors, dtype=torch.float32, device=device)
    n_edges = edge_vectors.size(0)

    # Initialize node features (random for demo)
    hidden_channels = 32
    num_irreps = 16  # (3+1)^2
    node_features = torch.randn(n_atoms, hidden_channels, num_irreps, device=device)

    print()
    print("=" * 60)
    print("Running SimpleMACELayer (unfused)")
    print("=" * 60)

    # Create model
    model = SimpleMACELayer(hidden_channels=32, out_channels=32).to(device)

    # Warmup
    for _ in range(3):
        output = model(node_features, edge_vectors, edge_index)
    torch.cuda.synchronize()

    # Benchmark
    import time
    start = time.perf_counter()
    for _ in range(100):
        output = model(node_features, edge_vectors, edge_index)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000

    print(f"  Output shape: {output.shape}")
    print(f"  Forward time: {elapsed:.3f} ms")

    # Compute forces
    print()
    print("Computing forces via autograd...")
    energy, grad_edge = compute_forces(model, node_features, edge_vectors, edge_index)
    print(f"  Energy: {energy:.4f}")
    if grad_edge is not None:
        print(f"  Gradient shape: {grad_edge.shape}")
        print(f"  Gradient norm: {grad_edge.norm().item():.4f}")
    else:
        print("  Gradient: None (no gradient path to edge_vectors)")

    print()
    print("=" * 60)
    print("Running SimpleMACELayerFused")
    print("=" * 60)

    # Create fused model
    model_fused = SimpleMACELayerFused(hidden_channels=32, out_channels=32).to(device)

    # Warmup
    for _ in range(3):
        output_fused = model_fused(node_features, edge_vectors, edge_index)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        output_fused = model_fused(node_features, edge_vectors, edge_index)
    torch.cuda.synchronize()
    elapsed_fused = (time.perf_counter() - start) / 100 * 1000

    print(f"  Output shape: {output_fused.shape}")
    print(f"  Forward time: {elapsed_fused:.3f} ms")
    print(f"  Speedup vs unfused: {elapsed / elapsed_fused:.2f}x")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
