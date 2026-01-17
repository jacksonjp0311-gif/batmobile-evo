"""
Benchmark neighbor list: batteries CUDA vs torch_cluster.radius_graph

Compare performance for molecular simulation workloads:
- N = 1K, 10K, 50K atoms
- Density ~0.05 atoms/A^3 (typical liquid/amorphous)
- Cutoff = 5.0 Angstroms
"""

import torch
import torch_cluster
import ctypes
import numpy as np
import time
from pathlib import Path

# Load the CUDA library
lib_path = Path(__file__).parent.parent / "libbatteries_neighbor.so"
if not lib_path.exists():
    raise RuntimeError(f"Library not found: {lib_path}. Build with: nvcc -shared -O3 ...")

lib = ctypes.CDLL(str(lib_path))

# Define the result structure
class NeighborListResult(ctypes.Structure):
    _fields_ = [
        ("edge_index", ctypes.c_void_p),
        ("edge_vectors", ctypes.c_void_p),
        ("num_edges", ctypes.c_int64),
        ("d_edge_index", ctypes.c_void_p),
        ("d_edge_vectors", ctypes.c_void_p),
    ]

# Function signatures
lib.neighbor_list_cuda.argtypes = [
    ctypes.c_void_p,  # positions
    ctypes.c_void_p,  # cell
    ctypes.c_int64,   # N
    ctypes.c_float,   # cutoff
    ctypes.c_bool,    # self_loops
    ctypes.POINTER(NeighborListResult),  # result
    ctypes.c_void_p,  # stream
]
lib.neighbor_list_cuda.restype = ctypes.c_int

lib.neighbor_list_free.argtypes = [ctypes.POINTER(NeighborListResult)]
lib.neighbor_list_free.restype = None


def generate_positions(N: int, box_size: float, seed: int = 42) -> torch.Tensor:
    """Generate random positions in a box."""
    torch.manual_seed(seed)
    return torch.rand(N, 3, device="cuda", dtype=torch.float32) * box_size


def benchmark_torch_cluster(positions: torch.Tensor, cutoff: float,
                           warmup: int = 10, iterations: int = 50) -> dict:
    """Benchmark torch_cluster.radius_graph."""
    N = positions.shape[0]

    # Warmup
    for _ in range(warmup):
        edge_index = torch_cluster.radius_graph(positions, r=cutoff, loop=False,
                                                max_num_neighbors=128)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        edge_index = torch_cluster.radius_graph(positions, r=cutoff, loop=False,
                                                max_num_neighbors=128)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000

    num_edges = edge_index.shape[1]

    return {
        "avg_ms": avg_ms,
        "num_edges": num_edges,
        "neighbors_per_atom": num_edges / N,
    }


def benchmark_batteries_cuda(positions: torch.Tensor, cell: torch.Tensor,
                            cutoff: float, warmup: int = 10,
                            iterations: int = 50) -> dict:
    """Benchmark batteries CUDA neighbor list."""
    N = positions.shape[0]

    # Warmup
    for _ in range(warmup):
        result = NeighborListResult()
        err = lib.neighbor_list_cuda(
            ctypes.c_void_p(positions.data_ptr()),
            ctypes.c_void_p(cell.data_ptr()),
            ctypes.c_int64(N),
            ctypes.c_float(cutoff),
            ctypes.c_bool(False),
            ctypes.byref(result),
            None
        )
        torch.cuda.synchronize()
        lib.neighbor_list_free(ctypes.byref(result))

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    num_edges = 0
    for _ in range(iterations):
        result = NeighborListResult()
        err = lib.neighbor_list_cuda(
            ctypes.c_void_p(positions.data_ptr()),
            ctypes.c_void_p(cell.data_ptr()),
            ctypes.c_int64(N),
            ctypes.c_float(cutoff),
            ctypes.c_bool(False),
            ctypes.byref(result),
            None
        )
        torch.cuda.synchronize()
        num_edges = result.num_edges
        lib.neighbor_list_free(ctypes.byref(result))

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000

    return {
        "avg_ms": avg_ms,
        "num_edges": num_edges,
        "neighbors_per_atom": num_edges / N,
    }


def main():
    print("=" * 60)
    print("Neighbor List Benchmark: batteries CUDA vs torch_cluster")
    print("=" * 60)
    print()

    density = 0.05  # atoms per cubic Angstrom
    cutoff = 5.0    # Angstroms

    test_sizes = [1000, 10000, 50000]

    print(f"Parameters: density={density} atoms/A^3, cutoff={cutoff} A")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    results = []

    for N in test_sizes:
        # Calculate box size from density
        volume = N / density
        box_size = volume ** (1/3)

        print(f"N={N:,}, box_size={box_size:.2f} A")

        # Generate positions
        positions = generate_positions(N, box_size)

        # Create cell matrix (orthorhombic)
        cell = torch.zeros(3, 3, device="cuda", dtype=torch.float32)
        cell[0, 0] = box_size
        cell[1, 1] = box_size
        cell[2, 2] = box_size

        # Benchmark torch_cluster
        print("  torch_cluster.radius_graph...", end=" ", flush=True)
        tc_result = benchmark_torch_cluster(positions, cutoff)
        print(f"{tc_result['avg_ms']:.3f} ms, {tc_result['num_edges']:,} edges")

        # Benchmark batteries CUDA
        print("  batteries CUDA...", end=" ", flush=True)
        bat_result = benchmark_batteries_cuda(positions, cell, cutoff)
        print(f"{bat_result['avg_ms']:.3f} ms, {bat_result['num_edges']:,} edges")

        speedup = tc_result['avg_ms'] / bat_result['avg_ms']
        print(f"  Speedup: {speedup:.2f}x")
        print()

        results.append({
            "N": N,
            "torch_cluster_ms": tc_result['avg_ms'],
            "batteries_ms": bat_result['avg_ms'],
            "speedup": speedup,
            "tc_edges": tc_result['num_edges'],
            "bat_edges": bat_result['num_edges'],
        })

    # Summary table
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'N':>8} | {'torch_cluster':>14} | {'batteries':>12} | {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['N']:>8,} | {r['torch_cluster_ms']:>11.3f} ms | {r['batteries_ms']:>9.3f} ms | {r['speedup']:>7.2f}x")

    # Edge count analysis
    print()
    print("Edge count analysis:")
    for r in results:
        diff_pct = 100 * (r['bat_edges'] - r['tc_edges']) / r['bat_edges']
        print(f"  N={r['N']:,}: torch_cluster={r['tc_edges']:,}, batteries={r['bat_edges']:,} (+{diff_pct:.1f}% from PBC)")

    print()
    print("Note: Edge count differences are expected:")
    print("  - torch_cluster.radius_graph does NOT apply periodic boundary conditions")
    print("  - batteries CUDA applies PBC using minimum image convention")
    print("  - Extra edges (~20%) come from neighbors across periodic boundaries")
    print("  - For molecular simulation with periodic cells, PBC is required")


if __name__ == "__main__":
    main()
