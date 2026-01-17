"""
Standalone benchmark for batteries CUDA neighbor list.

Measures performance without external dependencies.
"""

import torch
import ctypes
import numpy as np
import time
from pathlib import Path

# Load the CUDA library
lib_path = Path(__file__).parent.parent / "libbatteries_neighbor.so"
if not lib_path.exists():
    raise RuntimeError(f"Library not found: {lib_path}")

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


def benchmark_brute_force_torch(positions: torch.Tensor, cell_diag: torch.Tensor,
                               cutoff: float, warmup: int = 5,
                               iterations: int = 20) -> dict:
    """Brute force O(N^2) neighbor search in PyTorch for comparison."""
    N = positions.shape[0]
    cutoff_sq = cutoff * cutoff

    def find_neighbors():
        # Compute all pairwise distances (N^2)
        # This is naive but serves as baseline
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 3]

        # Apply minimum image convention
        diff = diff - torch.round(diff / cell_diag) * cell_diag

        dist_sq = (diff ** 2).sum(-1)  # [N, N]

        # Mask diagonal (no self-loops) and apply cutoff
        mask = dist_sq <= cutoff_sq
        mask.fill_diagonal_(False)

        edge_index = mask.nonzero(as_tuple=False).T  # [2, num_edges]
        return edge_index

    # Warmup
    for _ in range(warmup):
        edge_index = find_neighbors()
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        edge_index = find_neighbors()
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000

    num_edges = edge_index.shape[1]

    return {
        "avg_ms": avg_ms,
        "num_edges": num_edges,
        "neighbors_per_atom": num_edges / N,
    }


def main():
    print("=" * 60)
    print("Neighbor List Benchmark: batteries CUDA")
    print("=" * 60)
    print()

    density = 0.05  # atoms per cubic Angstrom
    cutoff = 5.0    # Angstroms

    test_sizes = [1000, 10000, 50000, 100000]

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
        cell_diag = torch.tensor([box_size, box_size, box_size], device="cuda")

        # O(N^2) brute force for small sizes only
        bf_result = None
        if N <= 10000:
            print("  PyTorch O(N^2) brute force...", end=" ", flush=True)
            bf_result = benchmark_brute_force_torch(positions, cell_diag, cutoff,
                                                    warmup=3, iterations=10)
            print(f"{bf_result['avg_ms']:.3f} ms, {bf_result['num_edges']:,} edges")

        # Benchmark batteries CUDA
        print("  batteries CUDA O(N)...", end=" ", flush=True)
        bat_result = benchmark_batteries_cuda(positions, cell, cutoff)
        print(f"{bat_result['avg_ms']:.3f} ms, {bat_result['num_edges']:,} edges")

        if bf_result:
            speedup = bf_result['avg_ms'] / bat_result['avg_ms']
            print(f"  Speedup vs brute force: {speedup:.1f}x")

        print()

        results.append({
            "N": N,
            "batteries_ms": bat_result['avg_ms'],
            "num_edges": bat_result['num_edges'],
            "neighbors_per_atom": bat_result['neighbors_per_atom'],
            "bf_ms": bf_result['avg_ms'] if bf_result else None,
        })

    # Summary table
    print("=" * 60)
    print("batteries CUDA Neighbor List Performance Summary")
    print("=" * 60)
    print(f"{'N':>8} | {'Time (ms)':>10} | {'Edges':>12} | {'Neighbors/atom':>15} | {'Throughput':>15}")
    print("-" * 75)
    for r in results:
        throughput = r['N'] / (r['batteries_ms'] / 1000)  # atoms/sec
        print(f"{r['N']:>8,} | {r['batteries_ms']:>10.3f} | {r['num_edges']:>12,} | {r['neighbors_per_atom']:>15.1f} | {throughput/1e6:>12.2f} M/s")

    print()
    print("Notes:")
    print("  - Cell-list algorithm is O(N) vs O(N^2) brute force")
    print("  - Expected ~26 neighbors/atom for density=0.05, cutoff=5.0")
    print("  - Throughput = atoms processed per second")


if __name__ == "__main__":
    main()
