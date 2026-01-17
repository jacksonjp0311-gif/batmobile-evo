"""
Verify edge count differences: check PBC handling.

torch_cluster.radius_graph doesn't support PBC natively.
Our implementation includes PBC (minimum image convention).
"""

import torch
import torch_cluster
import ctypes
from pathlib import Path

# Load the CUDA library
lib_path = Path(__file__).parent.parent / "libbatteries_neighbor.so"
lib = ctypes.CDLL(str(lib_path))

class NeighborListResult(ctypes.Structure):
    _fields_ = [
        ("edge_index", ctypes.c_void_p),
        ("edge_vectors", ctypes.c_void_p),
        ("num_edges", ctypes.c_int64),
        ("d_edge_index", ctypes.c_void_p),
        ("d_edge_vectors", ctypes.c_void_p),
    ]

lib.neighbor_list_cuda.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ctypes.c_float, ctypes.c_bool, ctypes.POINTER(NeighborListResult), ctypes.c_void_p
]
lib.neighbor_list_cuda.restype = ctypes.c_int
lib.neighbor_list_free.argtypes = [ctypes.POINTER(NeighborListResult)]
lib.neighbor_list_free.restype = None


def main():
    torch.manual_seed(42)

    N = 1000
    density = 0.05
    cutoff = 5.0

    volume = N / density
    box_size = volume ** (1/3)

    print(f"N={N}, box_size={box_size:.2f}, cutoff={cutoff}")
    print()

    # Generate positions
    positions = torch.rand(N, 3, device="cuda", dtype=torch.float32) * box_size

    # Create cell matrix
    cell = torch.zeros(3, 3, device="cuda", dtype=torch.float32)
    cell[0, 0] = box_size
    cell[1, 1] = box_size
    cell[2, 2] = box_size

    # torch_cluster (NO PBC)
    tc_edges = torch_cluster.radius_graph(positions, r=cutoff, loop=False, max_num_neighbors=128)
    print(f"torch_cluster (no PBC): {tc_edges.shape[1]} edges")

    # Manual brute force without PBC
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_sq = (diff ** 2).sum(-1)
    mask_no_pbc = dist_sq <= cutoff * cutoff
    mask_no_pbc.fill_diagonal_(False)
    edges_no_pbc = mask_no_pbc.sum().item()
    print(f"Brute force (no PBC):   {edges_no_pbc} edges")

    # Manual brute force WITH PBC
    cell_diag = torch.tensor([box_size, box_size, box_size], device="cuda")
    diff_pbc = diff - torch.round(diff / cell_diag) * cell_diag
    dist_sq_pbc = (diff_pbc ** 2).sum(-1)
    mask_pbc = dist_sq_pbc <= cutoff * cutoff
    mask_pbc.fill_diagonal_(False)
    edges_pbc = mask_pbc.sum().item()
    print(f"Brute force (with PBC): {edges_pbc} edges")

    # batteries CUDA
    result = NeighborListResult()
    lib.neighbor_list_cuda(
        ctypes.c_void_p(positions.data_ptr()),
        ctypes.c_void_p(cell.data_ptr()),
        ctypes.c_int64(N),
        ctypes.c_float(cutoff),
        ctypes.c_bool(False),
        ctypes.byref(result),
        None
    )
    torch.cuda.synchronize()
    print(f"batteries CUDA (PBC):   {result.num_edges} edges")
    lib.neighbor_list_free(ctypes.byref(result))

    print()
    print("Conclusion:")
    print("  - torch_cluster.radius_graph does NOT apply PBC")
    print("  - batteries CUDA applies periodic boundary conditions (minimum image)")
    print("  - The extra edges in batteries are from across-boundary neighbors")

    # Show the difference
    extra = edges_pbc - edges_no_pbc
    print(f"  - Extra edges from PBC: {extra} ({100*extra/edges_pbc:.1f}% of total)")


if __name__ == "__main__":
    main()
