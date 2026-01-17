/**
 * Neighbor List Construction with Cell-List Spatial Hashing
 *
 * Efficiently finds all atom pairs within a cutoff distance using spatial hashing.
 * Supports periodic boundary conditions (orthorhombic cells).
 *
 * Algorithm:
 * 1. Subdivide space into bins of size >= cutoff
 * 2. Assign each atom to a bin (parallel)
 * 3. For each atom, check only 27 neighboring bins
 * 4. Use parallel prefix sum for variable-length edge output
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace batteries {

// Maximum expected neighbors per atom (for buffer sizing)
constexpr int MAX_NEIGHBORS_PER_ATOM = 128;

// Result structure for neighbor list
struct NeighborListResult {
    int* edge_index;      // [2, num_edges] - source and target indices
    float* edge_vectors;  // [num_edges, 3] - displacement vectors (target - source)
    int64_t num_edges;

    // For memory management
    int* d_edge_index;
    float* d_edge_vectors;
};

/**
 * Build neighbor list using cell-list spatial hashing.
 *
 * @param positions    [N, 3] atom positions
 * @param cell         [3, 3] periodic cell vectors (row-major, assumes orthorhombic)
 * @param N            number of atoms
 * @param cutoff       neighbor cutoff distance
 * @param self_loops   include self-interactions (i == j)
 * @param stream       CUDA stream
 * @return             NeighborListResult with edge_index and edge_vectors
 */
cudaError_t neighbor_list_cuda(
    const float* positions,   // [N, 3]
    const float* cell,        // [3, 3] - only diagonal used for orthorhombic
    int64_t N,
    float cutoff,
    bool self_loops,
    NeighborListResult* result,
    cudaStream_t stream = 0
);

/**
 * Free memory allocated by neighbor_list_cuda.
 */
void neighbor_list_free(NeighborListResult* result);

/**
 * Get estimated memory requirements.
 *
 * @param N            number of atoms
 * @param density      atoms per cubic Angstrom (~0.05 for liquids)
 * @param cutoff       neighbor cutoff
 * @return             estimated bytes needed
 */
size_t neighbor_list_estimate_memory(int64_t N, float density, float cutoff);

} // namespace batteries
