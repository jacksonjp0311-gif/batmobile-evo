/**
 * Neighbor List Construction - Cell-List Spatial Hashing Implementation
 *
 * Two-pass algorithm:
 * Pass 1: Count neighbors per atom
 * Pass 2: Write edges using prefix-sum offsets
 *
 * Memory layout optimized for coalesced access.
 */

#include "neighbor_list.cuh"
#include <cstdio>
#include <cub/cub.cuh>

namespace batteries {

// Bin configuration
struct BinConfig {
    int3 num_bins;        // Number of bins in each dimension
    float3 bin_size;      // Size of each bin
    float3 box_size;      // Total box dimensions
    float3 inv_box_size;  // 1 / box_size for fast PBC wrapping
    int total_bins;       // Total number of bins
};

/**
 * Compute bin index from 3D bin coordinates.
 * Handles periodic wrapping.
 */
__device__ __forceinline__ int bin_index(int bx, int by, int bz, const BinConfig& cfg) {
    // Periodic wrap
    bx = (bx + cfg.num_bins.x) % cfg.num_bins.x;
    by = (by + cfg.num_bins.y) % cfg.num_bins.y;
    bz = (bz + cfg.num_bins.z) % cfg.num_bins.z;
    return bx + by * cfg.num_bins.x + bz * cfg.num_bins.x * cfg.num_bins.y;
}

/**
 * Get bin coordinates for a position.
 */
__device__ __forceinline__ int3 get_bin_coords(float3 pos, const BinConfig& cfg) {
    // Wrap position into box [0, box_size)
    pos.x = pos.x - floorf(pos.x * cfg.inv_box_size.x) * cfg.box_size.x;
    pos.y = pos.y - floorf(pos.y * cfg.inv_box_size.y) * cfg.box_size.y;
    pos.z = pos.z - floorf(pos.z * cfg.inv_box_size.z) * cfg.box_size.z;

    int3 bin;
    bin.x = min(int(pos.x / cfg.bin_size.x), cfg.num_bins.x - 1);
    bin.y = min(int(pos.y / cfg.bin_size.y), cfg.num_bins.y - 1);
    bin.z = min(int(pos.z / cfg.bin_size.z), cfg.num_bins.z - 1);
    return bin;
}

/**
 * Apply minimum image convention for periodic boundaries.
 */
__device__ __forceinline__ float3 minimum_image(float3 delta, const BinConfig& cfg) {
    delta.x -= rintf(delta.x * cfg.inv_box_size.x) * cfg.box_size.x;
    delta.y -= rintf(delta.y * cfg.inv_box_size.y) * cfg.box_size.y;
    delta.z -= rintf(delta.z * cfg.inv_box_size.z) * cfg.box_size.z;
    return delta;
}

/**
 * Kernel 1: Assign atoms to bins and count atoms per bin.
 */
__global__ void assign_atoms_to_bins_kernel(
    const float* __restrict__ positions,  // [N, 3]
    int* __restrict__ atom_bins,          // [N] - bin index for each atom
    int* __restrict__ bin_counts,         // [total_bins] - atoms per bin
    BinConfig cfg,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 pos;
    pos.x = positions[idx * 3 + 0];
    pos.y = positions[idx * 3 + 1];
    pos.z = positions[idx * 3 + 2];

    int3 bin_coords = get_bin_coords(pos, cfg);
    int bin_idx = bin_coords.x + bin_coords.y * cfg.num_bins.x +
                  bin_coords.z * cfg.num_bins.x * cfg.num_bins.y;

    atom_bins[idx] = bin_idx;
    atomicAdd(&bin_counts[bin_idx], 1);
}

/**
 * Kernel 2: Build bin start indices (prefix sum done externally).
 * Place atoms into sorted order by bin.
 */
__global__ void sort_atoms_by_bin_kernel(
    const int* __restrict__ atom_bins,    // [N] - bin index for each atom
    const int* __restrict__ bin_starts,   // [total_bins] - start index for each bin
    int* __restrict__ bin_offsets,        // [total_bins] - current offset (atomic counter)
    int* __restrict__ sorted_indices,     // [N] - atom indices sorted by bin
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int bin_idx = atom_bins[idx];
    int offset = atomicAdd(&bin_offsets[bin_idx], 1);
    sorted_indices[bin_starts[bin_idx] + offset] = idx;
}

/**
 * Kernel 3: Count neighbors for each atom (Pass 1).
 */
__global__ void count_neighbors_kernel(
    const float* __restrict__ positions,      // [N, 3]
    const int* __restrict__ atom_bins,        // [N]
    const int* __restrict__ bin_starts,       // [total_bins + 1]
    const int* __restrict__ sorted_indices,   // [N]
    int* __restrict__ neighbor_counts,        // [N] - output counts
    BinConfig cfg,
    float cutoff_sq,
    bool self_loops,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 pos_i;
    pos_i.x = positions[i * 3 + 0];
    pos_i.y = positions[i * 3 + 1];
    pos_i.z = positions[i * 3 + 2];

    int3 bin_i = get_bin_coords(pos_i, cfg);
    int count = 0;

    // Check 27 neighboring bins
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_bin = bin_index(bin_i.x + dx, bin_i.y + dy, bin_i.z + dz, cfg);

                int start = bin_starts[neighbor_bin];
                int end = bin_starts[neighbor_bin + 1];

                for (int k = start; k < end; k++) {
                    int j = sorted_indices[k];

                    // Skip self if not including self-loops
                    if (!self_loops && i == j) continue;

                    float3 pos_j;
                    pos_j.x = positions[j * 3 + 0];
                    pos_j.y = positions[j * 3 + 1];
                    pos_j.z = positions[j * 3 + 2];

                    float3 delta;
                    delta.x = pos_j.x - pos_i.x;
                    delta.y = pos_j.y - pos_i.y;
                    delta.z = pos_j.z - pos_i.z;

                    delta = minimum_image(delta, cfg);
                    float dist_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

                    if (dist_sq <= cutoff_sq) {
                        count++;
                    }
                }
            }
        }
    }

    neighbor_counts[i] = count;
}

/**
 * Kernel 4: Write neighbor pairs (Pass 2).
 */
__global__ void write_neighbors_kernel(
    const float* __restrict__ positions,      // [N, 3]
    const int* __restrict__ atom_bins,        // [N]
    const int* __restrict__ bin_starts,       // [total_bins + 1]
    const int* __restrict__ sorted_indices,   // [N]
    const int* __restrict__ neighbor_offsets, // [N] - prefix sum of counts
    int* __restrict__ edge_src,               // [num_edges]
    int* __restrict__ edge_dst,               // [num_edges]
    float* __restrict__ edge_vectors,         // [num_edges, 3]
    BinConfig cfg,
    float cutoff_sq,
    bool self_loops,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 pos_i;
    pos_i.x = positions[i * 3 + 0];
    pos_i.y = positions[i * 3 + 1];
    pos_i.z = positions[i * 3 + 2];

    int3 bin_i = get_bin_coords(pos_i, cfg);
    int write_idx = neighbor_offsets[i];

    // Check 27 neighboring bins
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighbor_bin = bin_index(bin_i.x + dx, bin_i.y + dy, bin_i.z + dz, cfg);

                int start = bin_starts[neighbor_bin];
                int end = bin_starts[neighbor_bin + 1];

                for (int k = start; k < end; k++) {
                    int j = sorted_indices[k];

                    if (!self_loops && i == j) continue;

                    float3 pos_j;
                    pos_j.x = positions[j * 3 + 0];
                    pos_j.y = positions[j * 3 + 1];
                    pos_j.z = positions[j * 3 + 2];

                    float3 delta;
                    delta.x = pos_j.x - pos_i.x;
                    delta.y = pos_j.y - pos_i.y;
                    delta.z = pos_j.z - pos_i.z;

                    delta = minimum_image(delta, cfg);
                    float dist_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

                    if (dist_sq <= cutoff_sq) {
                        edge_src[write_idx] = i;
                        edge_dst[write_idx] = j;
                        edge_vectors[write_idx * 3 + 0] = delta.x;
                        edge_vectors[write_idx * 3 + 1] = delta.y;
                        edge_vectors[write_idx * 3 + 2] = delta.z;
                        write_idx++;
                    }
                }
            }
        }
    }
}

cudaError_t neighbor_list_cuda(
    const float* positions,
    const float* cell,
    int64_t N,
    float cutoff,
    bool self_loops,
    NeighborListResult* result,
    cudaStream_t stream
) {
    if (N == 0) {
        result->num_edges = 0;
        result->edge_index = nullptr;
        result->edge_vectors = nullptr;
        return cudaSuccess;
    }

    cudaError_t err;

    // Read cell dimensions (assume orthorhombic - only diagonal)
    float h_cell[9];
    cudaMemcpy(h_cell, cell, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    BinConfig cfg;
    cfg.box_size.x = h_cell[0];  // cell[0,0]
    cfg.box_size.y = h_cell[4];  // cell[1,1]
    cfg.box_size.z = h_cell[8];  // cell[2,2]

    cfg.inv_box_size.x = 1.0f / cfg.box_size.x;
    cfg.inv_box_size.y = 1.0f / cfg.box_size.y;
    cfg.inv_box_size.z = 1.0f / cfg.box_size.z;

    // Determine number of bins (at least 3 in each dimension for 27-neighbor search)
    cfg.num_bins.x = max(3, int(cfg.box_size.x / cutoff));
    cfg.num_bins.y = max(3, int(cfg.box_size.y / cutoff));
    cfg.num_bins.z = max(3, int(cfg.box_size.z / cutoff));

    cfg.bin_size.x = cfg.box_size.x / cfg.num_bins.x;
    cfg.bin_size.y = cfg.box_size.y / cfg.num_bins.y;
    cfg.bin_size.z = cfg.box_size.z / cfg.num_bins.z;

    cfg.total_bins = cfg.num_bins.x * cfg.num_bins.y * cfg.num_bins.z;

    float cutoff_sq = cutoff * cutoff;

    // Allocate temporary buffers
    int* d_atom_bins;       // [N]
    int* d_bin_counts;      // [total_bins]
    int* d_bin_starts;      // [total_bins + 1]
    int* d_bin_offsets;     // [total_bins]
    int* d_sorted_indices;  // [N]
    int* d_neighbor_counts; // [N]
    int* d_neighbor_offsets;// [N + 1]

    err = cudaMalloc(&d_atom_bins, N * sizeof(int));
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&d_bin_counts, cfg.total_bins * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); return err; }

    err = cudaMalloc(&d_bin_starts, (cfg.total_bins + 1) * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); cudaFree(d_bin_counts); return err; }

    err = cudaMalloc(&d_bin_offsets, cfg.total_bins * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); cudaFree(d_bin_counts); cudaFree(d_bin_starts); return err; }

    err = cudaMalloc(&d_sorted_indices, N * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); cudaFree(d_bin_counts); cudaFree(d_bin_starts); cudaFree(d_bin_offsets); return err; }

    err = cudaMalloc(&d_neighbor_counts, N * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); cudaFree(d_bin_counts); cudaFree(d_bin_starts); cudaFree(d_bin_offsets); cudaFree(d_sorted_indices); return err; }

    err = cudaMalloc(&d_neighbor_offsets, (N + 1) * sizeof(int));
    if (err != cudaSuccess) { cudaFree(d_atom_bins); cudaFree(d_bin_counts); cudaFree(d_bin_starts); cudaFree(d_bin_offsets); cudaFree(d_sorted_indices); cudaFree(d_neighbor_counts); return err; }

    // CUB temp storage for prefix sums
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Initialize bin counts to zero
    cudaMemsetAsync(d_bin_counts, 0, cfg.total_bins * sizeof(int), stream);

    // Step 1: Assign atoms to bins
    const int block_size = 256;
    const int num_blocks_atoms = (N + block_size - 1) / block_size;

    assign_atoms_to_bins_kernel<<<num_blocks_atoms, block_size, 0, stream>>>(
        positions, d_atom_bins, d_bin_counts, cfg, N
    );

    // Step 2: Exclusive prefix sum on bin counts to get bin starts
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_bin_counts, d_bin_starts, cfg.total_bins + 1, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_bin_counts, d_bin_starts, cfg.total_bins + 1, stream);

    // Step 3: Sort atoms by bin
    cudaMemsetAsync(d_bin_offsets, 0, cfg.total_bins * sizeof(int), stream);

    sort_atoms_by_bin_kernel<<<num_blocks_atoms, block_size, 0, stream>>>(
        d_atom_bins, d_bin_starts, d_bin_offsets, d_sorted_indices, N
    );

    // Step 4: Count neighbors for each atom (Pass 1)
    count_neighbors_kernel<<<num_blocks_atoms, block_size, 0, stream>>>(
        positions, d_atom_bins, d_bin_starts, d_sorted_indices,
        d_neighbor_counts, cfg, cutoff_sq, self_loops, N
    );

    // Step 5: Prefix sum to get neighbor offsets
    cudaFree(d_temp_storage);
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_neighbor_counts, d_neighbor_offsets, N + 1, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                   d_neighbor_counts, d_neighbor_offsets, N + 1, stream);

    // Get total number of edges
    int total_edges_int;
    cudaMemcpyAsync(&total_edges_int, d_neighbor_offsets + N, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_edges = total_edges_int;

    // Allocate output buffers
    int* d_edge_src;
    int* d_edge_dst;
    float* d_edge_vectors;

    if (total_edges > 0) {
        err = cudaMalloc(&d_edge_src, total_edges * sizeof(int));
        if (err != cudaSuccess) goto cleanup;

        err = cudaMalloc(&d_edge_dst, total_edges * sizeof(int));
        if (err != cudaSuccess) { cudaFree(d_edge_src); goto cleanup; }

        err = cudaMalloc(&d_edge_vectors, total_edges * 3 * sizeof(float));
        if (err != cudaSuccess) { cudaFree(d_edge_src); cudaFree(d_edge_dst); goto cleanup; }

        // Step 6: Write neighbor pairs (Pass 2)
        write_neighbors_kernel<<<num_blocks_atoms, block_size, 0, stream>>>(
            positions, d_atom_bins, d_bin_starts, d_sorted_indices,
            d_neighbor_offsets, d_edge_src, d_edge_dst, d_edge_vectors,
            cfg, cutoff_sq, self_loops, N
        );

        // Allocate and copy edge_index (interleaved format [2, N_edges])
        err = cudaMalloc(&result->d_edge_index, 2 * total_edges * sizeof(int));
        if (err != cudaSuccess) {
            cudaFree(d_edge_src); cudaFree(d_edge_dst); cudaFree(d_edge_vectors);
            goto cleanup;
        }

        // Copy src and dst to interleaved format
        cudaMemcpyAsync(result->d_edge_index, d_edge_src,
                        total_edges * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result->d_edge_index + total_edges, d_edge_dst,
                        total_edges * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        result->d_edge_vectors = d_edge_vectors;
        result->edge_index = result->d_edge_index;
        result->edge_vectors = result->d_edge_vectors;

        cudaFree(d_edge_src);
        cudaFree(d_edge_dst);
    } else {
        result->d_edge_index = nullptr;
        result->d_edge_vectors = nullptr;
        result->edge_index = nullptr;
        result->edge_vectors = nullptr;
    }

    result->num_edges = total_edges;

cleanup:
    cudaFree(d_atom_bins);
    cudaFree(d_bin_counts);
    cudaFree(d_bin_starts);
    cudaFree(d_bin_offsets);
    cudaFree(d_sorted_indices);
    cudaFree(d_neighbor_counts);
    cudaFree(d_neighbor_offsets);
    cudaFree(d_temp_storage);

    return cudaGetLastError();
}

void neighbor_list_free(NeighborListResult* result) {
    if (result->d_edge_index) cudaFree(result->d_edge_index);
    if (result->d_edge_vectors) cudaFree(result->d_edge_vectors);
    result->d_edge_index = nullptr;
    result->d_edge_vectors = nullptr;
    result->edge_index = nullptr;
    result->edge_vectors = nullptr;
    result->num_edges = 0;
}

size_t neighbor_list_estimate_memory(int64_t N, float density, float cutoff) {
    // Estimate neighbors per atom
    float volume_sphere = (4.0f / 3.0f) * 3.14159f * cutoff * cutoff * cutoff;
    int avg_neighbors = int(density * volume_sphere) + 1;

    size_t edge_memory = N * avg_neighbors * (2 * sizeof(int) + 3 * sizeof(float));
    size_t temp_memory = N * sizeof(int) * 4;  // Various temporary buffers

    return edge_memory + temp_memory;
}

} // namespace batteries

// C API wrappers for Python ctypes binding
extern "C" {

cudaError_t neighbor_list_cuda(
    const float* positions,
    const float* cell,
    int64_t N,
    float cutoff,
    bool self_loops,
    batteries::NeighborListResult* result,
    cudaStream_t stream
) {
    return batteries::neighbor_list_cuda(positions, cell, N, cutoff, self_loops, result, stream);
}

void neighbor_list_free(batteries::NeighborListResult* result) {
    batteries::neighbor_list_free(result);
}

size_t neighbor_list_estimate_memory(int64_t N, float density, float cutoff) {
    return batteries::neighbor_list_estimate_memory(N, density, cutoff);
}

} // extern "C"
