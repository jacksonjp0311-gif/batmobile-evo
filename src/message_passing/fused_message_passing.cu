/**
 * Fused Message Passing CUDA Kernel
 *
 * Combines spherical harmonics, tensor product, and scatter-add into one kernel.
 * Eliminates global memory round-trips between operations.
 *
 * Architecture:
 * - Each block processes one target atom
 * - Threads cooperatively process incoming edges
 * - SH computed inline in registers
 * - TP computed inline, accumulated in shared memory
 * - Final result written to global memory
 */

#include "fused_message_passing.cuh"
#include "tensor_product_fused.cuh"
#include <cub/cub.cuh>

namespace batteries {

// Inline spherical harmonics for L_MAX=3 (16 components)
// Computes directly into register array, no global memory access
__device__ __forceinline__ void compute_sh_inline(
    float x, float y, float z,
    float* __restrict__ sh  // sh[16] output array
) {
    // Precompute powers
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float x2z2 = x2 + z2;

    // L=0: Y_0 = 1
    sh[0] = 1.0f;

    // L=1: sqrt(3) * (x, y, z)
    constexpr float c1 = 1.7320508075688772f;  // sqrt(3)
    sh[1] = c1 * x;
    sh[2] = c1 * y;
    sh[3] = c1 * z;

    // L=2: 5 components
    constexpr float sqrt15 = 3.872983346207417f;
    constexpr float sqrt5 = 2.23606797749979f;
    sh[4] = sqrt15 * x * z;                    // sh_2_0
    sh[5] = sqrt15 * x * y;                    // sh_2_1
    sh[6] = sqrt5 * (y2 - 0.5f * x2z2);       // sh_2_2
    sh[7] = sqrt15 * y * z;                    // sh_2_3
    sh[8] = 0.5f * sqrt15 * (z2 - x2);        // sh_2_4

    // L=3: 7 components (using recurrence from L=2)
    constexpr float c3_0 = 1.0801234497346432f;
    constexpr float c3_1 = 2.6457513110645907f;
    constexpr float c3_2 = 1.6201851746019651f;
    constexpr float c3_3 = 1.3228756555322954f;

    const float sh_2_0 = sh[4];
    const float sh_2_4 = sh[8];

    sh[9]  = c3_0 * (sh_2_0 * z + sh_2_4 * x);
    sh[10] = c3_1 * sh_2_0 * y;
    sh[11] = c3_2 * (4.0f * y2 - x2z2) * x;
    sh[12] = c3_3 * y * (2.0f * y2 - 3.0f * x2z2);
    sh[13] = c3_2 * z * (4.0f * y2 - x2z2);
    sh[14] = c3_1 * sh_2_4 * y;
    sh[15] = c3_0 * (sh_2_4 * z - sh_2_0 * x);
}

// ============================================================================
// CSR Conversion Kernels
// ============================================================================

// Count edges per target atom
__global__ void count_edges_per_atom_kernel(
    const int64_t* __restrict__ edge_index,  // [2, N_edges]
    int64_t* __restrict__ counts,            // [N_atoms]
    int64_t N_edges
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_edges) return;

    // edge_index[1, idx] = target atom
    int64_t target = edge_index[N_edges + idx];
    // Use unsigned long long atomicAdd (int64_t doesn't have direct atomicAdd support)
    atomicAdd(reinterpret_cast<unsigned long long*>(&counts[target]), 1ULL);
}

// Scatter edges to CSR positions
// col_idx stores ORIGINAL EDGE INDICES (not source atoms) so we can look up edge_vectors
__global__ void scatter_edges_kernel(
    const int64_t* __restrict__ edge_index,  // [2, N_edges]
    const int64_t* __restrict__ row_ptr,     // [N_atoms+1]
    int64_t* __restrict__ col_idx,           // [N_edges] - stores original edge indices
    int64_t* __restrict__ offsets,           // [N_atoms] current offset per atom
    int64_t N_edges
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_edges) return;

    int64_t target = edge_index[N_edges + idx];

    // Get position for this edge
    unsigned long long pos = atomicAdd(reinterpret_cast<unsigned long long*>(&offsets[target]), 1ULL);
    // Store original edge index so we can look up edge_vectors[idx]
    col_idx[row_ptr[target] + pos] = idx;
}

cudaError_t edge_index_to_csr(
    const int64_t* edge_index,
    int64_t* row_ptr,
    int64_t* col_idx,
    int64_t N_atoms,
    int64_t N_edges,
    cudaStream_t stream
) {
    if (N_edges == 0) {
        cudaMemsetAsync(row_ptr, 0, (N_atoms + 1) * sizeof(int64_t), stream);
        return cudaSuccess;
    }

    // Allocate temporary counts array (int64_t to match row_ptr type)
    int64_t* counts;
    cudaMallocAsync(&counts, N_atoms * sizeof(int64_t), stream);
    cudaMemsetAsync(counts, 0, N_atoms * sizeof(int64_t), stream);

    constexpr int block_size = 256;
    int num_blocks = (N_edges + block_size - 1) / block_size;

    // Count edges per target atom
    count_edges_per_atom_kernel<<<num_blocks, block_size, 0, stream>>>(
        edge_index, counts, N_edges
    );

    // Exclusive prefix sum to get row_ptr
    // Using CUB for efficiency
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temp storage size
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts, row_ptr, static_cast<int>(N_atoms + 1), stream
    );

    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

    // Run exclusive sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        counts, row_ptr, static_cast<int>(N_atoms + 1), stream
    );

    // Reset counts as offsets for scatter
    cudaMemsetAsync(counts, 0, N_atoms * sizeof(int64_t), stream);

    // Scatter edges
    scatter_edges_kernel<<<num_blocks, block_size, 0, stream>>>(
        edge_index, row_ptr, col_idx, counts, N_edges
    );

    cudaFreeAsync(counts, stream);
    cudaFreeAsync(d_temp_storage, stream);

    return cudaGetLastError();
}

// ============================================================================
// Fused Message Passing Kernel (Simple - no weights)
// ============================================================================

constexpr int NUM_IRREPS = 16;  // (3+1)^2 for L_max=3

// Simple version: no weights, just SH x features -> accumulate
template <int BLOCK_SIZE>
__global__ void fused_message_passing_simple_kernel(
    const float* __restrict__ node_features,  // [N_atoms, C, 16]
    const float* __restrict__ edge_vectors,   // [N_edges, 3]
    const int64_t* __restrict__ edge_index,   // [2, N_edges] - source indices at [0, :]
    const int64_t* __restrict__ row_ptr,      // [N_atoms+1]
    const int64_t* __restrict__ col_idx,      // [N_edges] - original edge indices in CSR order
    float* __restrict__ node_output,          // [N_atoms, C, 16]
    int64_t N_atoms,
    int64_t N_edges,
    int C
) {
    // Each block handles one target atom
    const int target_atom = blockIdx.x;
    if (target_atom >= N_atoms) return;

    const int tid = threadIdx.x;

    // Get edge range for this target atom
    const int64_t edge_start = row_ptr[target_atom];
    const int64_t edge_end = row_ptr[target_atom + 1];
    const int num_edges = edge_end - edge_start;

    if (num_edges == 0) {
        // Zero output for this atom
        for (int c = tid; c < C * NUM_IRREPS; c += BLOCK_SIZE) {
            node_output[target_atom * C * NUM_IRREPS + c] = 0.0f;
        }
        return;
    }

    // Shared memory for accumulation: [C, 16]
    extern __shared__ float smem[];
    float* accumulator = smem;  // [C * NUM_IRREPS]

    // Initialize accumulator to zero
    for (int i = tid; i < C * NUM_IRREPS; i += BLOCK_SIZE) {
        accumulator[i] = 0.0f;
    }
    __syncthreads();

    // Process edges - each thread handles a subset of edges
    for (int edge_offset = tid; edge_offset < num_edges; edge_offset += BLOCK_SIZE) {
        const int64_t csr_idx = edge_start + edge_offset;
        const int64_t orig_edge_idx = col_idx[csr_idx];  // Get original edge index
        const int64_t source_atom = edge_index[orig_edge_idx];  // Source is at edge_index[0, orig_edge_idx]

        // Load edge vector using original edge index
        float x = edge_vectors[orig_edge_idx * 3 + 0];
        float y = edge_vectors[orig_edge_idx * 3 + 1];
        float z = edge_vectors[orig_edge_idx * 3 + 2];

        float sh[NUM_IRREPS];
        compute_sh_inline(x, y, z, sh);

        // For each channel, compute tensor product and accumulate
        for (int c = 0; c < C; c++) {
            const float* src_feat = node_features + source_atom * C * NUM_IRREPS + c * NUM_IRREPS;

            float out_local[NUM_IRREPS];
            #pragma unroll
            for (int m = 0; m < NUM_IRREPS; m++) {
                out_local[m] = 0.0f;
            }

            // Use the fused tensor product (inline CG coefficients)
            fused::tensor_product_simple_unrolled(src_feat, sh, out_local);

            // Atomic add to shared memory accumulator
            #pragma unroll
            for (int m = 0; m < NUM_IRREPS; m++) {
                atomicAdd(&accumulator[c * NUM_IRREPS + m], out_local[m]);
            }
        }
    }
    __syncthreads();

    // Write accumulated result to global memory
    for (int i = tid; i < C * NUM_IRREPS; i += BLOCK_SIZE) {
        node_output[target_atom * C * NUM_IRREPS + i] = accumulator[i];
    }
}

cudaError_t fused_message_passing_simple(
    const float* node_features,
    const float* edge_vectors,
    const int64_t* edge_index,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    float* node_output,
    int64_t N_atoms,
    int64_t N_edges,
    int C,
    cudaStream_t stream
) {
    if (N_atoms == 0) return cudaSuccess;

    constexpr int BLOCK_SIZE = 128;
    const int smem_size = C * NUM_IRREPS * sizeof(float);

    fused_message_passing_simple_kernel<BLOCK_SIZE><<<N_atoms, BLOCK_SIZE, smem_size, stream>>>(
        node_features, edge_vectors, edge_index, row_ptr, col_idx, node_output,
        N_atoms, N_edges, C
    );

    return cudaGetLastError();
}

// ============================================================================
// Fused Message Passing Kernel (With Weights)
// ============================================================================

// Weighted version: features x SH with CG coefficients and learnable weights
template <int BLOCK_SIZE>
__global__ void fused_message_passing_weighted_kernel(
    const float* __restrict__ node_features,  // [N_atoms, C_in, 16]
    const float* __restrict__ edge_vectors,   // [N_edges, 3]
    const float* __restrict__ weights,        // [num_paths, C_in, C_out]
    const int64_t* __restrict__ row_ptr,      // [N_atoms+1]
    const int64_t* __restrict__ col_idx,      // [N_edges]
    float* __restrict__ node_output,          // [N_atoms, C_out, 16]
    int64_t N_atoms,
    int C_in,
    int C_out
) {
    // Each block handles one target atom
    const int target_atom = blockIdx.x;
    if (target_atom >= N_atoms) return;

    const int tid = threadIdx.x;

    // Get edge range for this target atom
    const int64_t edge_start = row_ptr[target_atom];
    const int64_t edge_end = row_ptr[target_atom + 1];
    const int num_edges = edge_end - edge_start;

    if (num_edges == 0) {
        // Zero output for this atom
        for (int c = tid; c < C_out * NUM_IRREPS; c += BLOCK_SIZE) {
            node_output[target_atom * C_out * NUM_IRREPS + c] = 0.0f;
        }
        return;
    }

    // Shared memory layout:
    // - accumulator: [C_out, NUM_IRREPS]
    extern __shared__ float smem[];
    float* accumulator = smem;

    // Initialize accumulator
    for (int i = tid; i < C_out * NUM_IRREPS; i += BLOCK_SIZE) {
        accumulator[i] = 0.0f;
    }
    __syncthreads();

    // Process edges
    for (int edge_offset = tid; edge_offset < num_edges; edge_offset += BLOCK_SIZE) {
        const int64_t edge_idx = edge_start + edge_offset;
        const int64_t source_atom = col_idx[edge_idx];

        // Load edge vector and compute SH inline
        float x = edge_vectors[edge_idx * 3 + 0];
        float y = edge_vectors[edge_idx * 3 + 1];
        float z = edge_vectors[edge_idx * 3 + 2];

        float sh[NUM_IRREPS];
        compute_sh_inline(x, y, z, sh);

        // For each input channel, compute TP and weight to output channels
        for (int c_in = 0; c_in < C_in; c_in++) {
            const float* src_feat = node_features + source_atom * C_in * NUM_IRREPS + c_in * NUM_IRREPS;

            // Compute tensor product for this input channel
            float tp_out[NUM_IRREPS];
            #pragma unroll
            for (int m = 0; m < NUM_IRREPS; m++) {
                tp_out[m] = 0.0f;
            }
            fused::tensor_product_simple_unrolled(src_feat, sh, tp_out);

            // Apply weights and accumulate to all output channels
            for (int c_out = 0; c_out < C_out; c_out++) {
                // Get weight for this path (simplified: using one weight per channel pair)
                // In full MACE, weights are per-path: weights[path, c_in, c_out]
                // Here we use a simplified dense weight: weights[c_in * C_out + c_out]
                float w = weights[c_in * C_out + c_out];

                #pragma unroll
                for (int m = 0; m < NUM_IRREPS; m++) {
                    atomicAdd(&accumulator[c_out * NUM_IRREPS + m], w * tp_out[m]);
                }
            }
        }
    }
    __syncthreads();

    // Write accumulated result to global memory
    for (int i = tid; i < C_out * NUM_IRREPS; i += BLOCK_SIZE) {
        node_output[target_atom * C_out * NUM_IRREPS + i] = accumulator[i];
    }
}

cudaError_t fused_message_passing(
    const float* node_features,
    const float* edge_vectors,
    const float* weights,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    float* node_output,
    int64_t N_atoms,
    int64_t N_edges,
    int C_in,
    int C_out,
    cudaStream_t stream
) {
    if (N_atoms == 0) return cudaSuccess;

    constexpr int BLOCK_SIZE = 128;
    const int smem_size = C_out * NUM_IRREPS * sizeof(float);

    fused_message_passing_weighted_kernel<BLOCK_SIZE><<<N_atoms, BLOCK_SIZE, smem_size, stream>>>(
        node_features, edge_vectors, weights, row_ptr, col_idx, node_output,
        N_atoms, C_in, C_out
    );

    return cudaGetLastError();
}

} // namespace batteries
