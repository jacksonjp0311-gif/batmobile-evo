/**
 * Fused Spherical Harmonics + Tensor Product Kernel
 *
 * Eliminates the intermediate Y_lm tensor by:
 * 1. Computing spherical harmonics in registers
 * 2. Immediately using them for tensor product
 * 3. Writing messages to global memory only once
 *
 * Parallelization: one thread per (edge, output_channel) pair
 * This matches the parallelism of the unfused tensor product kernel.
 */

#include "fused_sh_tp.cuh"
#include "tensor_product_fused.cuh"

namespace batteries {

constexpr int NUM_IRREPS = 16;  // (3+1)^2 for L_max=3

// Inline spherical harmonics for L_MAX=3 (16 components)
// Computes directly into register array
__device__ __forceinline__ void compute_sh_registers(
    float x, float y, float z,
    float* __restrict__ sh  // sh[16] in registers
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
// Simple fused SH+TP kernel (no weights)
// One thread per (edge, channel) pair
// ============================================================================

__global__ void fused_sh_tp_simple_kernel(
    const float* __restrict__ edge_vectors,   // [N_edges, 3]
    const float* __restrict__ node_features,  // [N_atoms, C, 16]
    const int64_t* __restrict__ source_idx,   // [N_edges]
    float* __restrict__ messages,             // [N_edges, C, 16]
    int64_t N_edges,
    int C
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t edge = idx / C;
    const int channel = idx % C;

    if (edge >= N_edges) return;

    // Load edge vector
    const float x = edge_vectors[edge * 3 + 0];
    const float y = edge_vectors[edge * 3 + 1];
    const float z = edge_vectors[edge * 3 + 2];

    // Compute spherical harmonics in registers
    float sh[NUM_IRREPS];
    compute_sh_registers(x, y, z, sh);

    // Load source atom index
    const int64_t src = source_idx[edge];

    // Load source features for this channel
    const float* src_feat = node_features + src * C * NUM_IRREPS + channel * NUM_IRREPS;
    float in1[NUM_IRREPS];
    #pragma unroll
    for (int m = 0; m < NUM_IRREPS; m++) {
        in1[m] = src_feat[m];
    }

    // Compute tensor product in registers
    float out[NUM_IRREPS];
    #pragma unroll
    for (int m = 0; m < NUM_IRREPS; m++) {
        out[m] = 0.0f;
    }

    // Use the fused tensor product (inline CG coefficients)
    fused::tensor_product_simple_unrolled(in1, sh, out);

    // Write output to global memory (one write per thread)
    float* msg_out = messages + edge * C * NUM_IRREPS + channel * NUM_IRREPS;
    #pragma unroll
    for (int m = 0; m < NUM_IRREPS; m++) {
        msg_out[m] = out[m];
    }
}

cudaError_t fused_sh_tp_simple(
    const float* edge_vectors,
    const float* node_features,
    const int64_t* source_idx,
    float* messages,
    int64_t N_edges,
    int C,
    cudaStream_t stream
) {
    if (N_edges == 0) return cudaSuccess;

    const int64_t total_threads = N_edges * C;
    constexpr int block_size = 256;
    const int64_t num_blocks = (total_threads + block_size - 1) / block_size;

    fused_sh_tp_simple_kernel<<<num_blocks, block_size, 0, stream>>>(
        edge_vectors, node_features, source_idx, messages,
        N_edges, C
    );

    return cudaGetLastError();
}

// ============================================================================
// Weighted fused SH+TP kernel
// One thread per (edge, output_channel) pair
// ============================================================================

__global__ void fused_sh_tp_weighted_kernel(
    const float* __restrict__ edge_vectors,   // [N_edges, 3]
    const float* __restrict__ node_features,  // [N_atoms, C_in, 16]
    const int64_t* __restrict__ source_idx,   // [N_edges]
    const float* __restrict__ weights,        // [num_paths, C_in, C_out]
    float* __restrict__ messages,             // [N_edges, C_out, 16]
    int64_t N_edges,
    int C_in,
    int C_out
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t edge = idx / C_out;
    const int c_out = idx % C_out;

    if (edge >= N_edges) return;

    // Load edge vector
    const float x = edge_vectors[edge * 3 + 0];
    const float y = edge_vectors[edge * 3 + 1];
    const float z = edge_vectors[edge * 3 + 2];

    // Compute spherical harmonics in registers
    float sh[NUM_IRREPS];
    compute_sh_registers(x, y, z, sh);

    // Load source atom index
    const int64_t src = source_idx[edge];

    // Initialize output accumulator
    float out[NUM_IRREPS];
    #pragma unroll
    for (int m = 0; m < NUM_IRREPS; m++) {
        out[m] = 0.0f;
    }

    // For each input channel, compute weighted tensor product
    for (int c_in = 0; c_in < C_in; c_in++) {
        // Load source features for this input channel
        const float* src_feat = node_features + src * C_in * NUM_IRREPS + c_in * NUM_IRREPS;
        float in1[NUM_IRREPS];
        #pragma unroll
        for (int m = 0; m < NUM_IRREPS; m++) {
            in1[m] = src_feat[m];
        }

        // Compute tensor product for this input channel
        float tp_out[NUM_IRREPS];
        #pragma unroll
        for (int m = 0; m < NUM_IRREPS; m++) {
            tp_out[m] = 0.0f;
        }
        fused::tensor_product_simple_unrolled(in1, sh, tp_out);

        // Get weight for this (c_in, c_out) pair
        // Simplified: use single weight per channel pair (not per-path)
        // For full MACE, weights would be [num_paths, C_in, C_out]
        float w = weights[c_in * C_out + c_out];

        // Accumulate weighted output
        #pragma unroll
        for (int m = 0; m < NUM_IRREPS; m++) {
            out[m] += w * tp_out[m];
        }
    }

    // Write output to global memory
    float* msg_out = messages + edge * C_out * NUM_IRREPS + c_out * NUM_IRREPS;
    #pragma unroll
    for (int m = 0; m < NUM_IRREPS; m++) {
        msg_out[m] = out[m];
    }
}

cudaError_t fused_sh_tp_weighted(
    const float* edge_vectors,
    const float* node_features,
    const int64_t* source_idx,
    const float* weights,
    float* messages,
    int64_t N_edges,
    int C_in,
    int C_out,
    cudaStream_t stream
) {
    if (N_edges == 0) return cudaSuccess;

    const int64_t total_threads = N_edges * C_out;
    constexpr int block_size = 256;
    const int64_t num_blocks = (total_threads + block_size - 1) / block_size;

    fused_sh_tp_weighted_kernel<<<num_blocks, block_size, 0, stream>>>(
        edge_vectors, node_features, source_idx, weights, messages,
        N_edges, C_in, C_out
    );

    return cudaGetLastError();
}

} // namespace batteries
