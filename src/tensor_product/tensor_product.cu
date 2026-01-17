/**
 * Tensor Product CUDA Kernel - FUSED VERSION
 *
 * Computes equivariant tensor products with all CG coefficients as compile-time constants.
 * All 34 paths and 611 CG entries are fully unrolled.
 *
 * Memory layout assumptions:
 * - input1: [N, C_in, 16] - node features with channels (16 = (3+1)^2 components)
 * - input2: [N, 16] - edge features from spherical harmonics (no channels)
 * - output: [N, C_out, 16] - output features
 * - weights: [num_paths, C_in, C_out] - learnable weights per path
 */

#include "tensor_product.cuh"
#include "tensor_product_fused.cuh"
#include "cg_coefficients_l3.cuh"
#include <cstdio>

namespace batteries {

// Constants for L_max=3
constexpr int NUM_IRREPS = 16;  // (3+1)^2

/**
 * FUSED simple tensor product kernel (no channels, no weights).
 * Each thread handles one output sample.
 * All 611 CG operations are compile-time constants.
 */
__global__ void tensor_product_simple_kernel(
    const float* __restrict__ input1,  // [N, 16]
    const float* __restrict__ input2,  // [N, 16]
    float* __restrict__ output,        // [N, 16]
    int64_t N
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float* in1 = input1 + idx * NUM_IRREPS;
    const float* in2 = input2 + idx * NUM_IRREPS;
    float* out = output + idx * NUM_IRREPS;

    // Initialize output to zero
    #pragma unroll
    for (int i = 0; i < NUM_IRREPS; ++i) {
        out[i] = 0.0f;
    }

    // FUSED: All 34 paths and 611 CG entries unrolled at compile time
    fused::tensor_product_simple_unrolled(in1, in2, out);
}

/**
 * Fully connected tensor product kernel.
 *
 * Each block handles one (sample, output_channel) pair.
 * Threads within a block cooperatively process paths and CG entries.
 *
 * output[n, c_out, :] = Σ_{path} Σ_{c_in} weight[path, c_in, c_out] *
 *                       Σ_{cg ∈ path} cg_coeff * input1[n, c_in, m1] * input2[n, m2]
 */
__global__ void tensor_product_full_kernel(
    const float* __restrict__ input1,   // [N, C_in, 16]
    const float* __restrict__ input2,   // [N, 16]
    const float* __restrict__ weights,  // [num_paths, C_in, C_out]
    float* __restrict__ output,         // [N, C_out, 16]
    int64_t N,
    int C_in,
    int C_out
) {
    // Block handles one (sample, output_channel) pair
    const int64_t n = blockIdx.x;
    const int c_out = blockIdx.y;

    if (n >= N || c_out >= C_out) return;

    // Shared memory for output accumulation
    __shared__ float shared_out[NUM_IRREPS];

    // Initialize shared output
    if (threadIdx.x < NUM_IRREPS) {
        shared_out[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Pointers to this sample's data
    const float* in2 = input2 + n * NUM_IRREPS;

    // Each thread iterates over a subset of (path, c_in) combinations
    const int total_work = cg::NUM_BLOCKS * C_in;

    for (int work_idx = threadIdx.x; work_idx < total_work; work_idx += blockDim.x) {
        const int path_idx = work_idx / C_in;
        const int c_in = work_idx % C_in;

        const int offset1 = cg::BLOCK_INFO[path_idx][3];
        const int offset2 = cg::BLOCK_INFO[path_idx][4];
        const int offset_out = cg::BLOCK_INFO[path_idx][5];

        // Get weight for this path
        const float w = weights[path_idx * C_in * C_out + c_in * C_out + c_out];

        // Pointer to input1 for this sample and channel
        const float* in1 = input1 + n * C_in * NUM_IRREPS + c_in * NUM_IRREPS;

        // Process all CG entries in this path
        const int start = cg::BLOCK_RANGES[path_idx][0];
        const int end = cg::BLOCK_RANGES[path_idx][1];

        for (int entry = start; entry < end; ++entry) {
            const int m1_idx = cg::ENTRY_INFO[entry][0];
            const int m2_idx = cg::ENTRY_INFO[entry][1];
            const int m_out_idx = cg::ENTRY_INFO[entry][2];
            const float cg_coeff = cg::CG_COEFFS[entry];

            const float contrib = w * cg_coeff * in1[offset1 + m1_idx] * in2[offset2 + m2_idx];
            atomicAdd(&shared_out[offset_out + m_out_idx], contrib);
        }
    }

    __syncthreads();

    // Write shared output to global memory
    if (threadIdx.x < NUM_IRREPS) {
        output[n * C_out * NUM_IRREPS + c_out * NUM_IRREPS + threadIdx.x] = shared_out[threadIdx.x];
    }
}

/**
 * FUSED tensor product kernel with compile-time CG constants.
 * Each thread handles one (sample, c_out) pair completely.
 * All 34 paths and 611 CG entries are fully unrolled.
 */
__global__ void tensor_product_batch_kernel(
    const float* __restrict__ input1,   // [N, C_in, 16]
    const float* __restrict__ input2,   // [N, 16]
    const float* __restrict__ weights,  // [num_paths, C_in, C_out]
    float* __restrict__ output,         // [N, C_out, 16]
    int64_t N,
    int C_in,
    int C_out
) {
    // Each thread handles one (sample, c_out) pair
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = idx / C_out;
    const int c_out = idx % C_out;

    if (n >= N) return;

    // Pointers to this sample's data
    const float* in1 = input1 + n * C_in * NUM_IRREPS;
    const float* in2 = input2 + n * NUM_IRREPS;
    float* out = output + n * C_out * NUM_IRREPS + c_out * NUM_IRREPS;

    // FUSED: Use compile-time unrolled tensor product
    fused::tensor_product_full_dynamic(in1, in2, weights, out, C_in, C_out, c_out);
}

// Kernel launch wrappers
cudaError_t tensor_product_simple_l3(
    const float* input1,
    const float* input2,
    float* output,
    int64_t N,
    cudaStream_t stream
) {
    if (N == 0) return cudaSuccess;

    constexpr int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    tensor_product_simple_kernel<<<num_blocks, block_size, 0, stream>>>(
        input1, input2, output, N
    );

    return cudaGetLastError();
}

cudaError_t tensor_product_l3(
    const float* input1,
    const float* input2,
    const float* weights,
    float* output,
    int64_t N,
    int C_in,
    int C_out,
    cudaStream_t stream
) {
    if (N == 0) return cudaSuccess;

    // FUSED kernel: each thread handles one (sample, c_out) pair
    // Total work items = N * C_out
    const int64_t total_work = N * C_out;
    constexpr int block_size = 128;  // Smaller block for better occupancy with high register usage
    const int num_blocks = (total_work + block_size - 1) / block_size;

    tensor_product_batch_kernel<<<num_blocks, block_size, 0, stream>>>(
        input1, input2, weights, output, N, C_in, C_out
    );

    return cudaGetLastError();
}

int get_num_paths_l3() {
    return cg::NUM_BLOCKS;
}

void get_path_info_l3(int* path_info) {
    // Use HOST_PATH_INFO since BLOCK_INFO is in __device__ __constant__ memory
    for (int i = 0; i < cg::NUM_BLOCKS; ++i) {
        path_info[i * 3 + 0] = cg::HOST_PATH_INFO[i][0];  // l1
        path_info[i * 3 + 1] = cg::HOST_PATH_INFO[i][1];  // l2
        path_info[i * 3 + 2] = cg::HOST_PATH_INFO[i][2];  // l_out
    }
}

} // namespace batteries
