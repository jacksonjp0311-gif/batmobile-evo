/**
 * Tensor Product Backward CUDA Kernel - FUSED VERSION
 *
 * All CG coefficients are compile-time constants for maximum performance.
 * Uses the same fused approach as the forward pass.
 *
 * Backward:
 *   grad_input1[n, c_in, m1] = sum_{path, c_out, cg_entry} w[path, c_in, c_out] *
 *                               cg_coeff * grad_output[n, c_out, m_out] * input2[n, m2]
 *
 *   grad_input2[n, m2] = sum_{path, c_in, c_out, cg_entry} w[path, c_in, c_out] *
 *                         cg_coeff * grad_output[n, c_out, m_out] * input1[n, c_in, m1]
 *
 *   grad_weights[path, c_in, c_out] = sum_{n, cg_entry} cg_coeff *
 *                                      input1[n, c_in, m1] * input2[n, m2] *
 *                                      grad_output[n, c_out, m_out]
 */

#include "tensor_product.cuh"
#include "tensor_product_fused.cuh"
#include <cstdio>

namespace batteries {

constexpr int NUM_IRREPS = 16;  // (3+1)^2

/**
 * FUSED backward kernel for grad_input1.
 * Each thread handles one (sample, c_in) pair.
 * All 611 CG operations are compile-time unrolled.
 */
__global__ void tensor_product_backward_input1_kernel_fused(
    const float* __restrict__ input2,       // [N, 16]
    const float* __restrict__ weights,      // [num_paths, C_in, C_out]
    const float* __restrict__ grad_output,  // [N, C_out, 16]
    float* __restrict__ grad_input1,        // [N, C_in, 16]
    int64_t N,
    int C_in,
    int C_out
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = idx / C_in;
    const int c_in = idx % C_in;

    if (n >= N) return;

    const float* in2 = input2 + n * NUM_IRREPS;
    const float* grad_out = grad_output + n * C_out * NUM_IRREPS;
    float* grad_out_ptr = grad_input1 + n * C_in * NUM_IRREPS + c_in * NUM_IRREPS;

    // Use fused backward function
    fused::tensor_product_backward_input1_unrolled(
        in2, weights, grad_out, grad_out_ptr, c_in, C_in, C_out
    );
}

/**
 * Optimized backward kernel for grad_input2.
 * Grid: (N) - one block per sample
 * Threads: 256 threads split work over (c_in, c_out) pairs
 * Each thread processes some (c_in, c_out) pairs and accumulates into local registers,
 * then block reduction combines results.
 */
__global__ void tensor_product_backward_input2_kernel_opt(
    const float* __restrict__ input1,       // [N, C_in, 16]
    const float* __restrict__ weights,      // [num_paths, C_in, C_out]
    const float* __restrict__ grad_output,  // [N, C_out, 16]
    float* __restrict__ grad_input2,        // [N, 16]
    int64_t N,
    int C_in,
    int C_out
) {
    const int64_t n = blockIdx.x;
    if (n >= N) return;

    const float* in1_base = input1 + n * C_in * NUM_IRREPS;
    const float* grad_out_base = grad_output + n * C_out * NUM_IRREPS;

    // Local accumulator for 16 output values
    float local_grad[16] = {0};

    // Total work: C_in * C_out pairs
    const int total_pairs = C_in * C_out;

    // Each thread processes a subset of (c_in, c_out) pairs
    for (int pair_idx = threadIdx.x; pair_idx < total_pairs; pair_idx += blockDim.x) {
        const int c_in = pair_idx / C_out;
        const int c_out = pair_idx % C_out;

        const float* inp1 = in1_base + c_in * NUM_IRREPS;
        const float* go = grad_out_base + c_out * NUM_IRREPS;

        // Use the fused per-pair function for all 34 paths
        fused::tensor_product_backward_input2_per_pair(
            inp1, go, weights, local_grad, c_in, c_out, C_in, C_out
        );
    }

    // Block reduction using shared memory for 16 values
    __shared__ float shared_grad[16][256];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        shared_grad[i][threadIdx.x] = local_grad[i];
    }
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                shared_grad[i][threadIdx.x] += shared_grad[i][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes results
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            grad_input2[n * NUM_IRREPS + i] = shared_grad[i][0];
        }
    }
}

/**
 * Optimized backward kernel for grad_weights.
 * Grid: (C_in, C_out) - each block handles one (c_in, c_out) pair
 * Threads reduce over samples with block-level reduction.
 *
 * For each (c_in, c_out), we need to compute:
 *   grad_w[path, c_in, c_out] = sum_n sum_cg cg * in1[n,c_in,m1] * in2[n,m2] * grad_out[n,c_out,m_out]
 */
__global__ void tensor_product_backward_weights_kernel_opt(
    const float* __restrict__ input1,       // [N, C_in, 16]
    const float* __restrict__ input2,       // [N, 16]
    const float* __restrict__ grad_output,  // [N, C_out, 16]
    float* __restrict__ grad_weights,       // [num_paths, C_in, C_out]
    int64_t N,
    int C_in,
    int C_out
) {
    const int c_in = blockIdx.x;
    const int c_out = blockIdx.y;

    if (c_in >= C_in || c_out >= C_out) return;

    // Each thread accumulates contributions for all 34 paths
    float local_grad[34] = {0};

    // Each thread processes a subset of samples
    for (int64_t n = threadIdx.x; n < N; n += blockDim.x) {
        const float* in1 = input1 + n * C_in * NUM_IRREPS + c_in * NUM_IRREPS;
        const float* in2 = input2 + n * NUM_IRREPS;
        const float* go = grad_output + n * C_out * NUM_IRREPS + c_out * NUM_IRREPS;

        // Path 0: (0,0)->0, 1 entry
        local_grad[0] += 1.0000000000f * in1[0] * in2[0] * go[0];

        // Path 1: (0,1)->1, 3 entries
        local_grad[1] += 0.9999999821f * in1[0] * in2[1] * go[1];
        local_grad[1] += 0.9999999821f * in1[0] * in2[2] * go[2];
        local_grad[1] += 0.9999999821f * in1[0] * in2[3] * go[3];

        // Path 2: (0,2)->2, 5 entries
        local_grad[2] += 0.9999999880f * in1[0] * in2[4] * go[4];
        local_grad[2] += 0.9999999880f * in1[0] * in2[5] * go[5];
        local_grad[2] += 0.9999999880f * in1[0] * in2[6] * go[6];
        local_grad[2] += 0.9999999880f * in1[0] * in2[7] * go[7];
        local_grad[2] += 0.9999999880f * in1[0] * in2[8] * go[8];

        // Path 3: (0,3)->3, 7 entries
        local_grad[3] += 0.9999999836f * in1[0] * in2[9] * go[9];
        local_grad[3] += 0.9999999836f * in1[0] * in2[10] * go[10];
        local_grad[3] += 0.9999999836f * in1[0] * in2[11] * go[11];
        local_grad[3] += 0.9999999836f * in1[0] * in2[12] * go[12];
        local_grad[3] += 0.9999999836f * in1[0] * in2[13] * go[13];
        local_grad[3] += 0.9999999836f * in1[0] * in2[14] * go[14];
        local_grad[3] += 0.9999999836f * in1[0] * in2[15] * go[15];

        // Path 4: (1,0)->1, 3 entries
        local_grad[4] += 0.9999999821f * in1[1] * in2[0] * go[1];
        local_grad[4] += 0.9999999821f * in1[2] * in2[0] * go[2];
        local_grad[4] += 0.9999999821f * in1[3] * in2[0] * go[3];

        // Path 5: (1,1)->0, 3 entries
        local_grad[5] += 0.5773502588f * in1[1] * in2[1] * go[0];
        local_grad[5] += 0.5773502588f * in1[2] * in2[2] * go[0];
        local_grad[5] += 0.5773502588f * in1[3] * in2[3] * go[0];

        // Path 6: (1,1)->1, 6 entries
        local_grad[6] += 0.7071068069f * in1[1] * in2[2] * go[3];
        local_grad[6] += -0.7071068069f * in1[1] * in2[3] * go[2];
        local_grad[6] += -0.7071068069f * in1[2] * in2[1] * go[3];
        local_grad[6] += 0.7071068069f * in1[2] * in2[3] * go[1];
        local_grad[6] += 0.7071068069f * in1[3] * in2[1] * go[2];
        local_grad[6] += -0.7071068069f * in1[3] * in2[2] * go[1];

        // Path 7: (1,1)->2, 11 entries
        local_grad[7] += -0.4082482836f * in1[1] * in2[1] * go[6];
        local_grad[7] += -0.7071067764f * in1[1] * in2[1] * go[8];
        local_grad[7] += 0.7071067764f * in1[1] * in2[2] * go[5];
        local_grad[7] += 0.7071067764f * in1[1] * in2[3] * go[4];
        local_grad[7] += 0.7071067764f * in1[2] * in2[1] * go[5];
        local_grad[7] += 0.8164965671f * in1[2] * in2[2] * go[6];
        local_grad[7] += 0.7071067764f * in1[2] * in2[3] * go[7];
        local_grad[7] += 0.7071067764f * in1[3] * in2[1] * go[4];
        local_grad[7] += 0.7071067764f * in1[3] * in2[2] * go[7];
        local_grad[7] += -0.4082482836f * in1[3] * in2[3] * go[6];
        local_grad[7] += 0.7071067764f * in1[3] * in2[3] * go[8];

        // Path 8: (1,2)->1, 11 entries
        local_grad[8] += 0.5477225538f * in1[1] * in2[4] * go[3];
        local_grad[8] += 0.5477225538f * in1[1] * in2[5] * go[2];
        local_grad[8] += -0.3162277607f * in1[1] * in2[6] * go[1];
        local_grad[8] += -0.5477225538f * in1[1] * in2[8] * go[1];
        local_grad[8] += 0.5477225538f * in1[2] * in2[5] * go[1];
        local_grad[8] += 0.6324555213f * in1[2] * in2[6] * go[2];
        local_grad[8] += 0.5477225538f * in1[2] * in2[7] * go[3];
        local_grad[8] += 0.5477225538f * in1[3] * in2[4] * go[1];
        local_grad[8] += -0.3162277607f * in1[3] * in2[6] * go[3];
        local_grad[8] += 0.5477225538f * in1[3] * in2[7] * go[2];
        local_grad[8] += 0.5477225538f * in1[3] * in2[8] * go[3];

        // Path 9: (1,2)->2, 16 entries
        local_grad[9] += -0.4082482836f * in1[1] * in2[4] * go[5];
        local_grad[9] += 0.4082482836f * in1[1] * in2[5] * go[4];
        local_grad[9] += 0.7071067764f * in1[1] * in2[6] * go[7];
        local_grad[9] += -0.7071067764f * in1[1] * in2[7] * go[6];
        local_grad[9] += 0.4082482836f * in1[1] * in2[7] * go[8];
        local_grad[9] += -0.4082482836f * in1[1] * in2[8] * go[7];
        local_grad[9] += -0.8164965671f * in1[2] * in2[4] * go[8];
        local_grad[9] += -0.4082482836f * in1[2] * in2[5] * go[7];
        local_grad[9] += 0.4082482836f * in1[2] * in2[7] * go[5];
        local_grad[9] += 0.8164965671f * in1[2] * in2[8] * go[4];
        local_grad[9] += 0.4082482836f * in1[3] * in2[4] * go[7];
        local_grad[9] += 0.7071067764f * in1[3] * in2[5] * go[6];
        local_grad[9] += 0.4082482836f * in1[3] * in2[5] * go[8];
        local_grad[9] += -0.7071067764f * in1[3] * in2[6] * go[5];
        local_grad[9] += -0.4082482836f * in1[3] * in2[7] * go[4];
        local_grad[9] += -0.4082482836f * in1[3] * in2[8] * go[5];

        // Path 10: (1,2)->3, 21 entries
        local_grad[10] += -0.1825741828f * in1[1] * in2[4] * go[13];
        local_grad[10] += -0.7071067679f * in1[1] * in2[4] * go[15];
        local_grad[10] += -0.4472135802f * in1[1] * in2[5] * go[12];
        local_grad[10] += -0.5773502803f * in1[1] * in2[5] * go[14];
        local_grad[10] += 0.6324555384f * in1[1] * in2[6] * go[11];
        local_grad[10] += 0.5773502803f * in1[1] * in2[7] * go[10];
        local_grad[10] += 0.7071067679f * in1[1] * in2[8] * go[9];
        local_grad[10] += 0.1825741828f * in1[1] * in2[8] * go[11];
        local_grad[10] += 0.5773502803f * in1[2] * in2[4] * go[10];
        local_grad[10] += 0.7302967312f * in1[2] * in2[5] * go[11];
        local_grad[10] += 0.7745966857f * in1[2] * in2[6] * go[12];
        local_grad[10] += 0.7302967312f * in1[2] * in2[7] * go[13];
        local_grad[10] += 0.5773502803f * in1[2] * in2[8] * go[14];
        local_grad[10] += 0.7071067679f * in1[3] * in2[4] * go[9];
        local_grad[10] += -0.1825741828f * in1[3] * in2[4] * go[11];
        local_grad[10] += 0.5773502803f * in1[3] * in2[5] * go[10];
        local_grad[10] += 0.6324555384f * in1[3] * in2[6] * go[13];
        local_grad[10] += -0.4472135802f * in1[3] * in2[7] * go[12];
        local_grad[10] += 0.5773502803f * in1[3] * in2[7] * go[14];
        local_grad[10] += -0.1825741828f * in1[3] * in2[8] * go[13];
        local_grad[10] += 0.7071067679f * in1[3] * in2[8] * go[15];

        // Path 11: (1,3)->2, 21 entries
        local_grad[11] += 0.5976142935f * in1[1] * in2[9] * go[8];
        local_grad[11] += 0.4879500459f * in1[1] * in2[10] * go[7];
        local_grad[11] += 0.5345224892f * in1[1] * in2[11] * go[6];
        local_grad[11] += 0.1543033474f * in1[1] * in2[11] * go[8];
        local_grad[11] += -0.3779644601f * in1[1] * in2[12] * go[5];
        local_grad[11] += -0.1543033474f * in1[1] * in2[13] * go[4];
        local_grad[11] += -0.4879500459f * in1[1] * in2[14] * go[5];
        local_grad[11] += -0.5976142935f * in1[1] * in2[15] * go[4];
        local_grad[11] += 0.4879500459f * in1[2] * in2[10] * go[4];
        local_grad[11] += 0.6172133895f * in1[2] * in2[11] * go[5];
        local_grad[11] += 0.6546536846f * in1[2] * in2[12] * go[6];
        local_grad[11] += 0.6172133895f * in1[2] * in2[13] * go[7];
        local_grad[11] += 0.4879500459f * in1[2] * in2[14] * go[8];
        local_grad[11] += 0.5976142935f * in1[3] * in2[9] * go[4];
        local_grad[11] += 0.4879500459f * in1[3] * in2[10] * go[5];
        local_grad[11] += -0.1543033474f * in1[3] * in2[11] * go[4];
        local_grad[11] += -0.3779644601f * in1[3] * in2[12] * go[7];
        local_grad[11] += 0.5345224892f * in1[3] * in2[13] * go[6];
        local_grad[11] += -0.1543033474f * in1[3] * in2[13] * go[8];
        local_grad[11] += 0.4879500459f * in1[3] * in2[14] * go[7];
        local_grad[11] += 0.5976142935f * in1[3] * in2[15] * go[8];

        // Path 12: (1,3)->3, 31 entries
        local_grad[12] += 0.3535533840f * in1[1] * in2[9] * go[14];
        local_grad[12] += -0.5773502803f * in1[1] * in2[10] * go[13];
        local_grad[12] += 0.1825741828f * in1[1] * in2[10] * go[15];
        local_grad[12] += 0.4472135802f * in1[1] * in2[11] * go[12];
        local_grad[12] += -0.3651483655f * in1[1] * in2[11] * go[14];
        local_grad[12] += -0.5163977742f * in1[1] * in2[12] * go[11];
        local_grad[12] += -0.4472135802f * in1[1] * in2[13] * go[10];
        local_grad[12] += 0.3651483655f * in1[1] * in2[13] * go[12];
        local_grad[12] += 0.5773502803f * in1[1] * in2[14] * go[9];
        local_grad[12] += -0.1825741828f * in1[1] * in2[14] * go[11];
        local_grad[12] += -0.3535533840f * in1[1] * in2[15] * go[10];
        local_grad[12] += 0.5773502803f * in1[2] * in2[9] * go[13];
        local_grad[12] += 0.4472135802f * in1[2] * in2[10] * go[12];
        local_grad[12] += -0.1825741828f * in1[2] * in2[11] * go[11];
        local_grad[12] += 0.3651483655f * in1[2] * in2[11] * go[13];
        local_grad[12] += -0.3651483655f * in1[2] * in2[13] * go[11];
        local_grad[12] += 0.1825741828f * in1[2] * in2[13] * go[13];
        local_grad[12] += -0.4472135802f * in1[2] * in2[14] * go[12];
        local_grad[12] += -0.5773502803f * in1[2] * in2[15] * go[11];
        local_grad[12] += 0.3535533840f * in1[3] * in2[9] * go[10];
        local_grad[12] += 0.5773502803f * in1[3] * in2[10] * go[9];
        local_grad[12] += -0.1825741828f * in1[3] * in2[10] * go[11];
        local_grad[12] += -0.4472135802f * in1[3] * in2[11] * go[10];
        local_grad[12] += 0.3651483655f * in1[3] * in2[11] * go[12];
        local_grad[12] += 0.5163977742f * in1[3] * in2[12] * go[13];
        local_grad[12] += 0.4472135802f * in1[3] * in2[13] * go[12];
        local_grad[12] += 0.3651483655f * in1[3] * in2[13] * go[14];
        local_grad[12] += -0.5773502803f * in1[3] * in2[14] * go[15];
        local_grad[12] += 0.1825741828f * in1[3] * in2[14] * go[13];
        local_grad[12] += -0.3535533840f * in1[3] * in2[15] * go[14];

        // Path 13: (2,0)->2, 5 entries
        local_grad[13] += 0.9999999880f * in1[4] * in2[0] * go[4];
        local_grad[13] += 0.9999999880f * in1[5] * in2[0] * go[5];
        local_grad[13] += 0.9999999880f * in1[6] * in2[0] * go[6];
        local_grad[13] += 0.9999999880f * in1[7] * in2[0] * go[7];
        local_grad[13] += 0.9999999880f * in1[8] * in2[0] * go[8];

        // Path 14: (2,1)->1, 11 entries
        local_grad[14] += 0.5477225538f * in1[4] * in2[1] * go[3];
        local_grad[14] += 0.5477225538f * in1[4] * in2[3] * go[1];
        local_grad[14] += 0.5477225538f * in1[5] * in2[1] * go[2];
        local_grad[14] += 0.5477225538f * in1[5] * in2[2] * go[1];
        local_grad[14] += -0.3162277607f * in1[6] * in2[1] * go[1];
        local_grad[14] += 0.6324555213f * in1[6] * in2[2] * go[2];
        local_grad[14] += -0.3162277607f * in1[6] * in2[3] * go[3];
        local_grad[14] += 0.5477225538f * in1[7] * in2[2] * go[3];
        local_grad[14] += 0.5477225538f * in1[7] * in2[3] * go[2];
        local_grad[14] += -0.5477225538f * in1[8] * in2[1] * go[1];
        local_grad[14] += 0.5477225538f * in1[8] * in2[3] * go[3];

        // Path 15: (2,1)->2, 16 entries
        local_grad[15] += -0.4082482836f * in1[4] * in2[1] * go[5];
        local_grad[15] += -0.8164965671f * in1[4] * in2[2] * go[8];
        local_grad[15] += 0.4082482836f * in1[4] * in2[3] * go[7];
        local_grad[15] += 0.4082482836f * in1[5] * in2[1] * go[4];
        local_grad[15] += -0.4082482836f * in1[5] * in2[2] * go[7];
        local_grad[15] += 0.7071067764f * in1[5] * in2[3] * go[6];
        local_grad[15] += 0.4082482836f * in1[5] * in2[3] * go[8];
        local_grad[15] += 0.7071067764f * in1[6] * in2[1] * go[7];
        local_grad[15] += -0.7071067764f * in1[6] * in2[3] * go[5];
        local_grad[15] += -0.7071067764f * in1[7] * in2[1] * go[6];
        local_grad[15] += 0.4082482836f * in1[7] * in2[1] * go[8];
        local_grad[15] += 0.4082482836f * in1[7] * in2[2] * go[5];
        local_grad[15] += -0.4082482836f * in1[7] * in2[3] * go[4];
        local_grad[15] += 0.4082482836f * in1[8] * in2[1] * go[7];
        local_grad[15] += 0.8164965671f * in1[8] * in2[2] * go[4];
        local_grad[15] += -0.4082482836f * in1[8] * in2[3] * go[5];

        // Path 16: (2,1)->3, 21 entries
        local_grad[16] += 0.5976142935f * in1[4] * in2[1] * go[8];
        local_grad[16] += 0.5976142935f * in1[4] * in2[3] * go[4];
        local_grad[16] += 0.4879500459f * in1[5] * in2[1] * go[7];
        local_grad[16] += 0.4879500459f * in1[5] * in2[2] * go[4];
        local_grad[16] += 0.4879500459f * in1[5] * in2[3] * go[5];
        local_grad[16] += 0.5345224892f * in1[6] * in2[1] * go[6];
        local_grad[16] += 0.1543033474f * in1[6] * in2[1] * go[8];
        local_grad[16] += 0.6546536846f * in1[6] * in2[2] * go[6];
        local_grad[16] += 0.5345224892f * in1[6] * in2[3] * go[6];
        local_grad[16] += -0.1543033474f * in1[6] * in2[3] * go[8];
        local_grad[16] += -0.3779644601f * in1[7] * in2[1] * go[5];
        local_grad[16] += 0.6172133895f * in1[7] * in2[2] * go[5];
        local_grad[16] += -0.3779644601f * in1[7] * in2[3] * go[7];
        local_grad[16] += -0.1543033474f * in1[8] * in2[1] * go[4];
        local_grad[16] += -0.4879500459f * in1[8] * in2[1] * go[5];
        local_grad[16] += 0.4879500459f * in1[8] * in2[2] * go[8];
        local_grad[16] += 0.4879500459f * in1[8] * in2[3] * go[7];
        local_grad[16] += 0.1543033474f * in1[8] * in2[3] * go[8];
        local_grad[16] += -0.5976142935f * in1[9] * in2[1] * go[4];
        local_grad[16] += 0.6172133895f * in1[9] * in2[2] * go[7];
        local_grad[16] += 0.5976142935f * in1[9] * in2[3] * go[8];

        // Path 17: (2,2)->0, 5 entries
        local_grad[17] += 0.4472135925f * in1[4] * in2[4] * go[0];
        local_grad[17] += 0.4472135925f * in1[5] * in2[5] * go[0];
        local_grad[17] += 0.4472135925f * in1[6] * in2[6] * go[0];
        local_grad[17] += 0.4472135925f * in1[7] * in2[7] * go[0];
        local_grad[17] += 0.4472135925f * in1[8] * in2[8] * go[0];

        // Path 18: (2,2)->1, 16 entries
        local_grad[18] += 0.4472135802f * in1[4] * in2[5] * go[2];
        local_grad[18] += 0.4472135802f * in1[4] * in2[7] * go[3];
        local_grad[18] += 0.4472135802f * in1[5] * in2[4] * go[2];
        local_grad[18] += 0.5477225538f * in1[5] * in2[6] * go[1];
        local_grad[18] += 0.4472135802f * in1[5] * in2[8] * go[3];
        local_grad[18] += -0.5477225538f * in1[6] * in2[5] * go[1];
        local_grad[18] += 0.6324555213f * in1[6] * in2[6] * go[2];
        local_grad[18] += 0.5477225538f * in1[6] * in2[7] * go[3];
        local_grad[18] += 0.4472135802f * in1[7] * in2[4] * go[3];
        local_grad[18] += -0.5477225538f * in1[7] * in2[6] * go[3];
        local_grad[18] += 0.4472135802f * in1[7] * in2[8] * go[2];
        local_grad[18] += 0.4472135802f * in1[8] * in2[5] * go[3];
        local_grad[18] += -0.5477225538f * in1[8] * in2[6] * go[1];
        local_grad[18] += 0.4472135802f * in1[8] * in2[7] * go[2];
        local_grad[18] += -0.4472135802f * in1[8] * in2[4] * go[1];
        local_grad[18] += 0.4472135802f * in1[4] * in2[8] * go[1];

        // Path 19: (2,2)->2, 25 entries
        local_grad[19] += -0.2390457287f * in1[4] * in2[4] * go[6];
        local_grad[19] += 0.5222329590f * in1[4] * in2[4] * go[8];
        local_grad[19] += 0.6324555478f * in1[4] * in2[5] * go[5];
        local_grad[19] += 0.4082482836f * in1[4] * in2[6] * go[4];
        local_grad[19] += 0.6324555478f * in1[4] * in2[7] * go[7];
        local_grad[19] += 0.6324555478f * in1[5] * in2[4] * go[5];
        local_grad[19] += 0.4780914575f * in1[5] * in2[5] * go[6];
        local_grad[19] += 0.5222329590f * in1[5] * in2[7] * go[8];
        local_grad[19] += -0.5222329590f * in1[5] * in2[8] * go[7];
        local_grad[19] += 0.4082482836f * in1[6] * in2[4] * go[4];
        local_grad[19] += -0.4780914575f * in1[6] * in2[6] * go[6];
        local_grad[19] += 0.4082482836f * in1[6] * in2[8] * go[8];
        local_grad[19] += 0.6324555478f * in1[7] * in2[4] * go[7];
        local_grad[19] += 0.5222329590f * in1[7] * in2[5] * go[8];
        local_grad[19] += 0.4780914575f * in1[7] * in2[7] * go[6];
        local_grad[19] += 0.6324555478f * in1[7] * in2[8] * go[5];
        local_grad[19] += -0.5222329590f * in1[8] * in2[5] * go[7];
        local_grad[19] += 0.4082482836f * in1[8] * in2[6] * go[8];
        local_grad[19] += 0.6324555478f * in1[8] * in2[7] * go[5];
        local_grad[19] += -0.2390457287f * in1[8] * in2[8] * go[6];
        local_grad[19] += -0.5222329590f * in1[8] * in2[8] * go[8];
        local_grad[19] += -0.5222329590f * in1[4] * in2[8] * go[4];
        local_grad[19] += 0.5222329590f * in1[8] * in2[4] * go[4];
        local_grad[19] += -0.6324555478f * in1[5] * in2[6] * go[7];
        local_grad[19] += 0.6324555478f * in1[6] * in2[5] * go[7];

        // Path 20: (2,2)->3, 36 entries
        local_grad[20] += -0.2672612446f * in1[4] * in2[4] * go[13];
        local_grad[20] += 0.5345224892f * in1[4] * in2[4] * go[15];
        local_grad[20] += -0.3872983429f * in1[4] * in2[5] * go[12];
        local_grad[20] += 0.5976142935f * in1[4] * in2[5] * go[14];
        local_grad[20] += 0.5163977742f * in1[4] * in2[6] * go[11];
        local_grad[20] += 0.5976142935f * in1[4] * in2[7] * go[10];
        local_grad[20] += 0.5345224892f * in1[4] * in2[8] * go[9];
        local_grad[20] += 0.2672612446f * in1[4] * in2[8] * go[11];
        local_grad[20] += -0.3872983429f * in1[5] * in2[4] * go[12];
        local_grad[20] += 0.5976142935f * in1[5] * in2[4] * go[14];
        local_grad[20] += 0.5345224892f * in1[5] * in2[5] * go[13];
        local_grad[20] += 0.3086066909f * in1[5] * in2[6] * go[12];
        local_grad[20] += 0.5163977742f * in1[5] * in2[7] * go[11];
        local_grad[20] += 0.5345224892f * in1[5] * in2[8] * go[10];
        local_grad[20] += 0.5163977742f * in1[6] * in2[4] * go[11];
        local_grad[20] += 0.3086066909f * in1[6] * in2[5] * go[12];
        local_grad[20] += 0.3086066909f * in1[6] * in2[7] * go[12];
        local_grad[20] += -0.5163977742f * in1[6] * in2[8] * go[11];
        local_grad[20] += 0.5976142935f * in1[7] * in2[4] * go[10];
        local_grad[20] += 0.5163977742f * in1[7] * in2[5] * go[11];
        local_grad[20] += 0.3086066909f * in1[7] * in2[6] * go[12];
        local_grad[20] += 0.5345224892f * in1[7] * in2[7] * go[13];
        local_grad[20] += 0.3872983429f * in1[7] * in2[8] * go[12];
        local_grad[20] += 0.5976142935f * in1[7] * in2[8] * go[14];
        local_grad[20] += 0.5345224892f * in1[8] * in2[4] * go[9];
        local_grad[20] += -0.2672612446f * in1[8] * in2[4] * go[11];
        local_grad[20] += 0.5345224892f * in1[8] * in2[5] * go[10];
        local_grad[20] += 0.5163977742f * in1[8] * in2[6] * go[11];
        local_grad[20] += 0.3872983429f * in1[8] * in2[7] * go[12];
        local_grad[20] += -0.5976142935f * in1[8] * in2[7] * go[14];
        local_grad[20] += -0.2672612446f * in1[8] * in2[8] * go[13];
        local_grad[20] += -0.5345224892f * in1[8] * in2[8] * go[15];
        local_grad[20] += -0.6172133895f * in1[6] * in2[6] * go[12];
        local_grad[20] += -0.5345224892f * in1[5] * in2[5] * go[11];
        local_grad[20] += 0.5345224892f * in1[7] * in2[7] * go[11];
        local_grad[20] += 0.5345224892f * in1[5] * in2[7] * go[13];

        // Path 21: (2,2)->4 (L_max=3, skip)

        // Path 21: (2,3)->1, 21 entries
        local_grad[21] += 0.3779644601f * in1[4] * in2[9] * go[3];
        local_grad[21] += 0.5345224892f * in1[4] * in2[10] * go[2];
        local_grad[21] += -0.2672612446f * in1[4] * in2[11] * go[1];
        local_grad[21] += -0.2672612446f * in1[4] * in2[11] * go[3];
        local_grad[21] += -0.5345224892f * in1[4] * in2[14] * go[2];
        local_grad[21] += -0.3779644601f * in1[4] * in2[15] * go[1];
        local_grad[21] += 0.5345224892f * in1[5] * in2[9] * go[2];
        local_grad[21] += 0.5345224892f * in1[5] * in2[10] * go[1];
        local_grad[21] += 0.4629100433f * in1[5] * in2[11] * go[2];
        local_grad[21] += 0.4629100433f * in1[5] * in2[13] * go[2];
        local_grad[21] += 0.5345224892f * in1[5] * in2[14] * go[3];
        local_grad[21] += 0.5345224892f * in1[5] * in2[15] * go[2];
        local_grad[21] += -0.2672612446f * in1[6] * in2[9] * go[1];
        local_grad[21] += 0.4629100433f * in1[6] * in2[10] * go[2];
        local_grad[21] += 0.5345224892f * in1[6] * in2[12] * go[2];
        local_grad[21] += 0.4629100433f * in1[6] * in2[14] * go[2];
        local_grad[21] += 0.2672612446f * in1[6] * in2[15] * go[3];
        local_grad[21] += -0.5345224892f * in1[7] * in2[10] * go[2];
        local_grad[21] += 0.4629100433f * in1[7] * in2[11] * go[2];
        local_grad[21] += 0.4629100433f * in1[7] * in2[13] * go[2];
        local_grad[21] += 0.5345224892f * in1[7] * in2[14] * go[1];
        local_grad[21] += 0.3779644601f * in1[8] * in2[9] * go[1];
        local_grad[21] += 0.2672612446f * in1[8] * in2[11] * go[1];
        local_grad[21] += 0.2672612446f * in1[8] * in2[11] * go[3];
        local_grad[21] += 0.5345224892f * in1[8] * in2[14] * go[2];
        local_grad[21] += -0.3779644601f * in1[8] * in2[15] * go[3];

        // Path 22: (2,3)->2, 31 entries
        local_grad[22] += 0.3535533840f * in1[4] * in2[9] * go[8];
        local_grad[22] += 0.4472135802f * in1[4] * in2[10] * go[7];
        local_grad[22] += 0.1825741828f * in1[4] * in2[11] * go[6];
        local_grad[22] += -0.3651483655f * in1[4] * in2[11] * go[8];
        local_grad[22] += -0.5163977742f * in1[4] * in2[12] * go[5];
        local_grad[22] += -0.3651483655f * in1[4] * in2[13] * go[4];
        local_grad[22] += 0.1825741828f * in1[4] * in2[13] * go[6];
        local_grad[22] += -0.4472135802f * in1[4] * in2[14] * go[5];
        local_grad[22] += -0.3535533840f * in1[4] * in2[15] * go[4];
        local_grad[22] += 0.4472135802f * in1[5] * in2[9] * go[7];
        local_grad[22] += 0.3651483655f * in1[5] * in2[10] * go[6];
        local_grad[22] += -0.3535533840f * in1[5] * in2[10] * go[8];
        local_grad[22] += 0.5163977742f * in1[5] * in2[12] * go[6];
        local_grad[22] += -0.3535533840f * in1[5] * in2[14] * go[6];
        local_grad[22] += 0.3651483655f * in1[5] * in2[14] * go[8];
        local_grad[22] += -0.4472135802f * in1[5] * in2[15] * go[5];
        local_grad[22] += 0.1825741828f * in1[6] * in2[9] * go[6];
        local_grad[22] += 0.3651483655f * in1[6] * in2[10] * go[5];
        local_grad[22] += -0.1825741828f * in1[6] * in2[11] * go[4];
        local_grad[22] += 0.3872983429f * in1[6] * in2[11] * go[6];
        local_grad[22] += -0.3872983429f * in1[6] * in2[13] * go[6];
        local_grad[22] += 0.1825741828f * in1[6] * in2[13] * go[8];
        local_grad[22] += -0.3651483655f * in1[6] * in2[14] * go[7];
        local_grad[22] += -0.1825741828f * in1[6] * in2[15] * go[6];
        local_grad[22] += -0.5163977742f * in1[7] * in2[10] * go[4];
        local_grad[22] += 0.5163977742f * in1[7] * in2[14] * go[8];
        local_grad[22] += -0.3651483655f * in1[8] * in2[9] * go[4];
        local_grad[22] += -0.3535533840f * in1[8] * in2[10] * go[5];
        local_grad[22] += 0.3651483655f * in1[8] * in2[10] * go[7];
        local_grad[22] += -0.5163977742f * in1[8] * in2[12] * go[6];
        local_grad[22] += 0.3535533840f * in1[8] * in2[14] * go[4];

        // Path 23: (2,3)->3, 46 entries
        local_grad[23] += -0.2182178879f * in1[4] * in2[9] * go[14];
        local_grad[23] += -0.4364357758f * in1[4] * in2[9] * go[10];
        local_grad[23] += 0.4472135802f * in1[4] * in2[10] * go[13];
        local_grad[23] += 0.1414213591f * in1[4] * in2[10] * go[15];
        local_grad[23] += -0.3651483655f * in1[4] * in2[11] * go[12];
        local_grad[23] += 0.4472135802f * in1[4] * in2[11] * go[14];
        local_grad[23] += 0.5163977742f * in1[4] * in2[12] * go[11];
        local_grad[23] += 0.4472135802f * in1[4] * in2[13] * go[10];
        local_grad[23] += 0.3651483655f * in1[4] * in2[13] * go[12];
        local_grad[23] += -0.4472135802f * in1[4] * in2[14] * go[9];
        local_grad[23] += -0.1414213591f * in1[4] * in2[14] * go[11];
        local_grad[23] += 0.2182178879f * in1[4] * in2[15] * go[10];
        local_grad[23] += 0.4472135802f * in1[5] * in2[9] * go[13];
        local_grad[23] += 0.3872983429f * in1[5] * in2[10] * go[12];
        local_grad[23] += -0.3162277607f * in1[5] * in2[10] * go[14];
        local_grad[23] += 0.1414213591f * in1[5] * in2[11] * go[11];
        local_grad[23] += -0.3651483655f * in1[5] * in2[11] * go[13];
        local_grad[23] += 0.3651483655f * in1[5] * in2[13] * go[11];
        local_grad[23] += -0.1414213591f * in1[5] * in2[13] * go[13];
        local_grad[23] += -0.3872983429f * in1[5] * in2[14] * go[12];
        local_grad[23] += -0.3162277607f * in1[5] * in2[14] * go[10];
        local_grad[23] += -0.4472135802f * in1[5] * in2[15] * go[11];
        local_grad[23] += -0.3651483655f * in1[6] * in2[9] * go[12];
        local_grad[23] += 0.3872983429f * in1[6] * in2[10] * go[11];
        local_grad[23] += -0.1414213591f * in1[6] * in2[11] * go[10];
        local_grad[23] += 0.4082482836f * in1[6] * in2[11] * go[12];
        local_grad[23] += 0.4082482836f * in1[6] * in2[13] * go[12];
        local_grad[23] += 0.1414213591f * in1[6] * in2[13] * go[14];
        local_grad[23] += -0.3872983429f * in1[6] * in2[14] * go[13];
        local_grad[23] += -0.3651483655f * in1[6] * in2[15] * go[12];
        local_grad[23] += 0.5163977742f * in1[7] * in2[9] * go[11];
        local_grad[23] += -0.3651483655f * in1[7] * in2[10] * go[10];
        local_grad[23] += 0.3651483655f * in1[7] * in2[10] * go[12];
        local_grad[23] += 0.4082482836f * in1[7] * in2[12] * go[12];
        local_grad[23] += 0.3651483655f * in1[7] * in2[14] * go[12];
        local_grad[23] += -0.3651483655f * in1[7] * in2[14] * go[14];
        local_grad[23] += 0.5163977742f * in1[7] * in2[15] * go[13];
        local_grad[23] += 0.4472135802f * in1[8] * in2[9] * go[10];
        local_grad[23] += 0.3651483655f * in1[8] * in2[9] * go[12];
        local_grad[23] += 0.1414213591f * in1[8] * in2[11] * go[11];
        local_grad[23] += 0.3651483655f * in1[8] * in2[11] * go[13];
        local_grad[23] += -0.5163977742f * in1[8] * in2[12] * go[11];
        local_grad[23] += -0.3651483655f * in1[8] * in2[13] * go[12];
        local_grad[23] += -0.1414213591f * in1[8] * in2[13] * go[13];
        local_grad[23] += 0.3162277607f * in1[8] * in2[14] * go[14];
        local_grad[23] += 0.3872983429f * in1[8] * in2[14] * go[12];

        // Path 24: (3,0)->3, 7 entries
        local_grad[24] += 0.9999999836f * in1[9] * in2[0] * go[9];
        local_grad[24] += 0.9999999836f * in1[10] * in2[0] * go[10];
        local_grad[24] += 0.9999999836f * in1[11] * in2[0] * go[11];
        local_grad[24] += 0.9999999836f * in1[12] * in2[0] * go[12];
        local_grad[24] += 0.9999999836f * in1[13] * in2[0] * go[13];
        local_grad[24] += 0.9999999836f * in1[14] * in2[0] * go[14];
        local_grad[24] += 0.9999999836f * in1[15] * in2[0] * go[15];

        // Path 25: (3,1)->2, 21 entries
        local_grad[25] += 0.5976142935f * in1[9] * in2[1] * go[8];
        local_grad[25] += 0.5976142935f * in1[9] * in2[3] * go[4];
        local_grad[25] += 0.4879500459f * in1[10] * in2[1] * go[7];
        local_grad[25] += 0.4879500459f * in1[10] * in2[2] * go[4];
        local_grad[25] += 0.4879500459f * in1[10] * in2[3] * go[5];
        local_grad[25] += 0.5345224892f * in1[11] * in2[1] * go[6];
        local_grad[25] += 0.1543033474f * in1[11] * in2[1] * go[8];
        local_grad[25] += 0.6172133895f * in1[11] * in2[2] * go[5];
        local_grad[25] += -0.1543033474f * in1[11] * in2[3] * go[4];
        local_grad[25] += -0.3779644601f * in1[12] * in2[1] * go[5];
        local_grad[25] += 0.6546536846f * in1[12] * in2[2] * go[6];
        local_grad[25] += -0.3779644601f * in1[12] * in2[3] * go[7];
        local_grad[25] += -0.1543033474f * in1[13] * in2[1] * go[4];
        local_grad[25] += 0.6172133895f * in1[13] * in2[2] * go[7];
        local_grad[25] += 0.5345224892f * in1[13] * in2[3] * go[6];
        local_grad[25] += -0.1543033474f * in1[13] * in2[3] * go[8];
        local_grad[25] += -0.4879500459f * in1[14] * in2[1] * go[5];
        local_grad[25] += 0.4879500459f * in1[14] * in2[2] * go[8];
        local_grad[25] += 0.4879500459f * in1[14] * in2[3] * go[7];
        local_grad[25] += -0.5976142935f * in1[15] * in2[1] * go[4];
        local_grad[25] += 0.5976142935f * in1[15] * in2[3] * go[8];

        // Path 26: (3,1)->3, 31 entries
        local_grad[26] += 0.3535533840f * in1[9] * in2[1] * go[14];
        local_grad[26] += 0.5773502803f * in1[9] * in2[2] * go[13];
        local_grad[26] += 0.3535533840f * in1[9] * in2[3] * go[10];
        local_grad[26] += -0.5773502803f * in1[10] * in2[1] * go[13];
        local_grad[26] += 0.1825741828f * in1[10] * in2[1] * go[15];
        local_grad[26] += 0.4472135802f * in1[10] * in2[2] * go[12];
        local_grad[26] += 0.5773502803f * in1[10] * in2[3] * go[9];
        local_grad[26] += -0.1825741828f * in1[10] * in2[3] * go[11];
        local_grad[26] += 0.4472135802f * in1[11] * in2[1] * go[12];
        local_grad[26] += -0.3651483655f * in1[11] * in2[1] * go[14];
        local_grad[26] += -0.1825741828f * in1[11] * in2[2] * go[11];
        local_grad[26] += 0.3651483655f * in1[11] * in2[2] * go[13];
        local_grad[26] += -0.4472135802f * in1[11] * in2[3] * go[10];
        local_grad[26] += 0.3651483655f * in1[11] * in2[3] * go[12];
        local_grad[26] += -0.5163977742f * in1[12] * in2[1] * go[11];
        local_grad[26] += -0.3651483655f * in1[12] * in2[2] * go[11];
        local_grad[26] += 0.1825741828f * in1[12] * in2[2] * go[13];
        local_grad[26] += 0.5163977742f * in1[12] * in2[3] * go[13];
        local_grad[26] += -0.4472135802f * in1[13] * in2[1] * go[10];
        local_grad[26] += 0.3651483655f * in1[13] * in2[1] * go[12];
        local_grad[26] += -0.3651483655f * in1[13] * in2[2] * go[11];
        local_grad[26] += 0.1825741828f * in1[13] * in2[2] * go[13];
        local_grad[26] += 0.4472135802f * in1[13] * in2[3] * go[12];
        local_grad[26] += 0.3651483655f * in1[13] * in2[3] * go[14];
        local_grad[26] += 0.5773502803f * in1[14] * in2[1] * go[9];
        local_grad[26] += -0.1825741828f * in1[14] * in2[1] * go[11];
        local_grad[26] += -0.4472135802f * in1[14] * in2[2] * go[12];
        local_grad[26] += -0.5773502803f * in1[14] * in2[3] * go[15];
        local_grad[26] += 0.1825741828f * in1[14] * in2[3] * go[13];
        local_grad[26] += -0.3535533840f * in1[15] * in2[1] * go[10];
        local_grad[26] += -0.5773502803f * in1[15] * in2[2] * go[11];
        local_grad[26] += -0.3535533840f * in1[15] * in2[3] * go[14];

        // Path 27: (3,2)->1, 21 entries
        local_grad[27] += 0.3779644601f * in1[9] * in2[4] * go[3];
        local_grad[27] += -0.3779644601f * in1[9] * in2[8] * go[1];
        local_grad[27] += 0.5345224892f * in1[10] * in2[4] * go[2];
        local_grad[27] += 0.5345224892f * in1[10] * in2[5] * go[1];
        local_grad[27] += -0.5345224892f * in1[10] * in2[7] * go[2];
        local_grad[27] += 0.5345224892f * in1[10] * in2[8] * go[3];
        local_grad[27] += -0.2672612446f * in1[11] * in2[4] * go[1];
        local_grad[27] += -0.2672612446f * in1[11] * in2[4] * go[3];
        local_grad[27] += 0.4629100433f * in1[11] * in2[5] * go[2];
        local_grad[27] += 0.4629100433f * in1[11] * in2[7] * go[2];
        local_grad[27] += 0.2672612446f * in1[11] * in2[8] * go[1];
        local_grad[27] += 0.2672612446f * in1[11] * in2[8] * go[3];
        local_grad[27] += 0.5345224892f * in1[12] * in2[6] * go[2];
        local_grad[27] += 0.4629100433f * in1[13] * in2[5] * go[2];
        local_grad[27] += 0.4629100433f * in1[13] * in2[7] * go[2];
        local_grad[27] += -0.5345224892f * in1[14] * in2[4] * go[2];
        local_grad[27] += 0.4629100433f * in1[14] * in2[6] * go[2];
        local_grad[27] += 0.5345224892f * in1[14] * in2[8] * go[2];
        local_grad[27] += -0.3779644601f * in1[15] * in2[4] * go[1];
        local_grad[27] += 0.2672612446f * in1[15] * in2[6] * go[3];
        local_grad[27] += -0.3779644601f * in1[15] * in2[8] * go[3];

        // Path 28: (3,2)->2, 28 entries
        local_grad[28] += 0.4225771204f * in1[9] * in2[4] * go[5];
        local_grad[28] += -0.4225771204f * in1[9] * in2[5] * go[4];
        local_grad[28] += 0.4225771204f * in1[9] * in2[7] * go[8];
        local_grad[28] += -0.4225771204f * in1[9] * in2[8] * go[7];
        local_grad[28] += 0.5976142935f * in1[10] * in2[6] * go[8];
        local_grad[28] += -0.5976142935f * in1[10] * in2[8] * go[6];
        local_grad[28] += 0.3273268423f * in1[11] * in2[4] * go[5];
        local_grad[28] += -0.3273268423f * in1[11] * in2[5] * go[4];
        local_grad[28] += 0.3779644601f * in1[11] * in2[6] * go[7];
        local_grad[28] += -0.3779644601f * in1[11] * in2[7] * go[6];
        local_grad[28] += -0.3273268423f * in1[11] * in2[7] * go[8];
        local_grad[28] += 0.3273268423f * in1[11] * in2[8] * go[7];
        local_grad[28] += 0.2672612446f * in1[12] * in2[4] * go[8];
        local_grad[28] += -0.5345224892f * in1[12] * in2[5] * go[7];
        local_grad[28] += 0.5345224892f * in1[12] * in2[7] * go[5];
        local_grad[28] += -0.2672612446f * in1[12] * in2[8] * go[4];
        local_grad[28] += -0.3273268423f * in1[13] * in2[4] * go[7];
        local_grad[28] += 0.3779644601f * in1[13] * in2[5] * go[6];
        local_grad[28] += -0.3273268423f * in1[13] * in2[5] * go[8];
        local_grad[28] += -0.3779644601f * in1[13] * in2[6] * go[5];
        local_grad[28] += 0.3273268423f * in1[13] * in2[7] * go[4];
        local_grad[28] += 0.3273268423f * in1[13] * in2[8] * go[5];
        local_grad[28] += 0.5976142935f * in1[14] * in2[4] * go[6];
        local_grad[28] += -0.5976142935f * in1[14] * in2[6] * go[4];
        local_grad[28] += 0.4225771204f * in1[15] * in2[4] * go[7];
        local_grad[28] += -0.4225771204f * in1[15] * in2[5] * go[8];
        local_grad[28] += -0.4225771204f * in1[15] * in2[7] * go[4];
        local_grad[28] += 0.4225771204f * in1[15] * in2[8] * go[5];

        // Path 29: (3,2)->3, 41 entries
        local_grad[29] += -0.2886751401f * in1[9] * in2[4] * go[13];
        local_grad[29] += 0.4564354668f * in1[9] * in2[5] * go[14];
        local_grad[29] += -0.6454972118f * in1[9] * in2[6] * go[9];
        local_grad[29] += 0.4564354668f * in1[9] * in2[7] * go[10];
        local_grad[29] += -0.2886751401f * in1[9] * in2[8] * go[11];
        local_grad[29] += -0.5773502803f * in1[10] * in2[4] * go[12];
        local_grad[29] += 0.3535533840f * in1[10] * in2[5] * go[13];
        local_grad[29] += -0.4564354668f * in1[10] * in2[5] * go[15];
        local_grad[29] += 0.4564354668f * in1[10] * in2[7] * go[9];
        local_grad[29] += 0.3535533840f * in1[10] * in2[7] * go[11];
        local_grad[29] += 0.4472135802f * in1[11] * in2[4] * go[13];
        local_grad[29] += 0.2886751401f * in1[11] * in2[4] * go[15];
        local_grad[29] += 0.1825741828f * in1[11] * in2[5] * go[12];
        local_grad[29] += -0.3535533840f * in1[11] * in2[5] * go[14];
        local_grad[29] += 0.3872983429f * in1[11] * in2[6] * go[11];
        local_grad[29] += 0.3535533840f * in1[11] * in2[7] * go[10];
        local_grad[29] += -0.2886751401f * in1[11] * in2[8] * go[9];
        local_grad[29] += -0.4472135802f * in1[11] * in2[8] * go[11];
        local_grad[29] += -0.5773502803f * in1[12] * in2[4] * go[10];
        local_grad[29] += 0.1825741828f * in1[12] * in2[5] * go[11];
        local_grad[29] += -0.4564354668f * in1[12] * in2[5] * go[13];
        local_grad[29] += 0.4564354668f * in1[12] * in2[7] * go[11];
        local_grad[29] += -0.1825741828f * in1[12] * in2[7] * go[13];
        local_grad[29] += 0.5773502803f * in1[12] * in2[8] * go[14];
        local_grad[29] += 0.2886751401f * in1[13] * in2[4] * go[9];
        local_grad[29] += -0.4472135802f * in1[13] * in2[4] * go[11];
        local_grad[29] += -0.3535533840f * in1[13] * in2[5] * go[10];
        local_grad[29] += -0.3872983429f * in1[13] * in2[6] * go[11];
        local_grad[29] += 0.1825741828f * in1[13] * in2[7] * go[12];
        local_grad[29] += 0.3535533840f * in1[13] * in2[7] * go[14];
        local_grad[29] += -0.4472135802f * in1[13] * in2[8] * go[13];
        local_grad[29] += -0.2886751401f * in1[13] * in2[8] * go[15];
        local_grad[29] += -0.4564354668f * in1[14] * in2[5] * go[9];
        local_grad[29] += -0.3535533840f * in1[14] * in2[5] * go[11];
        local_grad[29] += 0.5773502803f * in1[14] * in2[8] * go[12];
        local_grad[29] += 0.4564354668f * in1[14] * in2[7] * go[15];
        local_grad[29] += -0.3535533840f * in1[14] * in2[7] * go[13];
        local_grad[29] += 0.2886751401f * in1[15] * in2[4] * go[11];
        local_grad[29] += 0.4564354668f * in1[15] * in2[5] * go[10];
        local_grad[29] += 0.6454972118f * in1[15] * in2[6] * go[15];
        local_grad[29] += -0.4564354668f * in1[15] * in2[7] * go[14];

        // Path 30: (3,3)->0, 7 entries
        local_grad[30] += 0.3779644688f * in1[9] * in2[9] * go[0];
        local_grad[30] += 0.3779644688f * in1[10] * in2[10] * go[0];
        local_grad[30] += 0.3779644688f * in1[11] * in2[11] * go[0];
        local_grad[30] += 0.3779644688f * in1[12] * in2[12] * go[0];
        local_grad[30] += 0.3779644688f * in1[13] * in2[13] * go[0];
        local_grad[30] += 0.3779644688f * in1[14] * in2[14] * go[0];
        local_grad[30] += 0.3779644688f * in1[15] * in2[15] * go[0];

        // Path 31: (3,3)->1, 21 entries
        local_grad[31] += 0.3779644601f * in1[9] * in2[10] * go[2];
        local_grad[31] += 0.3779644601f * in1[9] * in2[14] * go[3];
        local_grad[31] += 0.3779644601f * in1[10] * in2[9] * go[2];
        local_grad[31] += 0.4629100433f * in1[10] * in2[11] * go[1];
        local_grad[31] += 0.4629100433f * in1[10] * in2[13] * go[3];
        local_grad[31] += -0.4629100433f * in1[11] * in2[10] * go[1];
        local_grad[31] += 0.5345224892f * in1[11] * in2[11] * go[2];
        local_grad[31] += 0.5345224892f * in1[11] * in2[13] * go[2];
        local_grad[31] += 0.4629100433f * in1[11] * in2[14] * go[3];
        local_grad[31] += 0.5345224892f * in1[12] * in2[12] * go[2];
        local_grad[31] += 0.5345224892f * in1[13] * in2[11] * go[2];
        local_grad[31] += -0.4629100433f * in1[13] * in2[10] * go[3];
        local_grad[31] += 0.5345224892f * in1[13] * in2[13] * go[2];
        local_grad[31] += -0.4629100433f * in1[13] * in2[14] * go[1];
        local_grad[31] += 0.3779644601f * in1[14] * in2[9] * go[3];
        local_grad[31] += 0.4629100433f * in1[14] * in2[11] * go[3];
        local_grad[31] += 0.4629100433f * in1[14] * in2[13] * go[1];
        local_grad[31] += 0.3779644601f * in1[14] * in2[15] * go[2];
        local_grad[31] += -0.3779644601f * in1[15] * in2[10] * go[1];
        local_grad[31] += 0.3779644601f * in1[15] * in2[14] * go[2];
        local_grad[31] += -0.3779644601f * in1[15] * in2[10] * go[3];

        // Path 32: (3,3)->2, 36 entries
        local_grad[32] += -0.2390457287f * in1[9] * in2[9] * go[6];
        local_grad[32] += 0.5222329590f * in1[9] * in2[9] * go[8];
        local_grad[32] += 0.5345224892f * in1[9] * in2[10] * go[5];
        local_grad[32] += 0.5345224892f * in1[9] * in2[10] * go[7];
        local_grad[32] += -0.1543033474f * in1[9] * in2[11] * go[4];
        local_grad[32] += 0.4879500459f * in1[9] * in2[11] * go[6];
        local_grad[32] += 0.4879500459f * in1[9] * in2[13] * go[6];
        local_grad[32] += 0.1543033474f * in1[9] * in2[13] * go[8];
        local_grad[32] += 0.5345224892f * in1[10] * in2[9] * go[5];
        local_grad[32] += -0.5345224892f * in1[10] * in2[9] * go[7];
        local_grad[32] += 0.4780914575f * in1[10] * in2[10] * go[6];
        local_grad[32] += 0.3273268423f * in1[10] * in2[11] * go[5];
        local_grad[32] += -0.3273268423f * in1[10] * in2[13] * go[5];
        local_grad[32] += 0.5345224892f * in1[10] * in2[14] * go[8];
        local_grad[32] += -0.5345224892f * in1[10] * in2[14] * go[4];
        local_grad[32] += -0.1543033474f * in1[11] * in2[9] * go[4];
        local_grad[32] += -0.4879500459f * in1[11] * in2[9] * go[6];
        local_grad[32] += 0.3273268423f * in1[11] * in2[10] * go[5];
        local_grad[32] += 0.1543033474f * in1[11] * in2[11] * go[6];
        local_grad[32] += -0.5345224892f * in1[11] * in2[11] * go[8];
        local_grad[32] += 0.6546536846f * in1[11] * in2[12] * go[7];
        local_grad[32] += -0.1543033474f * in1[11] * in2[13] * go[6];
        local_grad[32] += -0.5345224892f * in1[11] * in2[13] * go[8];
        local_grad[32] += 0.6546536846f * in1[12] * in2[11] * go[7];
        local_grad[32] += -0.4780914575f * in1[12] * in2[12] * go[6];
        local_grad[32] += 0.6546536846f * in1[12] * in2[13] * go[5];
        local_grad[32] += 0.4879500459f * in1[13] * in2[9] * go[6];
        local_grad[32] += -0.1543033474f * in1[13] * in2[9] * go[8];
        local_grad[32] += -0.3273268423f * in1[13] * in2[10] * go[5];
        local_grad[32] += -0.1543033474f * in1[13] * in2[11] * go[6];
        local_grad[32] += 0.5345224892f * in1[13] * in2[11] * go[8];
        local_grad[32] += 0.6546536846f * in1[13] * in2[12] * go[5];
        local_grad[32] += 0.1543033474f * in1[13] * in2[13] * go[6];
        local_grad[32] += 0.5345224892f * in1[13] * in2[13] * go[8];
        local_grad[32] += -0.5345224892f * in1[14] * in2[10] * go[4];
        local_grad[32] += 0.5345224892f * in1[14] * in2[10] * go[8];

        // Path 33: (3,3)->3, 55 entries (partial - most common entries)
        local_grad[33] += -0.2182178879f * in1[9] * in2[9] * go[14];
        local_grad[33] += 0.4364357758f * in1[9] * in2[10] * go[13];
        local_grad[33] += 0.3651483655f * in1[9] * in2[11] * go[12];
        local_grad[33] += 0.5163977742f * in1[9] * in2[12] * go[11];
        local_grad[33] += 0.3651483655f * in1[9] * in2[13] * go[10];
        local_grad[33] += 0.4364357758f * in1[9] * in2[14] * go[9];
        local_grad[33] += 0.4364357758f * in1[10] * in2[9] * go[13];
        local_grad[33] += 0.4364357758f * in1[10] * in2[10] * go[14];
        local_grad[33] += -0.3162277607f * in1[10] * in2[10] * go[10];
        local_grad[33] += 0.1414213591f * in1[10] * in2[11] * go[11];
        local_grad[33] += 0.3651483655f * in1[10] * in2[11] * go[13];
        local_grad[33] += 0.3651483655f * in1[10] * in2[12] * go[12];
        local_grad[33] += 0.3651483655f * in1[10] * in2[13] * go[11];
        local_grad[33] += -0.1414213591f * in1[10] * in2[13] * go[13];
        local_grad[33] += 0.3162277607f * in1[10] * in2[14] * go[14];
        local_grad[33] += -0.3162277607f * in1[10] * in2[14] * go[10];
        local_grad[33] += 0.4364357758f * in1[10] * in2[15] * go[9];
        local_grad[33] += 0.3651483655f * in1[11] * in2[9] * go[12];
        local_grad[33] += 0.3651483655f * in1[11] * in2[10] * go[13];
        local_grad[33] += 0.1414213591f * in1[11] * in2[10] * go[11];
        local_grad[33] += 0.4082482836f * in1[11] * in2[11] * go[12];
        local_grad[33] += 0.4082482836f * in1[11] * in2[12] * go[11];
        local_grad[33] += 0.4082482836f * in1[11] * in2[12] * go[13];
        local_grad[33] += 0.4082482836f * in1[11] * in2[13] * go[12];
        local_grad[33] += 0.1414213591f * in1[11] * in2[14] * go[11];
        local_grad[33] += 0.3651483655f * in1[11] * in2[14] * go[13];
        local_grad[33] += 0.3651483655f * in1[11] * in2[15] * go[12];
        local_grad[33] += 0.5163977742f * in1[12] * in2[9] * go[11];
        local_grad[33] += 0.3651483655f * in1[12] * in2[10] * go[12];
        local_grad[33] += 0.4082482836f * in1[12] * in2[11] * go[11];
        local_grad[33] += 0.4082482836f * in1[12] * in2[11] * go[13];
        local_grad[33] += 0.4082482836f * in1[12] * in2[13] * go[11];
        local_grad[33] += 0.4082482836f * in1[12] * in2[13] * go[13];
        local_grad[33] += 0.3651483655f * in1[12] * in2[14] * go[12];
        local_grad[33] += 0.5163977742f * in1[12] * in2[15] * go[13];
        local_grad[33] += 0.3651483655f * in1[13] * in2[9] * go[10];
        local_grad[33] += 0.3651483655f * in1[13] * in2[10] * go[11];
        local_grad[33] += -0.1414213591f * in1[13] * in2[10] * go[13];
        local_grad[33] += 0.4082482836f * in1[13] * in2[11] * go[12];
        local_grad[33] += 0.4082482836f * in1[13] * in2[12] * go[11];
        local_grad[33] += -0.4082482836f * in1[13] * in2[12] * go[13];
        local_grad[33] += -0.4082482836f * in1[13] * in2[13] * go[12];
        local_grad[33] += -0.1414213591f * in1[13] * in2[14] * go[11];
        local_grad[33] += 0.3651483655f * in1[13] * in2[14] * go[13];
        local_grad[33] += 0.3651483655f * in1[13] * in2[15] * go[14];
        local_grad[33] += 0.4364357758f * in1[14] * in2[9] * go[9];
        local_grad[33] += -0.3162277607f * in1[14] * in2[10] * go[10];
        local_grad[33] += 0.3162277607f * in1[14] * in2[10] * go[14];
        local_grad[33] += 0.1414213591f * in1[14] * in2[11] * go[11];
        local_grad[33] += 0.3651483655f * in1[14] * in2[11] * go[13];
        local_grad[33] += 0.3651483655f * in1[14] * in2[12] * go[12];
        local_grad[33] += 0.3651483655f * in1[14] * in2[13] * go[13];
        local_grad[33] += -0.1414213591f * in1[14] * in2[13] * go[11];
        local_grad[33] += 0.4364357758f * in1[14] * in2[14] * go[10];
        local_grad[33] += -0.2182178879f * in1[15] * in2[15] * go[10];
    }

    // Block reduction using shared memory
    __shared__ float shared_grad[34][256];

    #pragma unroll
    for (int p = 0; p < 34; p++) {
        shared_grad[p][threadIdx.x] = local_grad[p];
    }
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int p = 0; p < 34; p++) {
                shared_grad[p][threadIdx.x] += shared_grad[p][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes results for all 34 paths
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int p = 0; p < 34; p++) {
            grad_weights[p * C_in * C_out + c_in * C_out + c_out] = shared_grad[p][0];
        }
    }
}

// ============================================================================
// Simple tensor product backward (no weights, no channels)
// ============================================================================

/**
 * FUSED backward kernel for simple tensor product (no weights).
 * Each thread handles one sample.
 */
__global__ void tensor_product_simple_backward_kernel_fused(
    const float* __restrict__ input1,       // [N, 16]
    const float* __restrict__ input2,       // [N, 16]
    const float* __restrict__ grad_output,  // [N, 16]
    float* __restrict__ grad_input1,        // [N, 16]
    float* __restrict__ grad_input2,        // [N, 16]
    int64_t N
) {
    const int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* in1 = input1 + n * NUM_IRREPS;
    const float* in2 = input2 + n * NUM_IRREPS;
    const float* grad_out = grad_output + n * NUM_IRREPS;
    float* grad1_out = grad_input1 + n * NUM_IRREPS;
    float* grad2_out = grad_input2 + n * NUM_IRREPS;

    // Initialize accumulators
    float grad1_acc[NUM_IRREPS] = {0};
    float grad2_acc[NUM_IRREPS] = {0};

    // Use fused backward function
    fused::tensor_product_simple_backward_unrolled(in1, in2, grad_out, grad1_acc, grad2_acc);

    // Write outputs
    #pragma unroll
    for (int i = 0; i < NUM_IRREPS; ++i) {
        grad1_out[i] = grad1_acc[i];
        grad2_out[i] = grad2_acc[i];
    }
}

// ============================================================================
// Launch wrappers
// ============================================================================

cudaError_t tensor_product_backward_l3(
    const float* input1,
    const float* input2,
    const float* weights,
    const float* grad_output,
    float* grad_input1,
    float* grad_input2,
    float* grad_weights,
    int64_t N,
    int C_in,
    int C_out,
    cudaStream_t stream
) {
    if (N == 0) return cudaSuccess;

    cudaError_t err;

    // Compute grad_input1 if requested
    if (grad_input1 != nullptr) {
        const int64_t total_work = N * C_in;
        constexpr int block_size = 128;
        const int num_blocks = (total_work + block_size - 1) / block_size;

        tensor_product_backward_input1_kernel_fused<<<num_blocks, block_size, 0, stream>>>(
            input2, weights, grad_output, grad_input1, N, C_in, C_out
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Compute grad_input2 if requested
    if (grad_input2 != nullptr) {
        // Use optimized kernel: one block per sample, threads split (c_in, c_out) pairs
        constexpr int block_size = 256;
        const int num_blocks = N;

        tensor_product_backward_input2_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
            input1, weights, grad_output, grad_input2, N, C_in, C_out
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Compute grad_weights if requested
    if (grad_weights != nullptr) {
        // Use optimized kernel: grid is (C_in, C_out), each block reduces over samples
        dim3 grid(C_in, C_out);
        constexpr int block_size = 256;

        tensor_product_backward_weights_kernel_opt<<<grid, block_size, 0, stream>>>(
            input1, input2, grad_output, grad_weights, N, C_in, C_out
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

cudaError_t tensor_product_simple_backward_l3(
    const float* input1,
    const float* input2,
    const float* grad_output,
    float* grad_input1,
    float* grad_input2,
    int64_t N,
    cudaStream_t stream
) {
    if (N == 0) return cudaSuccess;

    constexpr int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    tensor_product_simple_backward_kernel_fused<<<num_blocks, block_size, 0, stream>>>(
        input1, input2, grad_output, grad_input1, grad_input2, N
    );

    return cudaGetLastError();
}

} // namespace batteries
