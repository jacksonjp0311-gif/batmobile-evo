/**
 * Tensor Product CUDA Kernel Header
 *
 * Computes equivariant tensor products using Clebsch-Gordan coefficients:
 *   (u ⊗ v)_l^m = Σ_{l1,m1,l2,m2} C_{l1,m1,l2,m2}^{l,m} × u_{l1}^{m1} × v_{l2}^{m2}
 *
 * Supports "fully connected" tensor products where all valid (l1, l2) -> l_out
 * paths are computed, with independent learnable weights per path.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace batteries {

/**
 * Fully connected tensor product: all (l1, l2) -> l_out paths with channels.
 *
 * For each sample n and output channel c_out:
 *   output[n, c_out, :] = Σ_{path} weight[path, c_in, c_out] *
 *                         Σ_{cg} cg_coeff * input1[n, c_in, m1] * input2[n, m2]
 *
 * Where input2 is typically edge features (spherical harmonics), shared across channels.
 *
 * @param input1       [N, C_in, num_irreps1] - node features with channels
 * @param input2       [N, num_irreps2] - edge features (no channels, from spherical harmonics)
 * @param weights      [num_paths, C_in, C_out] - learnable path weights
 * @param output       [N, C_out, num_irreps_out] - output features
 * @param N            Number of samples (edges)
 * @param C_in         Number of input channels
 * @param C_out        Number of output channels
 * @param stream       CUDA stream
 */
cudaError_t tensor_product_l3(
    const float* input1,
    const float* input2,
    const float* weights,
    float* output,
    int64_t N,
    int C_in,
    int C_out,
    cudaStream_t stream = nullptr
);

/**
 * Tensor product without weights - pure CG contraction.
 * Useful for testing and simple cases.
 *
 * @param input1       [N, num_irreps1] - first input
 * @param input2       [N, num_irreps2] - second input
 * @param output       [N, num_irreps_out] - output
 * @param N            Number of samples
 * @param stream       CUDA stream
 */
cudaError_t tensor_product_simple_l3(
    const float* input1,
    const float* input2,
    float* output,
    int64_t N,
    cudaStream_t stream = nullptr
);

/**
 * Get the number of CG paths for L_max=3.
 * Each path is a valid (l1, l2) -> l_out combination.
 */
int get_num_paths_l3();

/**
 * Get path info for L_max=3.
 * Returns array of [l1, l2, l_out] for each path.
 */
void get_path_info_l3(int* path_info);

// ============================================================================
// Backward pass kernels
// ============================================================================

/**
 * Backward for fully connected tensor product.
 *
 * Computes gradients for input1, input2, and weights given grad_output.
 *
 * @param input1       [N, C_in, 16] - saved from forward
 * @param input2       [N, 16] - saved from forward
 * @param weights      [num_paths, C_in, C_out] - saved from forward
 * @param grad_output  [N, C_out, 16] - gradient from upstream
 * @param grad_input1  [N, C_in, 16] - output gradient (nullptr to skip)
 * @param grad_input2  [N, 16] - output gradient (nullptr to skip)
 * @param grad_weights [num_paths, C_in, C_out] - output gradient (nullptr to skip)
 * @param N            Number of samples
 * @param C_in         Number of input channels
 * @param C_out        Number of output channels
 * @param stream       CUDA stream
 */
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
    cudaStream_t stream = nullptr
);

/**
 * Backward for simple tensor product (no weights).
 *
 * @param input1       [N, 16] - saved from forward
 * @param input2       [N, 16] - saved from forward
 * @param grad_output  [N, 16] - gradient from upstream
 * @param grad_input1  [N, 16] - output gradient
 * @param grad_input2  [N, 16] - output gradient
 * @param N            Number of samples
 * @param stream       CUDA stream
 */
cudaError_t tensor_product_simple_backward_l3(
    const float* input1,
    const float* input2,
    const float* grad_output,
    float* grad_input1,
    float* grad_input2,
    int64_t N,
    cudaStream_t stream = nullptr
);

} // namespace batteries
