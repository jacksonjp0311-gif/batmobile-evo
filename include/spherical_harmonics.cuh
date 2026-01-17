#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace batteries {

// Template parameter L_MAX for compile-time optimization
// L_MAX=3 gives 16 components, L_MAX=6 gives 49 components
template <int L_MAX>
struct SphericalHarmonicsConfig {
    static constexpr int num_components = (L_MAX + 1) * (L_MAX + 1);
};

// Launch spherical harmonics kernel
// Input: edge_vectors [N, 3] - unit vectors (x, y, z)
// Output: Y_lm [N, (L_MAX+1)^2] - real spherical harmonics
// Returns cudaError_t for error checking
template <int L_MAX>
cudaError_t spherical_harmonics(
    const float* edge_vectors,  // [N, 3]
    float* Y_lm,                // [N, (L_MAX+1)^2]
    int64_t N,
    cudaStream_t stream = nullptr
);

// Explicit instantiations for common L_MAX values
extern template cudaError_t spherical_harmonics<3>(
    const float*, float*, int64_t, cudaStream_t);
extern template cudaError_t spherical_harmonics<6>(
    const float*, float*, int64_t, cudaStream_t);

// Runtime dispatch version (slightly slower due to switch)
cudaError_t spherical_harmonics_dispatch(
    const float* edge_vectors,
    float* Y_lm,
    int64_t N,
    int L_max,
    cudaStream_t stream = nullptr
);

// ============================================================================
// Backward pass kernels
// ============================================================================

// Backward kernel for spherical harmonics
// Computes grad_xyz = d(Loss)/d(x,y,z) given grad_Y_lm = d(Loss)/d(Y_lm)
// Input: edge_vectors [N, 3] - unit vectors used in forward pass
// Input: grad_Y_lm [N, (L_MAX+1)^2] - gradient w.r.t. output
// Output: grad_xyz [N, 3] - gradient w.r.t. input edge vectors
template <int L_MAX>
cudaError_t spherical_harmonics_backward(
    const float* edge_vectors,
    const float* grad_Y_lm,
    float* grad_xyz,
    int64_t N,
    cudaStream_t stream = nullptr
);

// Optimized backward for L_max=3 (most common case)
cudaError_t spherical_harmonics_backward_l3(
    const float* edge_vectors,
    const float* grad_Y_lm,
    float* grad_xyz,
    int64_t N,
    cudaStream_t stream = nullptr
);

// Runtime dispatch for backward
cudaError_t spherical_harmonics_backward_dispatch(
    const float* edge_vectors,
    const float* grad_Y_lm,
    float* grad_xyz,
    int64_t N,
    int L_max,
    cudaStream_t stream = nullptr
);

} // namespace batteries
