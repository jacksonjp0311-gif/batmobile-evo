/**
 * Fused Spherical Harmonics + Tensor Product Kernel
 *
 * Eliminates intermediate Y_lm tensor by computing SH in registers
 * and immediately using it for tensor product.
 *
 * Pipeline:
 *   edge_vectors -> [SH in registers] -> tensor_product -> messages
 *
 * Memory savings: eliminates N_edges * 16 * 4 bytes (Y_lm tensor)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace batteries {

// Simple version: no weights, computes messages[edge, c, m] for all edges/channels
// One thread per (edge, channel) pair
cudaError_t fused_sh_tp_simple(
    const float* edge_vectors,     // [N_edges, 3] unit vectors
    const float* node_features,    // [N_atoms, C, 16] source features
    const int64_t* source_idx,     // [N_edges] source atom for each edge
    float* messages,               // [N_edges, C, 16] output messages
    int64_t N_edges,
    int C,
    cudaStream_t stream = nullptr
);

// Weighted version: applies learnable weights per path
// weights[num_paths, C_in, C_out] maps input channels to output channels
cudaError_t fused_sh_tp_weighted(
    const float* edge_vectors,     // [N_edges, 3] unit vectors
    const float* node_features,    // [N_atoms, C_in, 16] source features
    const int64_t* source_idx,     // [N_edges] source atom for each edge
    const float* weights,          // [num_paths, C_in, C_out] learnable weights
    float* messages,               // [N_edges, C_out, 16] output messages
    int64_t N_edges,
    int C_in,
    int C_out,
    cudaStream_t stream = nullptr
);

} // namespace batteries
