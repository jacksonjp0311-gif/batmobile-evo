/**
 * Fused Message Passing CUDA Kernel
 *
 * Fuses three operations into a single kernel:
 * 1. Spherical harmonics computation (inline, no global memory)
 * 2. Tensor product (inline, no global memory)
 * 3. Scatter-add reduction (shared memory accumulation)
 *
 * Memory layout uses CSR format for edge indices:
 * - row_ptr[N_atoms+1]: starting edge index for each atom
 * - col_idx[N_edges]: source atom index for each edge
 *
 * Each block processes one target atom, cooperatively processing all edges.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace batteries {

// Convert COO edge_index to CSR format
// edge_index: [2, N_edges] where [0,:] = source, [1,:] = target
// Output: row_ptr[N_atoms+1], col_idx[N_edges]
cudaError_t edge_index_to_csr(
    const int64_t* edge_index,  // [2, N_edges] COO format
    int64_t* row_ptr,           // [N_atoms+1] CSR row pointers
    int64_t* col_idx,           // [N_edges] source indices
    int64_t N_atoms,
    int64_t N_edges,
    cudaStream_t stream = nullptr
);

// Fused message passing: SH -> TP -> scatter_add
// L_MAX=3 hardcoded for MACE (16 irreps = (3+1)^2)
cudaError_t fused_message_passing(
    const float* node_features,    // [N_atoms, C_in, 16] source node features
    const float* edge_vectors,     // [N_edges, 3] unit edge vectors
    const float* weights,          // [num_paths, C_in, C_out] tensor product weights
    const int64_t* row_ptr,        // [N_atoms+1] CSR row pointers
    const int64_t* col_idx,        // [N_edges] source atom indices
    float* node_output,            // [N_atoms, C_out, 16] output
    int64_t N_atoms,
    int64_t N_edges,
    int C_in,
    int C_out,
    cudaStream_t stream = nullptr
);

// Variant without weights (simple tensor product)
cudaError_t fused_message_passing_simple(
    const float* node_features,    // [N_atoms, C_in, 16]
    const float* edge_vectors,     // [N_edges, 3]
    const int64_t* edge_index,     // [2, N_edges] - source/target pairs
    const int64_t* row_ptr,        // [N_atoms+1]
    const int64_t* col_idx,        // [N_edges] - original edge indices in CSR order
    float* node_output,            // [N_atoms, C_in, 16]
    int64_t N_atoms,
    int64_t N_edges,
    int C,
    cudaStream_t stream = nullptr
);

} // namespace batteries
