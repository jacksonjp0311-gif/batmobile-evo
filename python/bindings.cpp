/**
 * Python bindings for batteries CUDA kernels
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>

#include "spherical_harmonics.cuh"
#include "tensor_product.cuh"
#include "fused_message_passing.cuh"
#include "fused_sh_tp.cuh"

namespace py = pybind11;

// Convert PyTorch tensor to raw pointer with validation
template <typename T>
const T* get_ptr(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t.data_ptr<T>();
}

template <typename T>
T* get_ptr_mut(torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t.data_ptr<T>();
}

/**
 * Compute real spherical harmonics Y_lm
 *
 * Args:
 *   edge_vectors: Float tensor [N, 3] of unit vectors
 *   L_max: Maximum angular momentum (0-6)
 *
 * Returns:
 *   Y_lm: Float tensor [N, (L_max+1)^2]
 */
torch::Tensor spherical_harmonics(
    const torch::Tensor& edge_vectors,
    int64_t L_max
) {
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N, 3]");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(L_max >= 0 && L_max <= 6, "L_max must be in [0, 6]");

    const int64_t N = edge_vectors.size(0);
    const int64_t num_components = (L_max + 1) * (L_max + 1);

    // Allocate output
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(edge_vectors.device());
    torch::Tensor Y_lm = torch::empty({N, num_components}, options);

    if (N == 0) return Y_lm;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    cudaError_t err = batteries::spherical_harmonics_dispatch(
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr_mut<float>(Y_lm, "Y_lm"),
        N,
        static_cast<int>(L_max),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return Y_lm;
}

/**
 * Compute spherical harmonics in-place (for pre-allocated output)
 */
void spherical_harmonics_inplace(
    const torch::Tensor& edge_vectors,
    torch::Tensor& Y_lm,
    int64_t L_max
) {
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N, 3]");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(L_max >= 0 && L_max <= 6, "L_max must be in [0, 6]");

    const int64_t N = edge_vectors.size(0);
    const int64_t num_components = (L_max + 1) * (L_max + 1);

    TORCH_CHECK(Y_lm.dim() == 2, "Y_lm must be 2D");
    TORCH_CHECK(Y_lm.size(0) == N, "Y_lm must have N rows");
    TORCH_CHECK(Y_lm.size(1) == num_components, "Y_lm must have (L_max+1)^2 columns");
    TORCH_CHECK(Y_lm.dtype() == torch::kFloat32, "Y_lm must be float32");

    if (N == 0) return;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::spherical_harmonics_dispatch(
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr_mut<float>(Y_lm, "Y_lm"),
        N,
        static_cast<int>(L_max),
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
}

/**
 * Simple tensor product (no channels, no weights)
 * For testing CG coefficient correctness.
 */
torch::Tensor tensor_product_simple(
    const torch::Tensor& input1,
    const torch::Tensor& input2
) {
    TORCH_CHECK(input1.dim() == 2, "input1 must be 2D [N, 16]");
    TORCH_CHECK(input2.dim() == 2, "input2 must be 2D [N, 16]");
    TORCH_CHECK(input1.size(1) == 16, "input1 must have 16 components (L_max=3)");
    TORCH_CHECK(input2.size(1) == 16, "input2 must have 16 components (L_max=3)");
    TORCH_CHECK(input1.size(0) == input2.size(0), "input1 and input2 must have same N");
    TORCH_CHECK(input1.dtype() == torch::kFloat32, "input1 must be float32");
    TORCH_CHECK(input2.dtype() == torch::kFloat32, "input2 must be float32");

    const int64_t N = input1.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input1.device());
    torch::Tensor output = torch::empty({N, 16}, options);

    if (N == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::tensor_product_simple_l3(
        get_ptr<float>(input1, "input1"),
        get_ptr<float>(input2, "input2"),
        get_ptr_mut<float>(output, "output"),
        N,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

/**
 * Full tensor product with channels and learnable weights.
 *
 * Args:
 *   input1: [N, C_in, 16] - node features with channels
 *   input2: [N, 16] - edge features (spherical harmonics)
 *   weights: [num_paths, C_in, C_out] - learnable weights
 *
 * Returns:
 *   output: [N, C_out, 16] - output features
 */
torch::Tensor tensor_product(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& weights
) {
    TORCH_CHECK(input1.dim() == 3, "input1 must be 3D [N, C_in, 16]");
    TORCH_CHECK(input2.dim() == 2, "input2 must be 2D [N, 16]");
    TORCH_CHECK(weights.dim() == 3, "weights must be 3D [num_paths, C_in, C_out]");
    TORCH_CHECK(input1.size(2) == 16, "input1 must have 16 components (L_max=3)");
    TORCH_CHECK(input2.size(1) == 16, "input2 must have 16 components (L_max=3)");
    TORCH_CHECK(input1.size(0) == input2.size(0), "input1 and input2 must have same N");
    TORCH_CHECK(input1.dtype() == torch::kFloat32, "input1 must be float32");
    TORCH_CHECK(input2.dtype() == torch::kFloat32, "input2 must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");

    const int64_t N = input1.size(0);
    const int C_in = input1.size(1);
    const int C_out = weights.size(2);
    const int num_paths = batteries::get_num_paths_l3();

    TORCH_CHECK(weights.size(0) == num_paths, "weights must have ", num_paths, " paths");
    TORCH_CHECK(weights.size(1) == C_in, "weights C_in must match input1");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input1.device());
    torch::Tensor output = torch::empty({N, C_out, 16}, options);

    if (N == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::tensor_product_l3(
        get_ptr<float>(input1, "input1"),
        get_ptr<float>(input2, "input2"),
        get_ptr<float>(weights, "weights"),
        get_ptr_mut<float>(output, "output"),
        N,
        C_in,
        C_out,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

/**
 * Get tensor product path info.
 * Returns [num_paths, 3] array with (l1, l2, l_out) for each path.
 */
torch::Tensor get_tp_path_info() {
    const int num_paths = batteries::get_num_paths_l3();
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor path_info = torch::empty({num_paths, 3}, options);
    batteries::get_path_info_l3(path_info.data_ptr<int>());
    return path_info;
}

int get_tp_num_paths() {
    return batteries::get_num_paths_l3();
}

// ============================================================================
// Backward pass bindings
// ============================================================================

/**
 * Backward for spherical harmonics
 */
torch::Tensor spherical_harmonics_backward(
    const torch::Tensor& edge_vectors,
    const torch::Tensor& grad_Y_lm
) {
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N, 3]");
    TORCH_CHECK(grad_Y_lm.dim() == 2, "grad_Y_lm must be 2D");
    TORCH_CHECK(edge_vectors.size(0) == grad_Y_lm.size(0), "Batch sizes must match");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(grad_Y_lm.dtype() == torch::kFloat32, "grad_Y_lm must be float32");

    const int64_t N = edge_vectors.size(0);
    const int64_t num_components = grad_Y_lm.size(1);

    // Infer L_max from num_components
    int L_max = -1;
    for (int l = 0; l <= 6; ++l) {
        if ((l + 1) * (l + 1) == num_components) {
            L_max = l;
            break;
        }
    }
    TORCH_CHECK(L_max >= 0, "Invalid number of components: ", num_components);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(edge_vectors.device());
    torch::Tensor grad_xyz = torch::empty({N, 3}, options);

    if (N == 0) return grad_xyz;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::spherical_harmonics_backward_dispatch(
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr<float>(grad_Y_lm, "grad_Y_lm"),
        get_ptr_mut<float>(grad_xyz, "grad_xyz"),
        N,
        L_max,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return grad_xyz;
}

/**
 * Backward for simple tensor product
 */
std::tuple<torch::Tensor, torch::Tensor> tensor_product_simple_backward(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& grad_output
) {
    TORCH_CHECK(input1.dim() == 2 && input1.size(1) == 16, "input1 must be [N, 16]");
    TORCH_CHECK(input2.dim() == 2 && input2.size(1) == 16, "input2 must be [N, 16]");
    TORCH_CHECK(grad_output.dim() == 2 && grad_output.size(1) == 16, "grad_output must be [N, 16]");
    TORCH_CHECK(input1.size(0) == input2.size(0), "Batch sizes must match");
    TORCH_CHECK(input1.size(0) == grad_output.size(0), "Batch sizes must match");

    const int64_t N = input1.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input1.device());
    torch::Tensor grad_input1 = torch::empty({N, 16}, options);
    torch::Tensor grad_input2 = torch::empty({N, 16}, options);

    if (N == 0) return std::make_tuple(grad_input1, grad_input2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::tensor_product_simple_backward_l3(
        get_ptr<float>(input1, "input1"),
        get_ptr<float>(input2, "input2"),
        get_ptr<float>(grad_output, "grad_output"),
        get_ptr_mut<float>(grad_input1, "grad_input1"),
        get_ptr_mut<float>(grad_input2, "grad_input2"),
        N,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return std::make_tuple(grad_input1, grad_input2);
}

/**
 * Backward for full tensor product with channels and weights
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensor_product_backward(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& weights,
    const torch::Tensor& grad_output,
    bool needs_input1_grad,
    bool needs_input2_grad,
    bool needs_weights_grad
) {
    TORCH_CHECK(input1.dim() == 3, "input1 must be 3D [N, C_in, 16]");
    TORCH_CHECK(input2.dim() == 2, "input2 must be 2D [N, 16]");
    TORCH_CHECK(weights.dim() == 3, "weights must be 3D [num_paths, C_in, C_out]");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be 3D [N, C_out, 16]");

    const int64_t N = input1.size(0);
    const int C_in = input1.size(1);
    const int C_out = weights.size(2);
    const int num_paths = batteries::get_num_paths_l3();

    TORCH_CHECK(weights.size(0) == num_paths, "weights must have ", num_paths, " paths");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input1.device());

    torch::Tensor grad_input1, grad_input2, grad_weights;

    if (needs_input1_grad) {
        grad_input1 = torch::empty({N, C_in, 16}, options);
    }
    if (needs_input2_grad) {
        grad_input2 = torch::empty({N, 16}, options);
    }
    if (needs_weights_grad) {
        grad_weights = torch::empty({num_paths, C_in, C_out}, options);
    }

    if (N == 0) {
        if (!grad_input1.defined()) grad_input1 = torch::empty({0, C_in, 16}, options);
        if (!grad_input2.defined()) grad_input2 = torch::empty({0, 16}, options);
        if (!grad_weights.defined()) grad_weights = torch::zeros({num_paths, C_in, C_out}, options);
        return std::make_tuple(grad_input1, grad_input2, grad_weights);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::tensor_product_backward_l3(
        get_ptr<float>(input1, "input1"),
        get_ptr<float>(input2, "input2"),
        get_ptr<float>(weights, "weights"),
        get_ptr<float>(grad_output, "grad_output"),
        needs_input1_grad ? get_ptr_mut<float>(grad_input1, "grad_input1") : nullptr,
        needs_input2_grad ? get_ptr_mut<float>(grad_input2, "grad_input2") : nullptr,
        needs_weights_grad ? get_ptr_mut<float>(grad_weights, "grad_weights") : nullptr,
        N,
        C_in,
        C_out,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    // Return empty tensors for gradients that weren't computed
    if (!grad_input1.defined()) grad_input1 = torch::Tensor();
    if (!grad_input2.defined()) grad_input2 = torch::Tensor();
    if (!grad_weights.defined()) grad_weights = torch::Tensor();

    return std::make_tuple(grad_input1, grad_input2, grad_weights);
}

// ============================================================================
// Fused Message Passing Bindings
// ============================================================================

/**
 * Convert COO edge_index to CSR format
 */
std::tuple<torch::Tensor, torch::Tensor> edge_index_to_csr(
    const torch::Tensor& edge_index,
    int64_t N_atoms
) {
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must be 2D [2, N_edges]");
    TORCH_CHECK(edge_index.size(0) == 2, "edge_index first dim must be 2");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");

    const int64_t N_edges = edge_index.size(1);

    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(edge_index.device());

    torch::Tensor row_ptr = torch::empty({N_atoms + 1}, options);
    torch::Tensor col_idx = torch::empty({N_edges}, options);

    if (N_edges == 0) {
        row_ptr.zero_();
        return std::make_tuple(row_ptr, col_idx);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::edge_index_to_csr(
        get_ptr<int64_t>(edge_index, "edge_index"),
        get_ptr_mut<int64_t>(row_ptr, "row_ptr"),
        get_ptr_mut<int64_t>(col_idx, "col_idx"),
        N_atoms,
        N_edges,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return std::make_tuple(row_ptr, col_idx);
}

/**
 * Fused message passing (simple - no weights)
 */
torch::Tensor fused_message_passing_simple_py(
    const torch::Tensor& node_features,
    const torch::Tensor& edge_vectors,
    const torch::Tensor& edge_index,
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx
) {
    TORCH_CHECK(node_features.dim() == 3, "node_features must be 3D [N_atoms, C, 16]");
    TORCH_CHECK(node_features.size(2) == 16, "node_features must have 16 irrep components");
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D [N_edges, 3]");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N_edges, 3]");
    TORCH_CHECK(edge_index.dim() == 2, "edge_index must be 2D [2, N_edges]");
    TORCH_CHECK(edge_index.size(0) == 2, "edge_index first dim must be 2");
    TORCH_CHECK(node_features.dtype() == torch::kFloat32, "node_features must be float32");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(row_ptr.dtype() == torch::kInt64, "row_ptr must be int64");
    TORCH_CHECK(col_idx.dtype() == torch::kInt64, "col_idx must be int64");

    const int64_t N_atoms = node_features.size(0);
    const int64_t N_edges = edge_vectors.size(0);
    const int C = node_features.size(1);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(node_features.device());
    torch::Tensor output = torch::empty({N_atoms, C, 16}, options);

    if (N_atoms == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::fused_message_passing_simple(
        get_ptr<float>(node_features, "node_features"),
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr<int64_t>(edge_index, "edge_index"),
        get_ptr<int64_t>(row_ptr, "row_ptr"),
        get_ptr<int64_t>(col_idx, "col_idx"),
        get_ptr_mut<float>(output, "output"),
        N_atoms,
        N_edges,
        C,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

/**
 * Fused message passing with weights
 */
torch::Tensor fused_message_passing_py(
    const torch::Tensor& node_features,
    const torch::Tensor& edge_vectors,
    const torch::Tensor& weights,
    const torch::Tensor& row_ptr,
    const torch::Tensor& col_idx
) {
    TORCH_CHECK(node_features.dim() == 3, "node_features must be 3D [N_atoms, C_in, 16]");
    TORCH_CHECK(node_features.size(2) == 16, "node_features must have 16 irrep components");
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D [N_edges, 3]");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N_edges, 3]");
    TORCH_CHECK(weights.dim() == 2, "weights must be 2D [C_in, C_out]");
    TORCH_CHECK(node_features.dtype() == torch::kFloat32, "node_features must be float32");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(row_ptr.dtype() == torch::kInt64, "row_ptr must be int64");
    TORCH_CHECK(col_idx.dtype() == torch::kInt64, "col_idx must be int64");

    const int64_t N_atoms = node_features.size(0);
    const int64_t N_edges = edge_vectors.size(0);
    const int C_in = node_features.size(1);
    const int C_out = weights.size(1);

    TORCH_CHECK(weights.size(0) == C_in, "weights first dim must match C_in");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(node_features.device());
    torch::Tensor output = torch::empty({N_atoms, C_out, 16}, options);

    if (N_atoms == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::fused_message_passing(
        get_ptr<float>(node_features, "node_features"),
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr<float>(weights, "weights"),
        get_ptr<int64_t>(row_ptr, "row_ptr"),
        get_ptr<int64_t>(col_idx, "col_idx"),
        get_ptr_mut<float>(output, "output"),
        N_atoms,
        N_edges,
        C_in,
        C_out,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return output;
}

// ============================================================================
// Fused SH + TP Bindings (separate scatter)
// ============================================================================

/**
 * Fused SH + TP (simple - no weights)
 * Computes messages = sh(edge_vectors) x node_features[sources]
 * One thread per (edge, channel) - maintains parallelism of unfused TP
 */
torch::Tensor fused_sh_tp_simple_py(
    const torch::Tensor& edge_vectors,
    const torch::Tensor& node_features,
    const torch::Tensor& source_idx
) {
    TORCH_CHECK(edge_vectors.dim() == 2, "edge_vectors must be 2D [N_edges, 3]");
    TORCH_CHECK(edge_vectors.size(1) == 3, "edge_vectors must have shape [N_edges, 3]");
    TORCH_CHECK(node_features.dim() == 3, "node_features must be 3D [N_atoms, C, 16]");
    TORCH_CHECK(node_features.size(2) == 16, "node_features must have 16 irrep components");
    TORCH_CHECK(source_idx.dim() == 1, "source_idx must be 1D [N_edges]");
    TORCH_CHECK(edge_vectors.dtype() == torch::kFloat32, "edge_vectors must be float32");
    TORCH_CHECK(node_features.dtype() == torch::kFloat32, "node_features must be float32");
    TORCH_CHECK(source_idx.dtype() == torch::kInt64, "source_idx must be int64");

    const int64_t N_edges = edge_vectors.size(0);
    const int C = node_features.size(1);

    TORCH_CHECK(source_idx.size(0) == N_edges, "source_idx must have N_edges elements");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(node_features.device());
    torch::Tensor messages = torch::empty({N_edges, C, 16}, options);

    if (N_edges == 0) return messages;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cudaError_t err = batteries::fused_sh_tp_simple(
        get_ptr<float>(edge_vectors, "edge_vectors"),
        get_ptr<float>(node_features, "node_features"),
        get_ptr<int64_t>(source_idx, "source_idx"),
        get_ptr_mut<float>(messages, "messages"),
        N_edges,
        C,
        stream
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return messages;
}

PYBIND11_MODULE(_batteries, m) {
    m.doc() = "Batteries: Optimized CUDA kernels for equivariant GNNs";

    m.def("spherical_harmonics", &spherical_harmonics,
          py::arg("edge_vectors"),
          py::arg("L_max"),
          R"doc(
Compute real spherical harmonics Y_lm.

Args:
    edge_vectors: Float tensor [N, 3] of unit vectors (must be normalized)
    L_max: Maximum angular momentum (0-6)

Returns:
    Y_lm: Float tensor [N, (L_max+1)^2] containing spherical harmonics
          ordered as [Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ...]
)doc");

    m.def("spherical_harmonics_inplace", &spherical_harmonics_inplace,
          py::arg("edge_vectors"),
          py::arg("Y_lm"),
          py::arg("L_max"),
          "Compute spherical harmonics with pre-allocated output buffer");

    m.def("tensor_product_simple", &tensor_product_simple,
          py::arg("input1"),
          py::arg("input2"),
          R"doc(
Simple tensor product without channels or weights.

Computes pure CG contraction for L_max=3:
  output[n, :] = sum_{l1,l2,l_out} CG * input1[n, m1] * input2[n, m2]

Args:
    input1: Float tensor [N, 16] - first input (16 = (3+1)^2 components)
    input2: Float tensor [N, 16] - second input

Returns:
    output: Float tensor [N, 16] - tensor product result
)doc");

    m.def("tensor_product", &tensor_product,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("weights"),
          R"doc(
Fully connected tensor product with channels and learnable weights.

For each sample n and output channel c_out:
  output[n, c_out, :] = sum_{path} sum_{c_in} weight[path, c_in, c_out] *
                        sum_{cg} cg_coeff * input1[n, c_in, m1] * input2[n, m2]

Args:
    input1: Float tensor [N, C_in, 16] - node features with channels
    input2: Float tensor [N, 16] - edge features (spherical harmonics)
    weights: Float tensor [num_paths, C_in, C_out] - learnable weights

Returns:
    output: Float tensor [N, C_out, 16] - output features
)doc");

    m.def("get_tp_path_info", &get_tp_path_info,
          "Get tensor product path info as [num_paths, 3] array of (l1, l2, l_out)");

    m.def("get_tp_num_paths", &get_tp_num_paths,
          "Get number of tensor product paths for L_max=3");

    // Backward pass functions
    m.def("spherical_harmonics_backward", &spherical_harmonics_backward,
          py::arg("edge_vectors"),
          py::arg("grad_Y_lm"),
          R"doc(
Backward pass for spherical harmonics.

Args:
    edge_vectors: Float tensor [N, 3] of unit vectors (saved from forward)
    grad_Y_lm: Float tensor [N, (L_max+1)^2] gradient w.r.t. output

Returns:
    grad_xyz: Float tensor [N, 3] gradient w.r.t. input edge vectors
)doc");

    m.def("tensor_product_simple_backward", &tensor_product_simple_backward,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("grad_output"),
          R"doc(
Backward pass for simple tensor product.

Args:
    input1: Float tensor [N, 16] (saved from forward)
    input2: Float tensor [N, 16] (saved from forward)
    grad_output: Float tensor [N, 16] gradient w.r.t. output

Returns:
    Tuple of (grad_input1, grad_input2)
)doc");

    m.def("tensor_product_backward", &tensor_product_backward,
          py::arg("input1"),
          py::arg("input2"),
          py::arg("weights"),
          py::arg("grad_output"),
          py::arg("needs_input1_grad") = true,
          py::arg("needs_input2_grad") = true,
          py::arg("needs_weights_grad") = true,
          R"doc(
Backward pass for full tensor product with channels and weights.

Args:
    input1: Float tensor [N, C_in, 16] (saved from forward)
    input2: Float tensor [N, 16] (saved from forward)
    weights: Float tensor [num_paths, C_in, C_out] (saved from forward)
    grad_output: Float tensor [N, C_out, 16] gradient w.r.t. output
    needs_input1_grad: Whether to compute gradient for input1
    needs_input2_grad: Whether to compute gradient for input2
    needs_weights_grad: Whether to compute gradient for weights

Returns:
    Tuple of (grad_input1, grad_input2, grad_weights)
)doc");

    // =========================================================================
    // Fused Message Passing
    // =========================================================================

    m.def("edge_index_to_csr", &edge_index_to_csr,
          py::arg("edge_index"),
          py::arg("N_atoms"),
          R"doc(
Convert COO edge_index to CSR format.

Args:
    edge_index: Long tensor [2, N_edges] where [0,:] = source, [1,:] = target
    N_atoms: Number of atoms

Returns:
    Tuple of (row_ptr, col_idx):
        row_ptr: Long tensor [N_atoms+1] CSR row pointers
        col_idx: Long tensor [N_edges] source indices for each edge
)doc");

    m.def("fused_message_passing_simple", &fused_message_passing_simple_py,
          py::arg("node_features"),
          py::arg("edge_vectors"),
          py::arg("edge_index"),
          py::arg("row_ptr"),
          py::arg("col_idx"),
          R"doc(
Fused message passing: SH -> TP -> scatter_add in one kernel.

Combines spherical harmonics, tensor product, and scatter-add
into a single kernel, eliminating global memory round-trips.

Args:
    node_features: Float tensor [N_atoms, C, 16] - source node features
    edge_vectors: Float tensor [N_edges, 3] - unit edge vectors
    edge_index: Long tensor [2, N_edges] - source/target pairs
    row_ptr: Long tensor [N_atoms+1] - CSR row pointers
    col_idx: Long tensor [N_edges] - original edge indices in CSR order

Returns:
    output: Float tensor [N_atoms, C, 16] - accumulated messages
)doc");

    m.def("fused_message_passing", &fused_message_passing_py,
          py::arg("node_features"),
          py::arg("edge_vectors"),
          py::arg("weights"),
          py::arg("row_ptr"),
          py::arg("col_idx"),
          R"doc(
Fused message passing with learnable weights.

Args:
    node_features: Float tensor [N_atoms, C_in, 16] - source node features
    edge_vectors: Float tensor [N_edges, 3] - unit edge vectors
    weights: Float tensor [C_in, C_out] - channel mixing weights
    row_ptr: Long tensor [N_atoms+1] - CSR row pointers
    col_idx: Long tensor [N_edges] - source atom indices

Returns:
    output: Float tensor [N_atoms, C_out, 16] - accumulated messages
)doc");

    // =========================================================================
    // Fused SH + TP (separate scatter)
    // =========================================================================

    m.def("fused_sh_tp_simple", &fused_sh_tp_simple_py,
          py::arg("edge_vectors"),
          py::arg("node_features"),
          py::arg("source_idx"),
          R"doc(
Fused spherical harmonics + tensor product (no weights).

Combines SH computation and tensor product in one kernel,
eliminating the intermediate Y_lm tensor from global memory.
One thread per (edge, channel) pair maintains full parallelism.

Pipeline:
  edge_vectors -> [SH in registers] -> TP -> messages

Memory savings: eliminates N_edges * 16 * 4 bytes (Y_lm tensor)

Args:
    edge_vectors: Float tensor [N_edges, 3] - unit edge vectors
    node_features: Float tensor [N_atoms, C, 16] - source node features
    source_idx: Long tensor [N_edges] - source atom index for each edge

Returns:
    messages: Float tensor [N_edges, C, 16] - edge messages
              (scatter separately with index_add_)
)doc");
}
