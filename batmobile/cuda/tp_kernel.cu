#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tp_stub_kernel(float* out) {
    out[0] = 999.0f;
}

torch::Tensor tp_stub_cuda() {
    auto out = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
    tp_stub_kernel<<<1,1>>>(out.data_ptr<float>());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tp_stub_cuda", &tp_stub_cuda, "Batmobile CUDA Stub Kernel");
}