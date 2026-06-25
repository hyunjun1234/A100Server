
#include <torch/extension.h>
#include <vector>

torch::Tensor conv_transpose_fused_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias);

torch::Tensor conv_transpose_fused(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
    
    return conv_transpose_fused_cuda(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused", &conv_transpose_fused, "Fused ConvTranspose2d operations");
}
