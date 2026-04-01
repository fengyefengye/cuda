#include <torch/extension.h>

#include <vector>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOAT(x)

}  // namespace

torch::Tensor fused_conv_relu_pool_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias);

torch::Tensor fused_conv_relu_pool_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  CHECK_INPUT(bias);

  TORCH_CHECK(input.dim() == 4, "input must be 4D NCHW tensor");
  TORCH_CHECK(weight.dim() == 4, "weight must be 4D OIHW tensor");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D tensor");

  TORCH_CHECK(input.size(1) == weight.size(1), "input channels and weight channels mismatch");
  TORCH_CHECK(weight.size(0) == bias.size(0), "weight output channels and bias size mismatch");
  TORCH_CHECK(input.size(2) >= weight.size(2), "input height must be >= kernel height");
  TORCH_CHECK(input.size(3) >= weight.size(3), "input width must be >= kernel width");

  const auto conv_out_h = input.size(2) - weight.size(2) + 1;
  const auto conv_out_w = input.size(3) - weight.size(3) + 1;
  TORCH_CHECK(conv_out_h % 2 == 0, "conv output height must be divisible by 2 for pooling");
  TORCH_CHECK(conv_out_w % 2 == 0, "conv output width must be divisible by 2 for pooling");

  return fused_conv_relu_pool_forward_cuda(input, weight, bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_conv_relu_pool_forward, "Fused Conv+ReLU+Pool forward (CUDA)");
}
