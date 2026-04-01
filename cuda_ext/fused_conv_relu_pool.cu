#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void fused_conv_relu_pool_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n,
    int c,
    int h,
    int w,
    int out_channels,
    int kh,
    int kw,
    int out_h,
    int out_w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * out_channels * out_h * out_w;
  if (idx >= total) {
    return;
  }

  int t = idx;
  const int ow = t % out_w;
  t /= out_w;
  const int oh = t % out_h;
  t /= out_h;
  const int oc = t % out_channels;
  const int ni = t / out_channels;

  float max_value = -3.402823466e+38F;

  const int conv_y_base = oh * 2;
  const int conv_x_base = ow * 2;

  for (int py = 0; py < 2; ++py) {
    for (int px = 0; px < 2; ++px) {
      const int conv_y = conv_y_base + py;
      const int conv_x = conv_x_base + px;

      float acc = bias[oc];
      for (int ic = 0; ic < c; ++ic) {
        for (int ky = 0; ky < kh; ++ky) {
          const int in_y = conv_y + ky;
          for (int kx = 0; kx < kw; ++kx) {
            const int in_x = conv_x + kx;
            const int input_offset = ((ni * c + ic) * h + in_y) * w + in_x;
            const int weight_offset = ((oc * c + ic) * kh + ky) * kw + kx;
            acc += input[input_offset] * weight[weight_offset];
          }
        }
      }

      if (acc < 0.0f) {
        acc = 0.0f;
      }
      if (acc > max_value) {
        max_value = acc;
      }
    }
  }

  output[idx] = max_value;
}

}  // namespace

torch::Tensor fused_conv_relu_pool_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
  const auto n = static_cast<int>(input.size(0));
  const auto c = static_cast<int>(input.size(1));
  const auto h = static_cast<int>(input.size(2));
  const auto w = static_cast<int>(input.size(3));

  const auto out_channels = static_cast<int>(weight.size(0));
  const auto kh = static_cast<int>(weight.size(2));
  const auto kw = static_cast<int>(weight.size(3));

  const auto conv_out_h = h - kh + 1;
  const auto conv_out_w = w - kw + 1;

  const int out_h = conv_out_h / 2;
  const int out_w = conv_out_w / 2;

  auto output = torch::empty({n, out_channels, out_h, out_w}, input.options());

  const int threads = 256;
  const int total = n * out_channels * out_h * out_w;
  const int blocks = (total + threads - 1) / threads;

  auto stream = at::cuda::getDefaultCUDAStream();
  fused_conv_relu_pool_forward_kernel<<<blocks, threads, 0, stream>>>(
      input.data_ptr<float>(),
      weight.data_ptr<float>(),
      bias.data_ptr<float>(),
      output.data_ptr<float>(),
      n,
      c,
      h,
      w,
      out_channels,
      kh,
      kw,
      out_h,
      out_w);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
