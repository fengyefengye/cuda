import os

from setuptools import setup
import torch.utils.cpp_extension as cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Optional escape hatch for environments where toolkit and wheel CUDA versions differ.
# This is intentionally opt-in to avoid masking real compatibility issues by default.
if os.environ.get("ALLOW_CUDA_MISMATCH", "0") == "1":
    cpp_ext._check_cuda_version = lambda *args, **kwargs: None

setup(
    name="fused_conv_relu_pool_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_conv_relu_pool_cuda",
            sources=[
                "cuda_ext/fused_conv_relu_pool_bindings.cpp",
                "cuda_ext/fused_conv_relu_pool.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
