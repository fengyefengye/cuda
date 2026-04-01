"""Model definitions and factory for eager, compile, and custom CUDA paths."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import cuda_ops


class LeNetEager(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNetFused(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1_weight = nn.Parameter(torch.empty(6, 1, 5, 5))
        self.conv1_bias = nn.Parameter(torch.empty(6))

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self._use_extension = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.conv1_weight, a=5**0.5)
        fan_in = self.conv1_weight.size(1) * self.conv1_weight.size(2) * self.conv1_weight.size(3)
        bound = 1.0 / fan_in**0.5
        nn.init.uniform_(self.conv1_bias, -bound, bound)

    def set_use_extension(self, enabled: bool) -> None:
        self._use_extension = enabled

    @staticmethod
    def from_eager(eager_model: LeNetEager, try_load_extension: bool = False, build_if_missing: bool = False) -> "LeNetFused":
        fused = LeNetFused(num_classes=eager_model.fc3.out_features)
        if eager_model.conv1.bias is None:
            raise ValueError("LeNetEager conv1 is expected to have bias")
        with torch.no_grad():
            fused.conv1_weight.copy_(eager_model.conv1.weight)
            fused.conv1_bias.copy_(eager_model.conv1.bias)
            fused.conv2.load_state_dict(eager_model.conv2.state_dict())
            fused.fc1.load_state_dict(eager_model.fc1.state_dict())
            fused.fc2.load_state_dict(eager_model.fc2.state_dict())
            fused.fc3.load_state_dict(eager_model.fc3.state_dict())

        if try_load_extension:
            ext = cuda_ops.load_extension(build_if_missing=build_if_missing)
            fused.set_use_extension(ext is not None)
        return fused

    def _first_block_fallback(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, padding=0)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_fused = (
            self._use_extension
            and x.is_cuda
            and x.dtype == torch.float32
            and self.conv1_weight.is_cuda
            and self.conv1_bias.is_cuda
        )

        if use_fused:
            x = cuda_ops.fused_conv_relu_pool(x.contiguous(), self.conv1_weight.contiguous(), self.conv1_bias.contiguous())
        else:
            x = self._first_block_fallback(x)

        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model_variants(
    device: torch.device,
    enable_compile: bool = True,
    compile_backend: Optional[str] = None,
    compile_mode: Optional[str] = "default",
    try_load_extension: bool = True,
    build_extension_if_missing: bool = False,
) -> Tuple[Dict[str, nn.Module], Dict[str, Optional[str]]]:
    eager = LeNetEager().to(device).eval()

    compiled_source = LeNetEager().to(device).eval()
    compiled_source.load_state_dict(eager.state_dict())

    compile_model: nn.Module = compiled_source
    compile_error: Optional[str] = None
    compile_enabled = False
    compile_backend_used = "none"

    if enable_compile and hasattr(torch, "compile"):
        try:
            kwargs = {}
            if compile_backend:
                kwargs["backend"] = compile_backend
            if compile_mode:
                kwargs["mode"] = compile_mode
            compile_model = cast(nn.Module, torch.compile(compiled_source, **kwargs))
            compile_enabled = True
            compile_backend_used = "torch.compile"
        except Exception as exc:  # pylint: disable=broad-except
            compile_error = repr(exc)

    if enable_compile and not compile_enabled:
        try:
            script_fn = getattr(torch.jit, "script")
            compile_model = cast(nn.Module, script_fn(compiled_source))
            compile_enabled = True
            compile_backend_used = "torch.jit.script"
            if compile_error:
                compile_error = compile_error + " | fallback=torch.jit.script"
        except Exception as exc:  # pylint: disable=broad-except
            if compile_error:
                compile_error = compile_error + " | jit_fallback_error=" + repr(exc)
            else:
                compile_error = repr(exc)
            compile_model = compiled_source

    fused = LeNetFused.from_eager(
        eager_model=eager,
        try_load_extension=try_load_extension,
        build_if_missing=build_extension_if_missing,
    ).to(device).eval()

    ext_status = cuda_ops.extension_status()

    models = {
        "eager": eager,
        "compile": compile_model,
        "fused": fused,
    }

    info = {
        "compile_enabled": str(compile_enabled),
        "compile_backend_used": compile_backend_used,
        "compile_error": compile_error,
        "fused_extension_loaded": str(bool(ext_status.get("loaded"))),
        "fused_extension_import_error": ext_status.get("import_error"),
        "fused_extension_build_error": ext_status.get("build_error"),
    }

    return models, info
