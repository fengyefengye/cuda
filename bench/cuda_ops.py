"""Custom CUDA extension loader and call helpers."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.cpp_extension import load

EXT_NAME = "fused_conv_relu_pool_cuda"

_ext_module: Optional[Any] = None
_import_error: Optional[str] = None
_build_error: Optional[str] = None


def _add_windows_dll_paths() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = []

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    candidates.append(torch_lib_dir)

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    default_cuda_bin = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin")
    candidates.append(default_cuda_bin)

    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            os.add_dll_directory(str(path))


def _try_import() -> Optional[Any]:
    global _ext_module, _import_error

    try:
        _add_windows_dll_paths()
        _ext_module = importlib.import_module(EXT_NAME)
        _import_error = None
    except Exception as exc:  # pylint: disable=broad-except
        _ext_module = None
        _import_error = repr(exc)

    return _ext_module


def load_extension(build_if_missing: bool = False, verbose: bool = False) -> Optional[Any]:
    global _ext_module, _build_error

    if _ext_module is not None:
        return _ext_module

    _try_import()
    if _ext_module is not None or not build_if_missing:
        return _ext_module

    project_root = Path(__file__).resolve().parents[1]
    sources = [
        str(project_root / "cuda_ext" / "fused_conv_relu_pool_bindings.cpp"),
        str(project_root / "cuda_ext" / "fused_conv_relu_pool.cu"),
    ]

    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math"]

    try:
        _ext_module = load(
            name=EXT_NAME,
            sources=sources,
            verbose=verbose,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
        )
        _build_error = None
    except Exception as exc:  # pylint: disable=broad-except
        _ext_module = None
        _build_error = repr(exc)

    return _ext_module


def extension_status() -> Dict[str, object]:
    return {
        "loaded": _ext_module is not None,
        "import_error": _import_error,
        "build_error": _build_error,
    }


def fused_conv_relu_pool(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if _ext_module is None:
        raise RuntimeError("CUDA extension is not loaded")
    return _ext_module.forward(input_tensor, weight, bias)
