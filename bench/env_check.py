"""Environment checker for CUDA benchmark prerequisites."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Dict, List, Tuple


def _run_command(cmd: List[str]) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:  # pylint: disable=broad-except
        return False, repr(exc)

    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _torch_checks() -> Dict[str, str]:
    try:
        import torch
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "torch_import": "FAIL",
            "torch_error": repr(exc),
            "torch_version": "n/a",
            "cuda_available": "n/a",
            "torch_cuda_version": "n/a",
            "cudnn_version": "n/a",
        }

    backends_obj = getattr(torch, "backends", None)
    cudnn_obj = getattr(backends_obj, "cudnn", None)
    cudnn_version = cudnn_obj.version() if cudnn_obj else None

    return {
        "torch_import": "OK",
        "torch_error": "",
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": str(cudnn_version),
    }


def main() -> int:
    report: Dict[str, str] = {}
    report.update(_torch_checks())

    nvcc_path = shutil.which("nvcc")
    report["nvcc_in_path"] = str(bool(nvcc_path))
    report["nvcc_path"] = nvcc_path or ""

    if nvcc_path:
        ok, nvcc_output = _run_command(["nvcc", "--version"])
        report["nvcc_version_ok"] = str(ok)
        report["nvcc_version_output"] = nvcc_output
    else:
        report["nvcc_version_ok"] = "False"
        report["nvcc_version_output"] = ""

    cl_path = shutil.which("cl")
    report["cl_in_path"] = str(bool(cl_path))
    report["cl_path"] = cl_path or ""

    if cl_path:
        ok, cl_output = _run_command(["cl"])
        report["cl_invocation_ok"] = str(ok)
        report["cl_output"] = cl_output
    else:
        report["cl_invocation_ok"] = "False"
        report["cl_output"] = ""

    print("=== Environment Check ===")
    for key in sorted(report.keys()):
        value = report[key]
        if key.endswith("_output") and len(value) > 350:
            value = value[:350] + "..."
        print(f"{key}: {value}")

    hard_fail = report["torch_import"] != "OK"
    if hard_fail:
        print("\nResult: FAILED (torch import failed)")
        return 1

    print("\nResult: COMPLETED (check fields above for missing items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
