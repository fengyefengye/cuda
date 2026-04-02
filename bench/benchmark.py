"""Unified benchmark entry for eager, torch.compile, and custom CUDA fused paths."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from bench.config import (
    ABS_ERR_TOL,
    DEFAULT_BATCH_SIZES,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_ITERS,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    REL_ERR_TOL,
)
from bench.models import build_model_variants
from bench.utils import (
    dump_json,
    ensure_dir,
    format_float,
    get_torch_dtype,
    max_abs_error,
    parse_batch_sizes,
    relative_error,
    set_global_seed,
    summarize_latencies,
    to_markdown_table,
)


def _resolve_device(device_name: str) -> torch.device:
    name = device_name.lower().strip()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device


def _benchmark_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    device = input_tensor.device
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: List[float] = []

    with torch.no_grad():
        if device.type == "cuda":
            for _ in range(iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(device)
                start_event.record(stream)
                _ = model(input_tensor)
                end_event.record(stream)
                end_event.synchronize()
                latencies_ms.append(float(start_event.elapsed_time(end_event)))
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    latency_stats = summarize_latencies(latencies_ms)

    median_ms = latency_stats["median_ms"]
    throughput = float(input_tensor.size(0) * 1000.0 / max(median_ms, 1e-12))

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))

    output = {
        **latency_stats,
        "throughput_samples_per_s": throughput,
        "peak_memory_mb": peak_mem_mb,
        "iters": float(iters),
        "warmup": float(warmup),
    }
    return output


def _correctness_report(
    models: Dict[str, torch.nn.Module],
    input_tensor: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    with torch.no_grad():
        reference = models["eager"](input_tensor)

    report: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        with torch.no_grad():
            out = model(input_tensor)

        abs_err = max_abs_error(reference, out)
        rel_err = relative_error(reference, out)

        report[name] = {
            "max_abs_err": abs_err,
            "relative_err": rel_err,
            "within_threshold": float(abs_err <= ABS_ERR_TOL and rel_err <= REL_ERR_TOL),
        }

    return report


def _select_modes(all_modes: Iterable[str], requested: str) -> List[str]:
    if requested.strip().lower() == "all":
        return list(all_modes)

    selected = [item.strip() for item in requested.split(",") if item.strip()]
    valid = set(all_modes)
    for mode in selected:
        if mode not in valid:
            raise ValueError(f"unsupported mode '{mode}', expected one of {sorted(valid)}")
    return selected


def _write_markdown_summary(path: Path, benchmark_data: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    rows = []
    for batch, mode_values in benchmark_data.items():
        for mode_name, metrics in mode_values.items():
            rows.append(
                {
                    "batch": batch,
                    "mode": mode_name,
                    "median_ms": format_float(metrics["median_ms"], 4),
                    "p95_ms": format_float(metrics["p95_ms"], 4),
                    "p99_ms": format_float(metrics["p99_ms"], 4),
                    "throughput_samples_per_s": format_float(metrics["throughput_samples_per_s"], 2),
                    "peak_memory_mb": format_float(metrics["peak_memory_mb"], 2),
                }
            )

    table = to_markdown_table(
        rows,
        columns=[
            "batch",
            "mode",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "throughput_samples_per_s",
            "peak_memory_mb",
        ],
    )

    path.write_text(table + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LeNet execution path benchmark")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--batch-sizes", type=str, default=",".join(str(x) for x in DEFAULT_BATCH_SIZES))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)

    parser.add_argument("--modes", type=str, default="all", help="comma list from eager,compile,fused or 'all'")
    parser.add_argument("--skip-correctness", action="store_true")

    parser.add_argument("--compile-backend", type=str, default="")
    parser.add_argument("--compile-mode", type=str, default="default")
    parser.add_argument("--enable-cudagraph", action="store_true", help="add *_graph modes via CUDA graph capture")
    parser.add_argument("--cudagraph-warmup", type=int, default=3, help="warmup iterations before CUDA graph capture")
    parser.add_argument(
        "--fuse-second-block",
        dest="fuse_second_block",
        action="store_true",
        help="in fused mode, use custom CUDA fused op for both conv blocks (default: enabled)",
    )
    parser.add_argument(
        "--no-fuse-second-block",
        dest="fuse_second_block",
        action="store_false",
        help="disable second-block fusion in fused mode",
    )
    parser.set_defaults(fuse_second_block=True)

    parser.add_argument("--build-extension-if-missing", action="store_true")
    parser.add_argument("--no-try-load-extension", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    set_global_seed(args.seed)
    dtype = get_torch_dtype(args.dtype)
    device = _resolve_device(args.device)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    models, build_info = build_model_variants(
        device=device,
        enable_compile=True,
        compile_backend=args.compile_backend or None,
        compile_mode=args.compile_mode or None,
        try_load_extension=not args.no_try_load_extension,
        build_extension_if_missing=args.build_extension_if_missing,
        fuse_second_block=args.fuse_second_block,
        enable_cudagraph=args.enable_cudagraph,
        cudagraph_warmup=args.cudagraph_warmup,
    )

    selected_modes = _select_modes(models.keys(), args.modes)

    correctness_input = torch.randn(8, *DEFAULT_INPUT_SHAPE, device=device, dtype=dtype)
    correctness = {}
    if not args.skip_correctness:
        correctness = _correctness_report(models, correctness_input)

    benchmark_data: Dict[str, Dict[str, Dict[str, float]]] = {}

    for batch in batch_sizes:
        mode_results: Dict[str, Dict[str, float]] = {}
        input_tensor = torch.randn(batch, *DEFAULT_INPUT_SHAPE, device=device, dtype=dtype)

        for mode_name in selected_modes:
            metrics = _benchmark_model(
                model=models[mode_name],
                input_tensor=input_tensor,
                warmup=args.warmup,
                iters=args.iters,
            )
            mode_results[mode_name] = metrics
            print(
                f"batch={batch:>4} mode={mode_name:<7} "
                f"median={metrics['median_ms']:.4f} ms "
                f"p95={metrics['p95_ms']:.4f} ms "
                f"throughput={metrics['throughput_samples_per_s']:.2f} samples/s"
            )

        benchmark_data[str(batch)] = mode_results

    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)

    summary = {
        "meta": {
            "device": str(device),
            "dtype": str(dtype),
            "batch_sizes": list(batch_sizes),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
        "build_info": build_info,
        "correctness": correctness,
        "benchmarks": benchmark_data,
    }

    json_path = results_dir / "summary.json"
    md_path = results_dir / "summary.md"

    dump_json(json_path, summary)
    _write_markdown_summary(md_path, benchmark_data)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
