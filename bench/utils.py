"""Utility helpers used by benchmark, tests, and reporting."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_batch_sizes(batch_sizes: str) -> Tuple[int, ...]:
    values = []
    for item in batch_sizes.split(","):
        value = int(item.strip())
        if value <= 0:
            raise ValueError("batch size must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one batch size is required")
    return tuple(values)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().strip()
    if name in {"float32", "fp32", "f32"}:
        return torch.float32
    if name in {"float16", "fp16", "f16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def summarize_latencies(latencies_ms: Iterable[float]) -> Dict[str, float]:
    arr = np.array(list(latencies_ms), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty latency list")

    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    num = torch.norm(candidate - reference).item()
    denom = torch.norm(reference).item()
    return float(num / (denom + 1e-12))


def max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((candidate - reference).abs().max().item())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def to_markdown_table(rows: List[Dict[str, str]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def format_float(value: float, digits: int = 4) -> str:
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return str(value)
