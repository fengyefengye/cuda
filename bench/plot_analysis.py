"""Plot and summarize benchmark results from two summary.json files."""

from __future__ import annotations

import argparse
import json
from math import exp, log
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_summary(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


BenchData = Dict[str, Dict[str, Dict[str, float]]]


def _extract_bench(summary: Dict[str, object]) -> BenchData:
    bench_obj = summary.get("benchmarks")
    if not isinstance(bench_obj, dict):
        raise ValueError("summary.json is missing 'benchmarks'")
    return cast(BenchData, bench_obj)


def _sorted_batches(bench: BenchData) -> List[int]:
    return sorted(int(k) for k in bench.keys())


def _extract_metric(
    bench: BenchData,
    batches: List[int],
    mode: str,
    metric: str,
) -> List[float]:
    return [float(bench[str(b)][mode][metric]) for b in batches]


def _speedup(eager_ms: List[float], target_ms: List[float]) -> List[float]:
    return [e / max(t, 1e-12) for e, t in zip(eager_ms, target_ms)]


def _gmean(values: List[float]) -> float:
    return exp(sum(log(max(v, 1e-12)) for v in values) / max(len(values), 1))


def _write_markdown(
    out_path: Path,
    batches: List[int],
    label_a: str,
    label_b: str,
    fused_speed_a: List[float],
    fused_speed_b: List[float],
    fused_graph_speed_a: List[float],
    fused_graph_speed_b: List[float],
) -> None:
    lines: List[str] = []
    lines.append("# 基准分析")
    lines.append("")
    lines.append("## 相对 Eager 的几何平均加速比")
    lines.append("")
    lines.append(f"- {label_a} fused: {_gmean(fused_speed_a):.3f}x")
    lines.append(f"- {label_b} fused: {_gmean(fused_speed_b):.3f}x")
    lines.append(f"- {label_a} fused_graph: {_gmean(fused_graph_speed_a):.3f}x")
    lines.append(f"- {label_b} fused_graph: {_gmean(fused_graph_speed_b):.3f}x")
    lines.append("")
    lines.append("## 分批次加速比（fused 对比 eager）")
    lines.append("")
    lines.append("| batch | " + label_a + " | " + label_b + " | 差值 |")
    lines.append("|---:|---:|---:|---:|")
    for batch, va, vb in zip(batches, fused_speed_a, fused_speed_b):
        lines.append(f"| {batch} | {va:.3f}x | {vb:.3f}x | {vb - va:+.3f}x |")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot_guide(
    out_path: Path,
    batches: List[int],
    label_a: str,
    label_b: str,
    fused_speed_a: List[float],
    fused_speed_b: List[float],
    fused_graph_speed_a: List[float],
    fused_graph_speed_b: List[float],
) -> None:
    faster_a = sum(1 for v in fused_speed_a if v > 1.0)
    faster_b = sum(1 for v in fused_speed_b if v > 1.0)
    fg_faster_a = sum(1 for v in fused_graph_speed_a if v > 1.0)
    fg_faster_b = sum(1 for v in fused_graph_speed_b if v > 1.0)

    lines: List[str] = []
    lines.append("# 看图说明")
    lines.append("")
    lines.append("下面用最直白的方式解释图的含义。")
    lines.append("")
    lines.append("## 1) latency_eager_fused.png")
    lines.append("")
    lines.append("- 横轴是 batch 大小，纵轴是延迟（毫秒）。")
    lines.append("- 线越低越快。")
    lines.append("- 线发生交叉，表示不同 batch 下最优方法不同。")
    lines.append("")
    lines.append("## 2) speedup_compare.png")
    lines.append("")
    lines.append("- 横轴是 batch 大小，纵轴是加速比。")
    lines.append("- 大于 1.0x 表示比 eager 快，小于 1.0x 表示比 eager 慢。")
    lines.append("- 线越高越好。")
    lines.append("")
    lines.append("## 3) speedup_ecf_compare.png（重点）")
    lines.append("")
    lines.append("- 这张图只比较 eager / compile / fused 三类，最适合做主结论。")
    lines.append("- eager 固定为 1.0x 基准线。")
    lines.append("- compile 和 fused 越高越好。")
    lines.append("")
    lines.append("## 快速阅读步骤")
    lines.append("")
    lines.append("- 先看 speedup_ecf_compare.png 选方案。")
    lines.append("- 再看 latency_eager_fused.png 确认毫秒值。")
    lines.append("")
    lines.append("## 自动统计")
    lines.append("")
    lines.append(f"- {label_a} fused 快于 eager 的 batch 点: {faster_a}/{len(batches)}")
    lines.append(f"- {label_b} fused 快于 eager 的 batch 点: {faster_b}/{len(batches)}")
    lines.append(f"- {label_a} fused_graph 快于 eager 的 batch 点: {fg_faster_a}/{len(batches)}")
    lines.append(f"- {label_b} fused_graph 快于 eager 的 batch 点: {fg_faster_b}/{len(batches)}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bar_metric_explanation(
    out_path: Path,
    label_a: str,
    label_b: str,
    fused_speed_a: List[float],
    fused_speed_b: List[float],
    fused_graph_speed_a: List[float],
    fused_graph_speed_b: List[float],
) -> None:
    lines: List[str] = []
    lines.append("# 条状图指标说明")
    lines.append("")
    lines.append("- 选用指标: 相对 eager 的几何平均加速比（基于每个 batch 的中位延迟）。")
    lines.append("- 选择原因: 能公平汇总全 batch 区间，且不容易被个别异常值带偏。")
    lines.append("- 读法: 大于 1.0x 表示快于 eager，越高越好。")
    lines.append("")
    lines.append("## 数值")
    lines.append(f"- {label_a} fused: {_gmean(fused_speed_a):.3f}x")
    lines.append(f"- {label_b} fused: {_gmean(fused_speed_b):.3f}x")
    lines.append(f"- {label_a} fused_graph: {_gmean(fused_graph_speed_a):.3f}x")
    lines.append(f"- {label_b} fused_graph: {_gmean(fused_graph_speed_b):.3f}x")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ecf_markdown(
    out_path: Path,
    label_a: str,
    label_b: str,
    compile_speed_a: List[float],
    compile_speed_b: List[float],
    fused_speed_a: List[float],
    fused_speed_b: List[float],
) -> None:
    lines: List[str] = []
    lines.append("# Eager / Compile / Fused 对比")
    lines.append("")
    lines.append("指标: 相对 eager 的几何平均加速比")
    lines.append("")
    lines.append("| 方案 | 加速比 |")
    lines.append("|---|---:|")
    lines.append(f"| {label_a} eager | 1.000x |")
    lines.append(f"| {label_a} compile | {_gmean(compile_speed_a):.3f}x |")
    lines.append(f"| {label_a} fused | {_gmean(fused_speed_a):.3f}x |")
    lines.append(f"| {label_b} eager | 1.000x |")
    lines.append(f"| {label_b} compile | {_gmean(compile_speed_b):.3f}x |")
    lines.append(f"| {label_b} fused | {_gmean(fused_speed_b):.3f}x |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark comparison and export analysis")
    parser.add_argument("--run-a", type=str, required=True, help="path to first summary.json")
    parser.add_argument("--run-b", type=str, required=True, help="path to second summary.json")
    parser.add_argument("--label-a", type=str, default="run_a")
    parser.add_argument("--label-b", type=str, default="run_b")
    parser.add_argument("--out-dir", type=str, default="results/analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_a = _load_summary(Path(args.run_a))
    run_b = _load_summary(Path(args.run_b))

    bench_a = _extract_bench(run_a)
    bench_b = _extract_bench(run_b)

    batches_a = _sorted_batches(bench_a)
    batches_b = _sorted_batches(bench_b)
    if batches_a != batches_b:
        raise ValueError("run-a and run-b batches do not match")

    batches = batches_a

    eager_a = _extract_metric(bench_a, batches, "eager", "median_ms")
    compile_a = _extract_metric(bench_a, batches, "compile", "median_ms")
    fused_a = _extract_metric(bench_a, batches, "fused", "median_ms")
    fused_graph_a = _extract_metric(bench_a, batches, "fused_graph", "median_ms")

    eager_b = _extract_metric(bench_b, batches, "eager", "median_ms")
    compile_b = _extract_metric(bench_b, batches, "compile", "median_ms")
    fused_b = _extract_metric(bench_b, batches, "fused", "median_ms")
    fused_graph_b = _extract_metric(bench_b, batches, "fused_graph", "median_ms")

    compile_speed_a = _speedup(eager_a, compile_a)
    compile_speed_b = _speedup(eager_b, compile_b)
    fused_speed_a = _speedup(eager_a, fused_a)
    fused_speed_b = _speedup(eager_b, fused_b)
    fused_graph_speed_a = _speedup(eager_a, fused_graph_a)
    fused_graph_speed_b = _speedup(eager_b, fused_graph_b)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: latency comparison for eager/fused.
    plt.figure(figsize=(10, 6))
    plt.plot(batches, eager_a, marker="o", label=f"{args.label_a} eager")
    plt.plot(batches, fused_a, marker="o", label=f"{args.label_a} fused")
    plt.plot(batches, eager_b, marker="s", linestyle="--", label=f"{args.label_b} eager")
    plt.plot(batches, fused_b, marker="s", linestyle="--", label=f"{args.label_b} fused")
    plt.xscale("log", base=2)
    plt.xlabel("Batch 大小")
    plt.ylabel("中位延迟 (ms)")
    plt.title("延迟对比（Eager vs Fused）")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latency_eager_fused.png", dpi=160)
    plt.close()

    # Plot 2: speedup comparison.
    plt.figure(figsize=(10, 6))
    plt.plot(batches, fused_speed_a, marker="o", label=f"{args.label_a} fused/eager")
    plt.plot(batches, fused_speed_b, marker="o", label=f"{args.label_b} fused/eager")
    plt.plot(batches, fused_graph_speed_a, marker="s", label=f"{args.label_a} fused_graph/eager")
    plt.plot(batches, fused_graph_speed_b, marker="s", label=f"{args.label_b} fused_graph/eager")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log", base=2)
    plt.xlabel("Batch 大小")
    plt.ylabel("加速比 (x)")
    plt.title("加速比对比")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_compare.png", dpi=160)
    plt.close()

    # Plot 2b: explicit eager/compile/fused comparison.
    plt.figure(figsize=(10, 6))
    ones = [1.0] * len(batches)
    plt.plot(batches, ones, marker="o", linestyle="-", color="#777777", label="eager 基准=1.0x")
    plt.plot(batches, compile_speed_a, marker="o", label=f"{args.label_a} compile/eager")
    plt.plot(batches, fused_speed_a, marker="o", label=f"{args.label_a} fused/eager")
    plt.plot(batches, compile_speed_b, marker="s", linestyle="--", label=f"{args.label_b} compile/eager")
    plt.plot(batches, fused_speed_b, marker="s", linestyle="--", label=f"{args.label_b} fused/eager")
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log", base=2)
    plt.xlabel("Batch 大小")
    plt.ylabel("相对 eager 的加速比 (x)")
    plt.title("Eager / Compile / Fused 对比")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_ecf_compare.png", dpi=170)
    plt.close()

    # Plot 3: single-metric bar chart (geo-mean speedup vs eager).
    bar_labels = [
        f"{args.label_a} fused",
        f"{args.label_b} fused",
        f"{args.label_a} fused_graph",
        f"{args.label_b} fused_graph",
    ]
    bar_values = [
        _gmean(fused_speed_a),
        _gmean(fused_speed_b),
        _gmean(fused_graph_speed_a),
        _gmean(fused_graph_speed_b),
    ]

    plt.figure(figsize=(10, 6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    bars = plt.bar(bar_labels, bar_values, color=colors)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("相对 Eager 的几何平均加速比 (x)")
    plt.title("单指标条状图：几何平均加速比")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, bar_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"{v:.3f}x", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_geomean_speedup.png", dpi=170)
    plt.close()

    _write_markdown(
        out_path=out_dir / "analysis.md",
        batches=batches,
        label_a=args.label_a,
        label_b=args.label_b,
        fused_speed_a=fused_speed_a,
        fused_speed_b=fused_speed_b,
        fused_graph_speed_a=fused_graph_speed_a,
        fused_graph_speed_b=fused_graph_speed_b,
    )
    _write_plot_guide(
        out_path=out_dir / "plot_guide.md",
        batches=batches,
        label_a=args.label_a,
        label_b=args.label_b,
        fused_speed_a=fused_speed_a,
        fused_speed_b=fused_speed_b,
        fused_graph_speed_a=fused_graph_speed_a,
        fused_graph_speed_b=fused_graph_speed_b,
    )
    _write_bar_metric_explanation(
        out_path=out_dir / "bar_metric_explanation.md",
        label_a=args.label_a,
        label_b=args.label_b,
        fused_speed_a=fused_speed_a,
        fused_speed_b=fused_speed_b,
        fused_graph_speed_a=fused_graph_speed_a,
        fused_graph_speed_b=fused_graph_speed_b,
    )
    _write_ecf_markdown(
        out_path=out_dir / "ecf_compare.md",
        label_a=args.label_a,
        label_b=args.label_b,
        compile_speed_a=compile_speed_a,
        compile_speed_b=compile_speed_b,
        fused_speed_a=fused_speed_a,
        fused_speed_b=fused_speed_b,
    )

    print(f"Saved: {out_dir / 'latency_eager_fused.png'}")
    print(f"Saved: {out_dir / 'speedup_compare.png'}")
    print(f"Saved: {out_dir / 'speedup_ecf_compare.png'}")
    print(f"Saved: {out_dir / 'bar_geomean_speedup.png'}")
    print(f"Saved: {out_dir / 'analysis.md'}")
    print(f"Saved: {out_dir / 'plot_guide.md'}")
    print(f"Saved: {out_dir / 'bar_metric_explanation.md'}")
    print(f"Saved: {out_dir / 'ecf_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
