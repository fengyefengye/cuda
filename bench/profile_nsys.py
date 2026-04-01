"""Optional helper to collect per-mode nsys traces for kernel-level comparison."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nsys profiling for benchmark modes")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--modes", type=str, default="eager,compile,fused")
    parser.add_argument("--out-dir", type=str, default="results/nsys")
    return parser.parse_args()


def _run(cmd):
    print(" ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def main() -> int:
    args = parse_args()

    nsys_bin = shutil.which("nsys")
    if not nsys_bin:
        print("nsys is not found in PATH. Install Nsight Systems or add it to PATH.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        out_prefix = out_dir / mode
        profile_cmd = [
            nsys_bin,
            "profile",
            "--force-overwrite=true",
            "--sample=none",
            "--trace=cuda,nvtx,osrt",
            "-o",
            str(out_prefix),
            args.python,
            "-m",
            "bench.benchmark",
            "--device",
            "cuda",
            "--batch-sizes",
            str(args.batch_size),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--modes",
            mode,
            "--results-dir",
            str(out_dir / "bench_results" / mode),
        ]

        prof = _run(profile_cmd)
        print(prof.stdout)
        if prof.returncode != 0:
            print(prof.stderr)
            print(f"profile failed for mode={mode}")
            continue

        rep_path = str(out_prefix) + ".nsys-rep"
        stats_cmd = [nsys_bin, "stats", rep_path]
        stats = _run(stats_cmd)

        stats_file = out_dir / f"{mode}_stats.txt"
        stats_file.write_text((stats.stdout or "") + "\n" + (stats.stderr or ""), encoding="utf-8")
        print(f"saved stats to: {stats_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
