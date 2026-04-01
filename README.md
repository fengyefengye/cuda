# LeNet Execution Benchmark (Eager vs Compile vs Custom CUDA)

This project compares three inference execution paths on the same LeNet model:

1. PyTorch eager mode (baseline)
2. PyTorch `torch.compile` mode
3. Custom CUDA fused op mode (fused first block: Conv + ReLU + Pool)

The purpose is to make the launch-overhead and intermediate-memory traffic differences visible with a reproducible benchmark.

## Project layout

- `bench/`: benchmark scripts and model code
- `cuda_ext/`: C++/CUDA extension source
- `tests/`: correctness and smoke tests
- `results/`: generated benchmark reports
- `setup.py`: build script for CUDA extension

## Requirements

- Windows + NVIDIA GPU
- CUDA Toolkit installed (`nvcc` available)
- Visual Studio Build Tools with C++ toolchain (`cl.exe` available)
- Python 3.10+ recommended
- PyTorch with CUDA support

## Quick start

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run environment check:

```powershell
python -m bench.env_check
```

3. Build CUDA extension (optional but recommended for fused mode):

```powershell
python setup.py build_ext --inplace
```

If your PyTorch wheel CUDA version differs from local CUDA toolkit version,
you can explicitly opt in to build attempt:

```powershell
$env:ALLOW_CUDA_MISMATCH='1'; python setup.py build_ext --inplace
```

4. Run benchmark:

```powershell
python -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100
```

If you want to run with a specific conda environment (example: `hamer`):

```powershell
C:/Users/fengye/.conda/envs/hamer/python.exe -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100
```

5. Run tests:

```powershell
pytest -q
```

## Outputs

- `results/summary.json`: machine-readable benchmark result
- `results/summary.md`: markdown summary table

## Notes

- All modes share the same initial weights for fair comparison.
- Benchmark uses warmup + synchronized timing to avoid undercounting async GPU execution.
- If `torch.compile` fails due environment constraints, the script records the failure reason and falls back to eager execution for that path.
- On Windows with older PyTorch versions (for example 2.0.x), `torch.compile` may be unsupported. The benchmark then falls back to `torch.jit.script` for graph-mode comparison.
- If custom CUDA extension is unavailable, fused model falls back to an equivalent PyTorch implementation and reports extension status.
