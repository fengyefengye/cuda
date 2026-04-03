# Triton compile benchmark runner for cuda_bench_new on Windows
# Usage: powershell -ExecutionPolicy Bypass -File .\docx\run_triton_compile_benchmark.ps1

$ErrorActionPreference = 'Stop'

& C:/ProgramData/anaconda3/shell/condabin/conda-hook.ps1
conda activate cuda_bench_new

# Workaround for WinError 183 in torch inductor pad_mm cache path on Windows.
$env:TORCHINDUCTOR_SHAPE_PADDING = '0'

Write-Host "[INFO] Python:" (python -c "import sys; print(sys.version)")
Write-Host "[INFO] Torch/Triton:" (python -c "import torch, triton; print('torch='+torch.__version__+', triton='+triton.__version__)")

python -m bench.benchmark --device cuda --batch-sizes 1 --warmup 1 --iters 3 --modes compile --results-dir results/triton_ready_correctness

Write-Host "[DONE] Results written to results/triton_ready_correctness"
