# Quick start runner for 1..1024 benchmark plus plotting analysis.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\scripts\quick_start_1to1024.ps1

param(
    [string]$PythonExe = "C:/Users/fengye/.conda/envs/hamer/python.exe",
    [string]$BatchSizes = "1,2,4,8,16,32,64,128,256,512,1024",
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [string]$Tag = "20260405",
    [switch]$RebuildExtension
)

$ErrorActionPreference = 'Stop'

Set-Location "$PSScriptRoot\.."

if ($RebuildExtension) {
    Write-Host "[STEP] Rebuilding CUDA extension in-place..."
    $env:ALLOW_CUDA_MISMATCH = '1'
    & $PythonExe setup.py build_ext --inplace
}

$runOn = "results/full_experiment_${Tag}_1to1024_fuse2_on"
$runOff = "results/full_experiment_${Tag}_1to1024_fuse2_off"
$outDir = "results/analysis_${Tag}_1to1024"

Write-Host "[STEP] Run benchmark (fuse second block ON)..."
& $PythonExe -m bench.benchmark --device cuda --batch-sizes $BatchSizes --warmup $Warmup --iters $Iters --enable-cudagraph --fuse-second-block --results-dir $runOn

Write-Host "[STEP] Run benchmark (fuse second block OFF)..."
& $PythonExe -m bench.benchmark --device cuda --batch-sizes $BatchSizes --warmup $Warmup --iters $Iters --enable-cudagraph --no-fuse-second-block --results-dir $runOff

Write-Host "[STEP] Plot and summarize..."
& $PythonExe -m bench.plot_analysis --run-a "$runOn/summary.json" --run-b "$runOff/summary.json" --label-a "fuse2_on" --label-b "fuse2_off" --out-dir $outDir

Write-Host "[DONE]"
Write-Host "  - Run A: $runOn"
Write-Host "  - Run B: $runOff"
Write-Host "  - Analysis: $outDir"
