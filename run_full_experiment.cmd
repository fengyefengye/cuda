@echo off
setlocal

if /I "%~1"=="/?" goto usage
if /I "%~1"=="-h" goto usage
if /I "%~1"=="--help" goto usage

powershell -ExecutionPolicy Bypass -File .\scripts\quick_start_1to1024.ps1 %*
endlocal
goto :eof

:usage
echo One-command full benchmark runner.
echo.
echo Usage:
echo   .\run_full_experiment.cmd
echo   .\run_full_experiment.cmd -RebuildExtension
echo.
echo This runs 1..1024 benchmark twice ^(fuse2 on/off^) and generates plots.
endlocal
