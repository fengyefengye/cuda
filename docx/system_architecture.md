# CUDA Benchmark System Architecture and Design Rationale

## 1. What this system is solving

The project evaluates multiple GPU inference execution strategies on the same LeNet workload and
produces directly comparable latency and throughput reports.

Primary goals:

- Compare execution backends under identical model weights and inputs.
- Isolate algorithm/backend gains from launch/scheduling gains.
- Keep experiments reproducible and diagnosable.

Non-goals:

- Training optimization.
- Dynamic-shape runtime serving at arbitrary shapes.
- General-purpose model zoo support.

## 2. Architectural principles

1. Separation of concerns
- Benchmark orchestration, model construction, CUDA extension loading, kernel code, and report formatting are split into independent modules.

2. Progressive reliability over hard failure
- Missing compile support or extension availability should not crash benchmark runs by default.
- The system records fallback decisions in output metadata.

3. Orthogonal optimization dimensions
- Backend optimization (eager vs compile vs fused) is one axis.
- Scheduling optimization (graph replay vs non-graph) is another axis.
- This allows meaningful A/B comparisons.

4. Reproducibility and observability
- Deterministic seeds and consistent warmup/measurement loops.
- Explicit build and runtime status emitted in `build_info`.

## 3. Layered system view

Layer A: CLI and experiment orchestration
- File: bench/benchmark.py
- Responsibilities:
  - Parse experiment options.
  - Resolve runtime device.
  - Build model variants.
  - Run correctness and timed loops.
  - Emit JSON/Markdown outputs.

Layer B: Variant factory and execution wrappers
- File: bench/models.py
- Responsibilities:
  - Define eager baseline model (`LeNetEager`).
  - Define fused model (`LeNetFused`).
  - Build compile variant and fallbacks.
  - Wrap any variant in CUDA Graph (`CUDAGraphModule`).

Layer C: Extension boundary and runtime loader
- File: bench/cuda_ops.py
- Responsibilities:
  - Import extension and handle Windows DLL paths.
  - Optionally JIT-build extension if configured.
  - Expose fused op call and extension status.

Layer D: Native extension interface
- File: cuda_ext/fused_conv_relu_pool_bindings.cpp
- Responsibilities:
  - Validate tensor properties and shape constraints.
  - Bind Python entrypoint to CUDA kernel launcher.

Layer E: CUDA kernel implementation
- File: cuda_ext/fused_conv_relu_pool.cu
- Responsibilities:
  - Single-kernel Conv + ReLU + Pool forward path.
  - Output tensor generation and launch checks.

Layer F: Build and packaging glue
- File: setup.py
- Responsibilities:
  - Define extension source list and compile flags.
  - Support optional CUDA version mismatch override.

Layer G: Environment checks and profiling helpers
- Files: bench/env_check.py, bench/profile_nsys.py
- Responsibilities:
  - Validate prerequisites (`torch`, CUDA, nvcc, cl).
  - Collect kernel timeline stats with Nsight Systems.

Layer H: Test verification
- Files: tests/test_correctness.py, tests/test_smoke_infer.py
- Responsibilities:
  - Correctness vs eager baseline under tolerance constraints.
  - Smoke coverage for all modes including graph variants.

## 4. End-to-end runtime flow

1. Input options are parsed (modes, batch sizes, warmup/iters, graph switch).
2. `build_model_variants` creates model dictionary:
   - eager
   - compile (or fallback)
   - fused (extension path or fallback)
   - optional graph-wrapped versions
3. Correctness check compares outputs to eager baseline.
4. Benchmark loop runs per mode and batch:
   - warmup
   - synchronized timing
   - latency distribution statistics
5. Export:
   - summary.json (structured)
   - summary.md (table)

## 5. Why modes are designed this way

Base modes:

- eager: pure PyTorch eager baseline.
- compile: graph/compiler optimization path.
- fused: custom operator path reducing kernel count and intermediate memory traffic.

Graph modes (`*_graph`):

- Keep backend semantics unchanged.
- Replace repeated launch orchestration with graph replay.
- Measure launch/scheduling reductions independently from backend differences.

This two-axis composition enables analyses like:

- `compile` vs `eager`: compiler optimization gain.
- `fused` vs `compile`: custom kernel gain.
- `compile_graph` vs `compile`: scheduling overhead gain.
- `fused_graph` vs `fused`: scheduling gain after kernel fusion.

## 6. Fallback strategy and failure containment

Compile fallback chain:

1. Try `torch.compile`.
2. If unavailable or failing, try `torch.jit.script`.
3. If still failing, keep eager-compatible execution.

Fused fallback behavior:

- If extension imports and links correctly, use CUDA fused op.
- Otherwise fallback to equivalent PyTorch ops in model forward.

All outcomes are reflected in `build_info`, so experiment reports remain explainable.

## 7. Measurement design choices

Timing methodology:

- GPU: CUDA events + stream sync.
- CPU fallback: wall-clock timing.
- Warmup before official iteration collection.

Reported metrics:

- mean, median, p95, p99, std, min, max latency.
- throughput based on median latency.
- peak allocated CUDA memory.

Reasoning:

- Median and p95/p99 describe both central tendency and tail stability.
- Peak memory helps identify performance-memory tradeoffs between modes.

## 8. Data contract of outputs

`summary.json` fields:

- `meta`: execution environment and run settings.
- `build_info`: backend/fallback/extension/graph runtime status.
- `correctness`: numeric error vs eager baseline.
- `benchmarks`: per-batch per-mode metric sets.

This contract is stable enough for downstream plotting scripts and regression comparisons.

## 9. Extensibility points

Add a new backend mode:

1. Implement model variant in bench/models.py.
2. Register it in `build_model_variants` dictionary.
3. Optionally add graph wrapping via existing wrapper path.
4. Reuse benchmark and report pipeline without structural changes.

Add a new fused operator:

1. Add C++ binding and CUDA kernel source.
2. Expose loader entry in bench/cuda_ops.py.
3. Integrate call path in model variant.
4. Add correctness and smoke tests.

Add new metrics:

- Extend metric dict in benchmark timing function.
- Extend markdown writer columns.

## 10. Current tradeoffs and future improvements

Current tradeoffs:

- CUDA Graph path assumes fixed shape per captured signature.
- Windows runtime may need specific environment variables for inductor stability.
- Fused kernel currently targets float32 forward path.

Recommended next improvements:

1. Add automated environment preflight and compatibility warnings at benchmark startup.
2. Add optional auto-tuning and more detailed kernel occupancy diagnostics.
3. Extend fused kernels to additional dtypes and model blocks.
4. Add CI-level regression checks against baseline medians/p95.
