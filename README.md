# LeNet 执行性能基准（Eager vs Compile vs 自定义 CUDA）

本项目在同一个 LeNet 模型上，对比三种（扩展为四种）推理执行路径：

1. PyTorch Eager 模式（基线）
2. PyTorch `torch.compile` 模式
3. 自定义 CUDA 融合算子模式（融合第一块：Conv + ReLU + Pool）
4. CUDA Graph 重放模式（`*_graph`），用于减少 launch / 调度开销

项目目标是通过**可复现的基准测试**，清晰展示：

* kernel launch 开销
* 调度间隙（scheduling gaps）
* 中间内存访问流量（intermediate memory traffic）

---

## 项目结构

* `bench/`：基准测试脚本与模型代码
* `cuda_ext/`：C++ / CUDA 扩展源码
* `tests/`：正确性与基础测试
* `results/`：生成的基准报告
* `setup.py`：CUDA 扩展构建脚本

---

## 环境要求

* Windows + NVIDIA GPU
* 已安装 CUDA Toolkit（`nvcc` 可用）
* Visual Studio Build Tools（包含 C++ 编译器 `cl.exe`）
* Python 3.10+（推荐）
* 支持 CUDA 的 PyTorch

---

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 运行环境检查

```powershell
python -m bench.env_check
```

---

### 3. 构建 CUDA 扩展（可选但推荐）

```powershell
python setup.py build_ext --inplace
```

如果你的 PyTorch CUDA 版本与本地 CUDA Toolkit 不一致，可以强制尝试构建：

```powershell
$env:ALLOW_CUDA_MISMATCH='1'; python setup.py build_ext --inplace
```

---

### 4. 运行基准测试

```powershell
python -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100
```

### 一键快速启动（1 到 1024 + 画图分析）

最简命令（推荐）：

```powershell
.\run_full_experiment.cmd
```

等价命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\quick_start_1to1024.ps1
```

可选：先重编译扩展再跑

```powershell
.\run_full_experiment.cmd -RebuildExtension
```

等价命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\quick_start_1to1024.ps1 -RebuildExtension
```

该脚本会自动执行两组实验（`fuse_second_block` 开/关），并在 `results/analysis_<tag>_1to1024` 下输出：

* `latency_eager_fused.png`
* `speedup_compare.png`
* `analysis.md`

---

### 独立画图分析（已有两个 summary.json 时）

```powershell
python -m bench.plot_analysis --run-a results/full_experiment_20260405_1to1024_fuse2_on/summary.json --run-b results/full_experiment_20260405_1to1024_fuse2_off/summary.json --label-a fuse2_on --label-b fuse2_off --out-dir results/analysis_20260405_1to1024
```

---

### 小 batch 延迟分析（推荐，包含 CUDA Graph）

```powershell
python -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100 --enable-cudagraph
```

---

### 融合策略控制

默认情况下会融合两个卷积块。

关闭第二块融合（用于消融实验）：

```powershell
python -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100 --no-fuse-second-block
```

同时开启 CUDA Graph + 第二块融合：

```powershell
python -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100 --enable-cudagraph --fuse-second-block
```

---

### 使用指定 Conda 环境运行（示例）

```powershell
C:/Users/fengye/.conda/envs/hamer/python.exe -m bench.benchmark --device cuda --batch-sizes 1,32,128 --warmup 20 --iters 100
```

---

### 5. 运行测试

```powershell
pytest -q
```

---

## 架构概览

该项目采用分层设计，使以下部分解耦：

* 模型逻辑
* 后端执行方式
* kernel 实现
* 基准调度
* 报告生成

---

### 执行流程

1. 解析 benchmark 参数，确定 device / dtype
2. 构建模型变体（`eager`、`compile`、`fused`、可选 `*_graph`）
3. 与 eager 基线进行正确性校验
4. 对每种模式与 batch size 执行计时循环
5. 导出 JSON 和 Markdown 报告

---

### 核心设计原则

* **渐进式回退（progressive fallback）**

  * 优先使用 `torch.compile`
  * 失败则回退到 `torch.jit.script`
  * 最终保证 eager 可执行

* **安全的融合路径**

  * 优先使用自定义 CUDA 扩展
  * 不可用时自动退化为等价 PyTorch 实现

* **正交优化（orthogonal optimization）**

  * CUDA Graph 是独立层（`*_graph`）
  * 可对所有执行模式统一对比

* **可复现性优先**

  * 共享初始权重
  * 固定随机种子
  * warmup + 同步计时
  * 输出包含构建与运行状态元数据

---

### 详细设计文档

* `docx/system_architecture.md`

---

## 输出结果

* `results/summary.json`：机器可读结果
* `results/summary.md`：Markdown 汇总表

---

## 注意事项

* 所有模式使用相同初始权重，确保公平对比

* 使用 warmup + 同步计时，避免 GPU 异步执行导致误差

* 若 `torch.compile` 因环境限制失败：

  * 会记录失败原因
  * 并回退到 eager 模式

* 在 Windows + 较旧 PyTorch（如 2.0.x）上：

  * `torch.compile` 可能不可用
  * 自动回退到 `torch.jit.script`

* 若 CUDA 扩展不可用：

  * fused 模式退化为 PyTorch 实现
  * 并在结果中标明状态

* `*_graph` 模式要求：

  * CUDA 环境
  * 固定输入 shape（本 benchmark 已满足）

---

如果你需要，我可以帮你把这个项目说明**优化成论文风格 / README（更偏开源项目）/ 或加上性能分析解读模板**。
