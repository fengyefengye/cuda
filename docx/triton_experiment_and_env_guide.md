Triton 实验过程总结与环境配置指南

文档目的
- 记录本次在 Windows 上为项目启用 Triton 的完整实验过程。
- 说明每一步失败或成功的原因。
- 给出可复现的最终环境配置与运行命令。
- 提供常见报错到处理动作的对照表。

一、实验背景
- 项目路径: e:/cuda
- 目标: 让 bench.benchmark 的 compile 路径稳定使用 torch.compile，而不是回退。
- 关键约束: Windows + CUDA + PyTorch 2.5.1+cu121

二、关键环境信息
- Python 环境: cuda_bench_new
- PyTorch 版本: 2.5.1+cu121
- Torch CUDA 版本: 12.1
- NumPy 版本: 2.4.4
- nvcc 版本: 12.6
- CUDA 可用状态: True

三、实验过程时间线
1) 初始状态检查
- 检查 triton 与 pytorch-triton 包均未安装。
- 结论: compile 报 Cannot find a working triton installation 属于预期。

2) 直接安装 triton 失败
- 在默认镜像和官方 PyPI 上安装 triton 失败，无匹配 wheel。
- 结论: Windows 上官方 triton wheel 不可直接使用。

3) 安装 triton-windows 3.6.0.post26
- triton 可 import，但 torch.compile 报错:
  ImportError: cannot import name triton_key
- 结论: triton-windows 3.6 与 torch 2.5.1 的 API 期望不匹配。

4) 降级到 triton-windows 3.1.0.post17
- 最小 torch.compile 烟测通过，能输出张量形状。
- 结论: torch 2.5.1 与 triton-windows 3.1.0.post17 兼容性更好。

5) 项目级 benchmark 仍出现 Windows 缓存报错
- 报错: FileExistsError [WinError 183] 来自 torch inductor codecache 写入。
- 触发位置: pad_mm 相关 autoheuristic 缓存写入。
- 结论: 这是 Windows 上 inductor 缓存路径的并发/原子重命名问题，不是 Triton 不可用。

6) 关闭 shape padding 后可跑通
- 设置环境变量 TORCHINDUCTOR_SHAPE_PADDING=0。
- compile 路径 benchmark 可完成，结果文件可生成。
- 结论: 当前机器上的可行运行方案成立。

四、最终可复现配置
1) 安装依赖
命令: (C:/ProgramData/anaconda3/shell/condabin/conda-hook.ps1); conda activate cuda_bench_new
命令: python -m pip install --force-reinstall triton-windows==3.1.0.post17

2) 运行前环境变量
命令: $env:TORCHINDUCTOR_SHAPE_PADDING='0'

3) 最小 compile 烟测
命令: python -c "import triton, torch; print('triton='+triton.__version__); m=torch.nn.Sequential(torch.nn.Linear(16,16), torch.nn.ReLU(), torch.nn.Linear(16,16)).cuda().eval(); x=torch.randn(32,16,device='cuda'); y=torch.compile(m)(x); print('ok', tuple(y.shape))"

4) 项目 benchmark 验证
命令: python -m bench.benchmark --device cuda --batch-sizes 1 --warmup 1 --iters 3 --modes compile --results-dir results/triton_ready_correctness

五、已验证结果文件
- 仅 compile 验证结果: results/triton_ready_correctness/summary.md
- eager+compile 烟测结果: results/triton_ready_smoke/summary.md

六、结果解读
- build_info.compile_backend_used 为 torch.compile，说明已进入 compile 路径。
- correctness 内 compile 误差在阈值内，说明数值可接受。
- 小样本性能会受 warmup 与图编译摊销影响，建议用更长 iters 做稳定评估。

七、常见报错与处理
1) Cannot find a working triton installation
- 原因: 环境中没有可用 Triton。
- 处理: 安装 triton-windows 并使用与 torch 匹配版本。

2) cannot import name triton_key
- 原因: Triton 版本过新，与 torch 内部接口不匹配。
- 处理: 固定到 triton-windows==3.1.0.post17。

3) FileExistsError WinError 183 in torchinductor cache
- 原因: Windows 下 inductor 某些缓存写入冲突。
- 处理: 运行前设置 TORCHINDUCTOR_SHAPE_PADDING=0。

八、推荐固化方式
- 方式 A: 每次运行前手动设置环境变量。
- 方式 B: 使用项目脚本统一设置，再执行 benchmark。
- 方式 C: 在 CI 或任务脚本里固定 triton 版本与变量，避免环境漂移。

九、后续优化建议
- 在 benchmark.py 内增加 Windows 检测并自动设置 shape padding 关闭开关。
- 增加启动时版本自检，输出 torch 与 triton 兼容提示。
- 补充 nsys 时间线，确认 compile 下 kernel 启动密度与空白区变化。
