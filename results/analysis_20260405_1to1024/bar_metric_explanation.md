# 条状图指标说明

- 选用指标: 相对 eager 的几何平均加速比（基于每个 batch 的中位延迟）。
- 选择原因: 能公平汇总全 batch 区间，且不容易被个别异常值带偏。
- 读法: 大于 1.0x 表示快于 eager，越高越好。

## 数值
- fuse2_on fused: 1.244x
- fuse2_off fused: 1.206x
- fuse2_on fused_graph: 5.336x
- fuse2_off fused_graph: 2.666x
