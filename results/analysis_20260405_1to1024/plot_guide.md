# 看图说明

下面用最直白的方式解释图的含义。

## 1) latency_eager_fused.png

- 横轴是 batch 大小，纵轴是延迟（毫秒）。
- 线越低越快。
- 线发生交叉，表示不同 batch 下最优方法不同。

## 2) speedup_compare.png

- 横轴是 batch 大小，纵轴是加速比。
- 大于 1.0x 表示比 eager 快，小于 1.0x 表示比 eager 慢。
- 线越高越好。

## 3) speedup_ecf_compare.png（重点）

- 这张图只比较 eager / compile / fused 三类，最适合做主结论。
- eager 固定为 1.0x 基准线。
- compile 和 fused 越高越好。

## 快速阅读步骤

- 先看 speedup_ecf_compare.png 选方案。
- 再看 latency_eager_fused.png 确认毫秒值。

## 自动统计

- fuse2_on fused 快于 eager 的 batch 点: 8/11
- fuse2_off fused 快于 eager 的 batch 点: 9/11
- fuse2_on fused_graph 快于 eager 的 batch 点: 11/11
- fuse2_off fused_graph 快于 eager 的 batch 点: 11/11
