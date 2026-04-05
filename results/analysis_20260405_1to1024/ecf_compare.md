# Eager / Compile / Fused 对比

指标: 相对 eager 的几何平均加速比

| 方案 | 加速比 |
|---|---:|
| fuse2_on eager | 1.000x |
| fuse2_on compile | 1.113x |
| fuse2_on fused | 1.244x |
| fuse2_off eager | 1.000x |
| fuse2_off compile | 1.084x |
| fuse2_off fused | 1.206x |
