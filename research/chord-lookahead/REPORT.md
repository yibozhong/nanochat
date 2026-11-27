# 保留切向速度时，前瞻读取还能降低多少 CE？

## 假设与实验设计

默认 Chord 用速度 v 计算 Attention 和 FFN 的输入，也用 v 分解建议位移并更新速度。这个实验只关闭第一种用途。若前瞻读取有独立贡献，固定 μ=0 后的 validation CE 应高于完整 Chord。

| 零初速配置 | 子层读取 | 位置与速度更新 |
|---|---|---|
| 完整 Chord（aa44a2d7） | 当前 token 决定 μ，读取归一化的 h + μv | 球面位置、双步长、切向速度和两个独立写入增益 |
| 固定 μ=0 | 每个 Attention 和 FFN 均读取经相同 pre-norm 的 h | 同上 |

两边使用 vanilla 8×512、v₀=0、BF16、soft force τ=0.05、零输出投影、同一份 FineWeb-Edu、相同 LR 日程与 seed 42，从零训练 30,518 × 32,768 tokens。消融在融合读取前将 μ 的偏置和 token 调制同时置零，保留完全相同的参数布局和优化器分组。验证记录各子层 μ 的均值与标准差。

按相同训练 tokens 比较 validation CE 曲线。终点 ΔCE 定义为固定 μ=0 的 CE 减去已完成的零初速默认版 CE（3.207224）。已有“仅球面”实验还移除了速度状态，两者可辅助判断读取和写入分别贡献了什么。[配对基线](../chord-initial-velocity/configs/aa44a2d7.json) · [组件消融](../chord-components/REPORT.md)

## 实验事实

<!-- sync:begin -->

等待正式训练指标。

<!-- sync:end -->

## 分析与结论

前瞻读取的条件收益待测。
