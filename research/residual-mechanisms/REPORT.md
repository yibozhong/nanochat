# 零输出初始化下，哪些 residual 机制改善 1B 预训练？

## 假设与实验设计

在 vanilla 主干上，分别测试历史状态读取、跨流混合和球面前瞻更新能否降低相同 tokens 下的 validation CE。四种方法共用同一个 vanilla baseline。另跑一个原生 nanochat baseline，记录带 SSSL、VE 等原生设计时的水平。

| 顺序 | 主干与 residual | 实验配置 |
| --- | --- | --- |
| 1 | vanilla baseline | 标准 residual 加法，全注意力 |
| 2 | native baseline | 原生 GPT，SSSL、VE、smear、x0、backout |
| 3 | vanilla + AttnRes | Block AttnRes，S=2 个子层 |
| 4 | vanilla + mHC | 4 条流，20 次 Sinkhorn |
| 5 | vanilla + Chord Next BF16 | 以 soft force 为位移，token-dependent 控制，从 h0 生成初速度 |

每条从零开始，seed 42，8 层、宽 512、上下文 2048，使用相同 packed FineWeb-Edu 数据顺序和 30,518 步学习率日程。每层 Attention 输出和 FFN down 投影均从零开始。六系数 Chord 使用 soft-force τ=0.05。原顺序作业逐配置验证首批梯度与断点恢复。

历史四标量配方和旧消融已移入 [Legacy 归档](../legacy-chord-1b/REPORT.md)。当前两组分别比较[三种初速度](../chord-initial-velocity/REPORT.md)和[球面与速度前瞻](../chord-components/REPORT.md)。

## 实验事实

<!-- sync:begin -->

| 配置 | 状态 | 验证 tokens | 最近 CE | ΔCE | 基线 |
|---|---|---:|---:|---:|---|
| [chord-next-bf16-zero-s42-1b](configs/2e37a95d.json) | 已完成 | 1000013824 | 3.19696 | -0.09569 | a2793a7e |
| [native-baseline-zero-s42-1b](configs/7e37ab98.json) | 已完成 | 1000013824 | 3.16445 | 0.00000 | self |
| [vanilla-attnres-zero-s42-1b](configs/d38f2bd0.json) | 已完成 | 1000013824 | 3.20451 | -0.08814 | a2793a7e |
| [vanilla-baseline-zero-s42-1b](configs/a2793a7e.json) | 已完成 | 1000013824 | 3.29265 | 0.00000 | self |
| [vanilla-mhc-zero-s42-1b](configs/b4e0525d.json) | 已完成 | 1000013824 | 3.23794 | -0.05472 | a2793a7e |

![08e694cd0517647a validation CE](curves-08e694cd0517647a.png)

![09c92a717b71cb92 validation CE](curves-09c92a717b71cb92.png)

[完整验证曲线数据](curves.csv)

<!-- sync:end -->

## 分析、结论与待验证问题

主要判据是 vanilla 组在相同 token 点的 ΔCE 和完整学习曲线，同时报告参数量、理论 FLOPs、吞吐、峰值显存及训练时间。Native baseline 独立列出。待验证哪种机制能在 1B 预算内持续改善 CE，以及改善对应多少训练开销。
