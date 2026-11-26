# 零输出初始化下，哪些 residual 机制改善 1B 预训练？

## 假设与实验设计

在 vanilla 主干上，分别测试历史状态读取、跨流混合和球面前瞻更新能否降低相同 tokens 下的 validation CE。四种方法共用同一个 vanilla baseline。另跑一个原生 nanochat baseline，记录带 SSSL、VE 等原生设计时的水平。

| 顺序 | 主干与 residual | 实验配置 |
| --- | --- | --- |
| 1 | vanilla baseline | 标准 residual 加法，全注意力 |
| 2 | native baseline | 原生 GPT，SSSL、VE、smear、x0、backout |
| 3 | vanilla + AttnRes | Block AttnRes，S=2 个子层 |
| 4 | vanilla + mHC | 4 条流，20 次 Sinkhorn |
| 5 | vanilla + Chord Scalar BF16 | 以 soft force 为目标，双步长，独立 token 速度表 |
| 6 | vanilla + Chord Next BF16 | 以 soft force 为位移，token-dependent 控制，从 h0 生成初速度 |

每条从零开始，seed 42，8 层、宽 512、上下文 2048，使用相同 packed FineWeb-Edu 数据顺序和 30,518 步学习率日程。每层 Attention 输出和 FFN down 投影均从零开始。两条 Chord 的 soft-force τ 均为 0.05。

同卡先验证六套配置的首批梯度、真实 NTP 更新及断点恢复，再依次正式训练。首步逐层比较两种 Chord 与 vanilla 的输出投影梯度范数。Scalar 与 Next 同时改变速度来源和读写规则，两者之差反映整套配方的效果。

## Scalar Chord 从静止出发的分量消融

沿用 vanilla 1B 协议、seed 42、BF16 和零输出投影，三条均从头训练。起点采用 Next 静态标量版的 v0=0。两条含速度的配置直接创建零张量；三条均无速度表或初速度投影。主干和学习率沿用 nanochat，球面分支使用当前 soft-force τ=0.05。[Next 标量版公式](https://github.com/lululu39/NextTransformer/blob/chordres-zero/nexttransformer/methods/chordres_zero/README.md)

| 配置 | 分支读取和位置更新 | 速度更新 |
| --- | --- | --- |
| 完整 Scalar，从静止出发 | 前瞻读取；沿速度、垂直速度两个标量步长，朝 soft-force 目标更新并回到单位 RMS 球面 | 用 β 混合旧速度与实际位移，再投影到终点切平面 |
| 仅球面写入 | 从 h 读取；保留一个正标量步长 α，更新为 N(h + α × (soft(F) − h)) | 无速度状态 |
| 仅速度前瞻 | 从 h + μv 读取并做原有 pre-norm；用 h′ = h + F 更新位置 | v′ = βv + (1 − β)(h′ − h)，在欧氏空间记录实际位移 |

有前瞻的子层均使用可学习标量 μ（初值 0）；有后续读取的速度使用可学习 β（初值 0.5）。球面写入的步长初值为 0.05。完整从静止版本和原 vanilla 基线 a2793a7e 配对；两条消融直接对比本次完整从静止版本。结合四组同 token 的 CE 可判断两个分量各自的收益，以及组合是否存在交互。

## 实验事实

<!-- sync:begin -->

| 配置 | 状态 | 验证 tokens | 最近 CE | ΔCE | 基线 |
|---|---|---:|---:|---:|---|
| [chord-next-bf16-zero-s42-1b](configs/2e37a95d.json) | 排队 | — | — | — | a2793a7e |
| [chord-scalar-bf16-zero-s42-1b](configs/57ba73d6.json) | 排队 | — | — | — | a2793a7e |
| [native-baseline-zero-s42-1b](configs/7e37ab98.json) | 排队 | — | — | — | self |
| [vanilla-attnres-zero-s42-1b](configs/d38f2bd0.json) | 排队 | — | — | — | a2793a7e |
| [vanilla-baseline-zero-s42-1b](configs/a2793a7e.json) | 排队 | — | — | — | self |
| [vanilla-mhc-zero-s42-1b](configs/b4e0525d.json) | 排队 | — | — | — | a2793a7e |

[完整验证曲线数据](curves.csv)

<!-- sync:end -->

## 分析、结论与待验证问题

主要判据是 vanilla 组在相同 token 点的 ΔCE 和完整学习曲线，同时报告参数量、理论 FLOPs、吞吐、峰值显存及训练时间。Native baseline 独立列出。待验证哪种机制能在 1B 预算内持续改善 CE，以及改善对应多少训练开销。
