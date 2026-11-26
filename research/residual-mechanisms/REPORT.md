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

## 实验事实

<!-- sync:begin -->

等待 GPU 预检与正式训练。

<!-- sync:end -->

## 分析、结论与待验证问题

主要判据是 vanilla 组在相同 token 点的 ΔCE 和完整学习曲线，同时报告参数量、理论 FLOPs、吞吐、峰值显存及训练时间。Native baseline 独立列出。待验证哪种机制能在 1B 预算内持续改善 CE，以及改善对应多少训练开销。
