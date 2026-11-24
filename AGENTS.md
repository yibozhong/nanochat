# Chord residual

用户提到“默认 Chord”时，新训练使用 [chord.args](research/connections/ngpt/chord.args)，跨仓库迁移沿用同一组参数。初速度只查独立的 token 速度表，再投到切平面。每个子层使用球面前瞻和终点切向投影。前瞻系数 μ、两个独立正步长 α∥、α⊥ 和可学习 β 默认都是逐子层标量，在通道与 token 之间共享。跨层 h、v 以 BF16 存储，几何运算用 FP32。开启前瞻读取与 RMSNorm 融合，关闭 force 重算。初始化、训练参照和验证入口见 [Chord 迁移说明](research/connections/ngpt/CHORD.md)。

移植外部方法时，训练超参数默认沿用 nanochat 对应的配对基线。native Chord 的 token gate 投影与偏置沿用本仓库 native mHC 路由参数的 AdamW 配方和学习率缩放，除非实验明确指定覆盖项。
