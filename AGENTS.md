# Chord residual

用户提到“默认 Chord”时，新训练使用 [chord.args](research/connections/ngpt/chord.args)，跨仓库迁移沿用同一组参数。初速度只查独立的 token 速度表，再投到切平面。每个子层使用球面前瞻、两个独立正步长、逐子层可学习 β 和终点切向投影。跨层 h、v 以 BF16 存储，几何运算用 FP32。开启前瞻读取与 RMSNorm 融合，关闭 force 重算。初始化、训练参照和验证入口见 [Chord 迁移说明](research/connections/ngpt/CHORD.md)。
