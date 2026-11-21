# Chord residual

用户提到“默认 Chord”时，采用球面前瞻、两个独立正步长、逐子层可学习 β，以及只做切向投影的速度。新实验和跨仓库迁移都应显式选择 `--chord-velocity-geometry project_only --chord-state-dtype bfloat16`。跨层 h、v 以 BF16 存储，归约及写回内部使用 FP32。完整配方、更新顺序和验证入口见 [Chord 迁移说明](research/connections/ngpt/CHORD.md)。
