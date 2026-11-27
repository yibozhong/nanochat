# Sequential Backward（已停止）

<!-- sync:summary-only -->

**实测结论** 在已完成的训练区间，最后三个相同 token 数的验证点，Sequential Backward 的 FineWeb-Edu 验证 CE 均高于 vanilla。第 5,000 步（163,840,000 tokens）两者分别为 3.818493 和 3.784088，差值 +0.034405。

seed 42，vanilla 基线 `a2793a7e`。同卡 H200 短测中，顺序更新约 30.2 万 tokens/s，vanilla 约 50.6 万 tokens/s。按相同口径估算，顺序更新每步的矩阵 FLOPs 增加 45.84%。

2026-11-27 按用户指令提前终止。最后记录的训练步为 5,250，累计 172,032,000 tokens。原定预算为 1B tokens。[W&B 训练记录](https://wandb.ai/yibozhong657-none/nanochat-1b/runs/bcc7c95f)

<!-- sync:begin -->
<!-- sync:end -->
