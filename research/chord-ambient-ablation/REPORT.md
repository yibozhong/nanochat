# Chord 查表时是否需要切面投影？从静止开始时前瞻贡献多少？

## 假设与设计

查表实验比较速度的几何约束。两边都有独立 token 速度表、可学习的前瞻系数 μ 和 75,545,228 个参数。旧版在初始化和逐层速度写入时都做切面投影；ambient 版直接使用查到的初速度，逐层保留 ambient 速度。这个对照含两处投影的合计影响。[切面查表基线 7337b20d](https://wandb.ai/yibozhong657-none/nanochat-1b/runs/7337b20d)

固定 μ=0 的实验单独检验前瞻。该配对的两边都从零初速度开始，使用 ambient 速度；基线按 token 学习 μ，消融把 μ 固定为零。[零初速、可学 μ 基线 011078b9](https://wandb.ai/yibozhong657-none/nanochat-1b/runs/011078b9)

两组均为 vanilla 8 层 × 512、2K 上下文、BF16、球面位置，Attention 和 FFN 输出投影零初始化。从零、seed 42 训练 30,518 步。ΔCE 取相同 token 数下实验组与各自基线的 validation CE 之差。

## 实验结果

<!-- sync:begin -->
待正式验证。
<!-- sync:end -->
