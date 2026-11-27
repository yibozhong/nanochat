# 查表 Chord 的六个系数需要按 token 调节吗？

## 假设与设计

在同一模型和初速度下，比较每个子层一组可学习标量（static）和根据当前 token 的 h 调节这组系数（token-dependent）。假设是 token 调制能降低验证 CE。static 仍学习前瞻 μ、两个位置步长 α、速度保留 β 和两个速度写入增益 γ。两组都保留球面位置、切面投影、独立 token 速度表和 BF16 状态。

配对基线为已完成的 token-dependent 查表版，最终 validation CE 为 3.199604。两组均用 vanilla 8 层、宽 512、2K 上下文，Attention 和 FFN 输出投影零初始化。从零、seed 42，以相同 FineWeb-Edu 数据顺序和学习率日程训练 30,518 步，共 1,000,013,824 tokens。[基线 7337b20d](https://wandb.ai/yibozhong657-none/nanochat-1b/runs/7337b20d)

## 判据与结果

ΔCE 取 static 减 token-dependent，正值表示 token 调制更好。对齐整条验证曲线，报告终点 CE 与最后 1 亿 tokens 共同验证点的平均差值，同时记录移除 token 调制后节省的参数和计算。

<!-- sync:begin -->
<!-- sync:end -->
