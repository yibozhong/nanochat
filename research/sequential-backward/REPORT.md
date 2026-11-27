# Sequential Backward

检验一个问题：每层更新后再计算传给前一层的输入梯度，能否用更少的 FineWeb-Edu tokens 达到相同验证 CE？

配对基线为 vanilla run `a2793a7e`。使用相同的 58,720,256 参数、全注意力、零初始化残差输出投影、seed 42、packed FineWeb-Edu 数据与顺序。每步 32,768 tokens，完整预算 30,518 步。基线最终验证 CE 为 3.292653，测试 CE 为 3.279382。[基线配置](../residual-mechanisms/configs/a2793a7e.json)

保留初次前向时的每层输入。LM head 按原参数计算交叉熵、头部参数梯度和最顶层传入梯度。每个 Transformer block 从后向前收齐整个有效 batch 的参数梯度，执行一次原 Muon 更新，再用相同的输入和传入梯度，通过更新后的 block 重算输入梯度。输入 embedding 收到第一个 block 重算的梯度。两个独立的端点各执行一次原 AdamW 更新。全局学习率、动量和权重衰减日程每个训练步前进一次。跨层共享存储会在创建优化器时拒绝，防止顺序更新更改尚待反传的参数。

| 比较 | 观测口径 |
| --- | --- |
| 数据效率 | 对齐训练 tokens，比较固定 FineWeb-Edu dev 集的 CE 和完整曲线。 |
| 计算效率 | vanilla 完整预算对应 Sequential Backward 的第 20,925 个整步；额外评估同一 dev 集，保持两组原有 LR 日程与 best checkpoint 的选择频率。 |
| 速度 | 同一次 GPU 分配，两个方法都从相同参数训练 50 步，排除前 10 步，以剩余 tokens 除以同步训练耗时。不计验证、编译预热与 checkpoint 写盘。 |

默认 microbatch 16 时，vanilla 每步估算 9.8964 TFLOPs，Sequential Backward 每步 14.4327 TFLOPs，增加 45.84%。按仓库口径，乘加计两次，注意力仅计因果位置对。新增工作包括更新后的 block 前向和输入梯度；旧参数梯度的计算省去 Q/K/V 对 block 输入的梯度。拆成多个 microbatch 或启用激活检查点时，另计为各 microbatch 重算原始 block 前向的开销。上述估算排除优化器、逐元素运算和内核内部重算。

实际运行结果由训练归档与曲线同步器记录。预检生成 `results/sequential-comparison.json`，正式训练在基线 FLOP 预算处生成 `results/train/equal-compute.json`。数据效率与计算效率分别以对应预算的验证 CE 判断。
