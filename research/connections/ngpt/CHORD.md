# Chord residual 配方

新实验默认使用 token-dependent 六系数 Chord。每个 token 跨层携带位置 h 和切向速度 v。位置的 RMS 为 1，球面半径为 sqrt(d_model)。Attention 和 FFN 都读取前瞻点，在球面上更新位置，再把这一步的位移写入速度。h、v 以 BF16 保存，几何归约使用 FP32。

## 两个入口，三种初速度

| 控制方式 | 1B 提交参数 | 现有参数文件的初速度 |
| --- | --- | --- |
| Token-dependent（默认），每个子层按当前 token 的 h 调制系数 | 省略 `--recipe` 或指定 `--recipe chord_next_bf16` | 独立 token 速度表 |
| Static（可选消融），每个子层一组跨 token 共享的系数 | `--recipe chord_scalar_bf16` | 独立速度表 |

两个入口都可用 `--velocity-init` 指定初速度。比较两种控制方式时，两侧使用相同的初速度。[Token-dependent 配方](chord.args) · [Static 配方](chord_scalar.args)

- `token` 是默认方式。查独立 token 速度表，v0 = P(h0, E_v[id])，表按标准差 0.02 初始化。
- `projected_h0` 是第二种方式。取 v0 = P(h0, W_v h0)，W_v 是共享的无偏置 d_model × d_model 矩阵，初始化标准差为 0.02 / sqrt(d_model)。
- `zero` 从静止开始，初速度没有参数。

三种都取 h0 = N(token_embedding[id])。N 将向量归一化到单位 RMS，P(h,x) = x − mean(h*x) h 表示切向投影。

## 每个子层如何更新

1. 用 μ 决定沿旧速度前瞻多远。融合读取计算 q = (h + μv) / sqrt(1 + mean((μv)²))，然后做分支 pre-norm，运行一次 Attention 或 FFN，得到输出 f。
2. 取建议位移 s = f / sqrt(mean(f²) + 0.05²)，按旧速度 v 拆为平行分量 s∥ 和垂直分量 s⊥。新位置为 h′ = N(h + α∥ s∥ + α⊥ s⊥)。
3. 在 h′ 的切平面上取 u = P(h′, v) 与 d = P(h′, h′ − h)。把 d 按 u 拆分，写入 v′ = βu + γ∥ d∥ + γ⊥ d⊥。当参考速度为零时，平行分量取零。最后一个 FFN 的速度没有下游消费者，只学习三个位置系数。

μ 初始为 0，两个 α 用 softplus，初始为 0.05。β 用 sigmoid，初始为 0.5。两个 γ 独立、无界，初始均为 0.5。八层 static 共学习 15 × 6 + 3 = 93 个控制标量。Token-dependent 版在各系数的原始参数上加上 RMSNorm(h) 的线性投影，再施加相同的参数化。投影从零初始化，前瞻、位置和速度的调制尺度从 0.01 开始，因此在相同输入和初速度下，两版的初始系数及更新公式一致。[速度写入的来源](https://github.com/lululu39/NextTransformer/tree/chordres-zero/nexttransformer/methods/chordres_velocity_gains)

## 优化器与训练

| 参数 | 对应的 nanochat 组 | 峰值 LR（宽 512、每步 32,768 tokens） |
| --- | --- | ---: |
| μ、α、β、γ，token 调制权重与尺度 | routing AdamW，无 weight decay，与 mHC routing 一致 | 0.00244949 |
| 独立速度表 E_v | 与输入 embedding 同一 AdamW 组 | 0.09185587 |
| 初速度变换 W_v | 与同形状主干矩阵同一 Muon 组 | 0.005 |

所有组沿用 nanochat 学习率日程。训练使用 vanilla 全注意力、FA2、融合前瞻读取和 NTP loss，关闭 force 重算。Attention 输出投影和 FFN down 投影置零。1B 协议从零、seed 42、30,518 步。通过提交器传入 recipe、初速度、研究问题 group 和配对 baseline run。正式训练调用 scripts.connection_train。[提交入口](../native/chord/submit.py) · [实验结果](../../RESULTS.csv)

此前的四标量 Scalar 用 soft-normalized 分支输出作为目标，两个速度写入增益固定为 1 − β。复现实验时用原 run 的 manifest 和归档配方。[历史参数](legacy/chord_scalar_four.args) · [其他历史实验](legacy/README.md)
