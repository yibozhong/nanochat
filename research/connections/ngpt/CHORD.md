# Chord residual 默认配方

Chord 为每个 token 维护球面上的 hidden `h` 和与它垂直的速度 `v`。每个 Attention 和 FFN 子层在前瞻点读取，用两个步长写回当前 hidden，再用实际位移更新速度。Attention、FFN、分支 pre-norm 和 NTP loss 沿用 vanilla 主干。

[交互演示](chord-demo.html)可下载后用浏览器离线打开。拖动球面、切换更新阶段或调节步长，即可观察一个示例 token 的位置与切向速度。

## nanochat 配置

新训练从仓库根目录向 `python -m scripts.connection_train` 传入 `@research/connections/ngpt/chord.args`，再指定本次的训练步数、学习率日程、数据与输出路径。跨仓库迁移时使用这份文件中的显式模型参数。历史实验按其保存的配置恢复。[默认模型参数](chord.args)

## 融合与反向重算

默认开启 `--fuse-chord-read-norm`，将 BF16 前瞻读取和分支入口的无权重 RMSNorm 融为一次读写，保留中间 BF16 舍入和原主干的 epsilon。反向从 h、v 和 mu 重建分支读取。`--recompute-chord-force` 可独立启用，默认为关闭。开启后，训练反向会用已经保存的输入与 BF16 权重各重算一次 Attention 输出投影和 FFN down 投影。KV 推理沿用正常投影。两项均使用 `--chord-kernel current`。

## 初始化与更新顺序

`mean` 沿当前 token 的 hidden 维度取均值。记 `N(x) = x / sqrt(max(mean(x*x), 1e-12))`，`P(h,x) = x - mean(h*x)*h`。正常尺度下，`N` 给出 RMS 为 1、半径为 `sqrt(d_model)` 的向量，`P` 把向量投影到该点的切平面。

输入时设 `h = N(token_embedding[id])`，再设 `v = P(h, velocity_token[id])`。速度表独立于主干词嵌入，形状为 `vocab_size × d_model`，按标准差 0.02 的零均值正态分布初始化。速度沿同一 token 的模型深度传递。位置由 Attention 的 RoPE 编码。复现历史 `token_position` 对照时，初速度额外加上绝对位置速度嵌入，再做切向投影。

对每层先执行 Attention，再执行 FFN；后者使用前者更新后的 `h, v`。下式的 `F` 表示带原 pre-norm 的当前分支，返回该分支的输出投影结果。当前默认以 BF16 保存跨子层的 `h, v`；球面归一化、投影、速度混合和弦位移在 FP32 计算。速度使用本次写回舍入前的 `h_next - h`，下一个子层的前瞻使用已舍入的两个状态。Attention 和 FFN 接收 BF16。一个位置和通道的两条跨层状态合计占 4 字节。旧参考训练 af1c3266 使用 FP32 状态，旧 checkpoint 继续沿用其精度。

```text
offset = mu * v
q = (h + offset) / sqrt(1 + mean(offset*offset))
delta = N(F(q)) - h
parallel = (mean(delta*v) / max(mean(v*v), 1e-12)) * v
h_next = N(h + alpha_perp*delta + (alpha_parallel - alpha_perp)*parallel)
chord = h_next - h
v_next = P(h_next, beta*v + (1 - beta)*chord)
```

- 前瞻系数 `mu` 是可取正负的子层标量，从 0 开始学习。前瞻公式利用了 `h` 的单位 RMS 和 `v` 的切向性。建议位移从当前 `h` 量到目标 `N(F(q))`。
- `alpha_parallel` 缩放建议位移沿历史速度的分量，`alpha_perp` 缩放其余分量。每个子层分别持有两个宽度为 d_model 的 softplus 参数向量。先沿历史速度分解建议位移，再逐通道缩放两个分量，每个通道的有效初值均为 0.05。
- `beta = sigmoid(raw_beta)`，初始 logit 为 0，有效初值为 0.5。每个把速度传给后续子层的位置各学习一个宽度为 d_model 的 β 向量。最后一个 FFN 之后，输出头只读取 hidden，因此 L 层共有 2L−1 个可学习 β 向量。两个步长和 β 在各自子层内跨 token 共享。对应的消融开关 `signed_chord_steps`、`token_chord_gates`、`token_beta_gate` 均为 `false`。
- `project_only` 将旧速度直接与实际弦混合，随后在新位置做一次切向投影。整个跨层更新参与反向传播。参考优化器将残差系数放入零权重衰减的 AdamW 组。速度 token 表默认与主词嵌入使用相同的 AdamW 超参。训练参数 `--velocity-embedding-lr` 可单独设置速度 token 表的绝对峰值 LR，沿用相同调度与权重衰减。

实验使用 `--velocity-init token --velocity-embedding-std 0` 时，独立 token 速度表所有行都从 0 开始，随训练一起更新，可用 `residual/velocity_embedding_rms` 观察它何时偏离 0。

实验使用 `--velocity-init zero` 时，模型省去速度表，每个 token 以 `v₀=0` 进入首个 Attention。第一个 FFN 的速度来自首个 Attention 实际位移的切向分量。

实验选项 `--fixed-first-attention-mu` 把首个 Attention 的前瞻系数全程固定为 1，直接从球面上的 `h0 + v0` 方向读取。初速度表继续学习。从同一 block 的 FFN 起，各子层按原配方学习自己的 μ。

默认配方启用 `--channel-chord-coefficients`。速度逐通道混合后整体投影到终点切平面。复现早期标量对照时，从配方中移除该选项。

实验选项 `--chord-velocity-write` 在归一化前把 `rho * v` 加入 hidden 的建议位移。rho 由每个子层的一个标量经 tanh 得到，初值为 0，此时复现默认写回。mu 控制前瞻读取距离，rho 控制旧速度对本次位移的直接贡献。下一子层的速度仍由实际弦与旧速度混合后做终点切向投影。8 层模型增加 16 个参数。

## 迁移校验与训练参照

对同一批输入，逐子层检查 `mean(h*h) ≈ 1` 和 `mean(h*v) ≈ 0`。把两个步长设为相同值时，写回应化简为 `N(h + alpha*delta)`。对照实现还覆盖非零前瞻、互异步长、速度为零时的梯度，以及整段推理与 KV cache 分段推理的一致性。[PyTorch 更新](../../../nanochat/lookahead.py) · [Triton 前反向](../../../nanochat/chord_triton.py) · [几何与梯度测试](../../../tests/test_chord_velocity_geometry.py) · [分支接线与优化器](../../../nanochat/connection_model.py)

对应的 10B 配方为 8 层、宽 512、4 头、上下文 2048、词表 32768。Attention 输出和 FFN down 投影按标准差 `0.02/sqrt(2*depth)` 初始化。参考训练使用 seed 42、FineWeb-Edu packed 2K、FA2 和 MuonAdamW，每步 16 条序列共 32768 tokens，目标 305176 步。[逐通道基线 9a7fd31d](https://wandb.ai/yibozhong657-none/nanochat/runs/9a7fd31d) · [该训练使用的源码](https://github.com/yibozhong/nanochat/tree/6765505b969c29e1cb0ccf7083c588113b0ff459)
