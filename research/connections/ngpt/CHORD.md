# Chord residual 默认配方

Chord 为每个 token 维护球面上的 hidden `h` 和与它垂直的速度 `v`。每个 Attention 和 FFN 子层在前瞻点读取，用两个步长写回当前 hidden，再用实际位移更新速度。Attention、FFN、分支 pre-norm 和 NTP loss 沿用 vanilla 主干。

## nanochat 配置

以下是传给 `python -m scripts.connection_train` 的模型参数片段。迁移时请显式保存 `project_only`。原 CLI 和缺少此字段的历史 Chord checkpoint 使用 `parallel_transport`。

```text
--variant baseline --architecture vanilla --streams 1
--residual-dynamics ngpt_lookahead --ngpt-velocity-mode lookahead_split_learned_beta
--chord-velocity-geometry project_only --chord-kernel current --chord-state-dtype bfloat16
--velocity-init token_position --velocity-embedding-std 0.02
--lookahead-mu-init 0 --ngpt-alpha-init 0.05 --momentum-beta-init 0.5
--ngpt-force-rms-floor 1e-6 --no-ngpt-weight-norm
--residual-projection-init normal --residual-projection-init-std 0.02
```

## 初始化与更新顺序

`mean` 沿当前 token 的 hidden 维度取均值。记 `N(x) = x / sqrt(max(mean(x*x), 1e-12))`，`P(h,x) = x - mean(h*x)*h`。正常尺度下，`N` 给出 RMS 为 1、半径为 `sqrt(d_model)` 的向量，`P` 把向量投影到该点的切平面。

输入时设 `h = N(token_embedding[id])`，再设 `v = P(h, velocity_token[id] + velocity_position[pos])`。两个速度表均独立学习，每项按标准差 0.02 的零均值正态分布初始化。缓存解码时 `pos` 包含 KV cache 的位置偏移。速度沿同一 token 的模型深度传递。

速度位置表消融只把上述参数改为 `--velocity-init token`，此时 `v = P(h, velocity_token[id])`。Attention 继续使用主干的 RoPE。

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
- `alpha_parallel` 缩放建议位移沿历史速度的分量，`alpha_perp` 缩放其余分量。每个子层分别持有两个 softplus 参数，有效初值均为 0.05。
- `beta = sigmoid(raw_beta)`，初始 logit 为 0，有效初值为 0.5。每个把速度传给后续子层的位置各学习一个 β。最后一个 FFN 之后，输出头只读取 hidden，因此 L 层共有 2L−1 个可学习 β。两个步长和 β 在各自子层内跨 token 共享。对应的消融开关 `signed_chord_steps`、`token_chord_gates`、`token_beta_gate` 均为 `false`。
- `project_only` 将旧速度直接与实际弦混合，随后在新位置做一次切向投影。整个跨层更新参与反向传播。参考优化器将残差标量放入零权重衰减的 AdamW 组，速度表放入 embedding 组。

## 迁移校验与训练参照

对同一批输入，逐子层检查 `mean(h*h) ≈ 1` 和 `mean(h*v) ≈ 0`。把两个步长设为相同值时，写回应化简为 `N(h + alpha*delta)`。对照实现还覆盖非零前瞻、互异步长、速度为零时的梯度，以及整段推理与 KV cache 分段推理的一致性。[PyTorch 更新](../../../nanochat/lookahead.py) · [Triton 前反向](../../../nanochat/chord_triton.py) · [几何与梯度测试](../../../tests/test_chord_velocity_geometry.py) · [分支接线与优化器](../../../nanochat/connection_model.py)

对应的 10B 配方为 8 层、宽 512、4 头、上下文 2048、词表 32768。Attention 输出和 FFN down 投影按标准差 `0.02/sqrt(2*depth)` 初始化。参考训练使用 seed 42、FineWeb-Edu packed 2K、FA2 和 MuonAdamW，每步 16 条序列共 32768 tokens，训练 305176 步。实现版本为 `84a4e16`。[投影版训练 af1c3266](https://wandb.ai/yibozhong657-none/nanochat/runs/af1c3266) · [固定版本源码](https://github.com/yibozhong/nanochat/tree/84a4e16b148de495a2c14c7ceb9c4bde10afb638)
