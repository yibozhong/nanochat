# Chord residual 配方

Chord 在每个 token 上沿模型深度传递位置 h 和速度 v。h 的 RMS 为 1，球面半径为 sqrt(d_model)。v 位于 h 的切平面。每个 Attention 或 FFN 从前瞻点读取，在球面上更新 h，再用实际位移更新 v。跨子层的 h、v 以 BF16 保存，几何运算使用 FP32。[标量版交互演示](chord-demo.html)

| 1B 配方 | 初速度 | 分支输出如何参与位置更新 | Attention 输出与 FFN down 投影初始化 |
| --- | --- | --- | --- |
| `chord_scalar_bf16`，原默认 | 独立 token 速度表，查表后投影到切平面 | 把 soft-normalized 输出当作目标，取从 h 到目标的差 | 正态，标准差 0.02 / sqrt(2 × depth) |
| `chord_next_bf16`，Next 主配置 | 对归一化后的 h0 作无偏置线性变换，再投影到切平面 | 把 soft-normalized 输出直接用作建议位移 | 零 |

两版都用 vanilla 全注意力、FA2、融合前瞻读取、原分支 pre-norm 和 NTP loss，默认关闭 force 重算。训练入口是 `scripts.connection_train`。跨仓库迁移使用对应参数文件。[Scalar BF16](chord.args) · [Next BF16](chord_next.args)

## 初始化与子层更新

以下均按单个 token 计算。N 表示将向量归一化到单位 RMS，P(h,x) = x − mean(h*x) h 是球面点 h 处的切向投影。h0 取 N(token_embedding[id])。标量版初速度为 P(h0, velocity_embedding[id])，速度表按标准差 0.02 初始化。Next 主配置为 P(h0, W_v h0)，W_v 形状为 d_model × d_model，按标准差 0.02 / sqrt(d_model) 初始化，归入主干矩阵 Muon 组。

令 q 为前瞻点。实现利用切向性计算 q = (h + μv) / sqrt(1 + mean((μv)²))。F(q) 表示当前 Attention 或 FFN 在分支 pre-norm 后的输出。两版共用 s = F(q) / sqrt(mean(F(q)²) + 0.05²)。Scalar 将 s 当作目标，取 delta = s − h。Next 将 s 当作建议位移，取 delta = s。

Scalar 的 Attention 输出及 FFN down 投影默认仍用随机初始化。若将它们置零，初始两个步长同为 0.05 时，从分支输出到下一位置的切向导数为 0.05 / (0.95 × 0.05) ≈ 1.053。旧 Scalar checkpoint 使用原配置；复现历史硬归一化配方时在参数文件后指定 `--chord-soft-force-tau=0`。

把 delta 拆为沿 v 的平行分量和其余分量。两版均以 h_next = N(h + α∥ delta∥ + α⊥ delta⊥) 更新位置。当 v 为零时，平行分量为零。

- **Scalar** 每个子层的 μ、两个正步长 α 和 β 跨 token 与通道共享。有效初值依次为 0、0.05、0.05 和 0.5。速度更新为 v_next = P(h_next, βv + (1 − β)(h_next − h))。
- **Next** 由该位置当前的 RMSNorm(h) 调制 μ、两个 α、β 和两个速度写入系数 γ。先将旧速度和本次弦位移投影到终点切平面，得到 u 和 d，再将 d 按 u 的方向拆分。v_next = βu + γ∥ d∥ + γ⊥ d⊥。两个 γ 初值均为 0.5，token 调制投影从零初始化，三组可学习调制尺度从 0.01 开始。末端 FFN 只写位置。[速度读写实现](../../../nanochat/chord_token_controls.py)

## 1B 协议与复现

从仓库根目录调用 `python research/connections/native/chord/submit.py`，传入表中的 `--recipe`、研究问题 `--group` 和同协议基线 `--baseline-run`。配对时也要匹配表中的输出初始化。新授权实验从零、seed 42，30,518 步 × 32,768 tokens。两种配方的选择属于研究约定，1B 实测见研究总表。[提交入口](../native/chord/submit.py) · [研究总表](../../RESULTS.csv)

历史变体仅在明确指定时查阅。[Legacy 索引](legacy/README.md)
