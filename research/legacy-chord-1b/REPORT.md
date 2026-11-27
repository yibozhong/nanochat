# Legacy · Chord 1B 历史配方与消融

## 结论

旧四标量完整版的最终 validation CE 为 3.19776（查表）和 3.20808（零初速）。它们使用 target-based 位置更新，两个速度写入系数均为 1−β。当前 token-dependent 的六系数版本使用 update-based 写入，初速度研究另行配对。

## 以切向 Chord 为参照的六系数消融

三条 run 均完成 1,000,013,824 tokens，seed 42、零初速、零输出投影。历史配对基线是完整的仅投影 Chord（aa44a2d7，CE 3.207224）。默认速度切换为 ambient 后，这组三条按原 run ID 归档。训练时各自采用的更新如下。

| 消融 | Run ID | 速度处理 | 最终 validation CE | 相对原基线 ΔCE |
|---|---|---|---:|---:|
| 固定 μ=0 | e1579b8b | 保留切向速度与两次方向分解，关闭前瞻读取 | 3.209975 | +0.002750 |
| 仅速度前瞻 | 961d9458 | 普通残差加法，速度在欧氏空间写入 | 3.267725 | +0.060501 |
| 仅球面 | 33c7f10a | 保留球面位置更新，无速度状态 | 3.261842 | +0.054617 |

在相同零初速、seed 和日程下，ambient 完整版的 CE 为 3.200898。固定 μ=0 若要隔离新版的前瞻贡献，需要使用 ambient 速度写入。后两条的自身递推不随完整模型的速度几何改变，可直接用已存曲线重新配对。原报告和对照图分别见[固定 μ](chord-lookahead/REPORT.md)和[仅前瞻、仅球面](chord-components/REPORT.md)。

## Scalar Chord 从静止出发的分量消融

沿用 vanilla 1B 协议、seed 42、BF16 和零输出投影，三条均从头训练。起点采用 Next 静态标量版的 v0=0。两条含速度的配置直接创建零张量；三条均无速度表或初速度投影。主干和学习率沿用 nanochat，球面分支使用当前 soft-force τ=0.05。[Next 标量版公式](https://github.com/lululu39/NextTransformer/blob/chordres-zero/nexttransformer/methods/chordres_zero/README.md)

| 配置 | 分支读取和位置更新 | 速度更新 |
| --- | --- | --- |
| 完整 Scalar，从静止出发 | 前瞻读取；沿速度、垂直速度两个标量步长，朝 soft-force 目标更新并回到单位 RMS 球面 | 用 β 混合旧速度与实际位移，再投影到终点切平面 |
| 仅球面写入 | 从 h 读取；保留一个正标量步长 α，更新为 N(h + α × (soft(F) − h)) | 无速度状态 |
| 仅速度前瞻 | 从 h + μv 读取并做原有 pre-norm；用 h′ = h + F 更新位置 | v′ = βv + (1 − β)(h′ − h)，在欧氏空间记录实际位移 |

有前瞻的子层均使用可学习标量 μ（初值 0）；有后续读取的速度使用可学习 β（初值 0.5）。球面写入的步长初值为 0.05。完整从静止版本和原 vanilla 基线 a2793a7e 配对；两条消融直接对比本次完整从静止版本。结合四组同 token 的 CE 可判断两个分量各自的收益，以及组合是否存在交互。

## 实验事实

<!-- sync:begin -->

| 配置 | 状态 | 验证 tokens | 最近 CE | ΔCE | 基线 |
|---|---|---:|---:|---:|---|
| [chord-scalar-bf16-zero-s42-1b](configs/57ba73d6.json) | 已完成 | 1000013824 | 3.19776 | -0.09489 | a2793a7e |
| [chord-scalar-from-rest-v0zero-s42-1b](configs/9c3d50ec.json) | 已完成 | 1000013824 | 3.20808 | — | a2793a7e |
| [chord-scalar-lookahead-only-v0zero-s42-1b](configs/62357991.json) | 主动终止 | — | — | — | 9c3d50ec |
| [chord-scalar-sphere-only-v0zero-s42-1b](configs/da4e2ef3.json) | 主动终止 | — | — | — | 9c3d50ec |
| [chord-token-dependent-lookahead-only-s42-1b](configs/961d9458.json) | 已完成 | 1000013824 | 3.26773 | 0.06050 | aa44a2d7 |
| [chord-token-dependent-sphere-only-s42-1b](configs/33c7f10a.json) | 已完成 | 1000013824 | 3.26184 | 0.05462 | aa44a2d7 |
| [chord-v0-zero-fixed-mu0-s42-1b](configs/e1579b8b.json) | 已完成 | 1000013824 | 3.20997 | 0.00275 | aa44a2d7 |

![09c92a717b71cb92 validation CE](curves-09c92a717b71cb92.png)

[完整验证曲线数据](curves.csv)

<!-- sync:end -->

## 来源与覆盖范围

- 57ba73d6 的配置和训练摘要取自原作业 9090785.0 回传包，验证曲线取自此前完整同步的 W&B 缓存。
- 9c3d50ec 的正式训练摘要记录了 30,518 步和终点 CE，保留摘要中的起点和终点。da4e2ef3 在训练中停止，最后记录第 3,550 步。62357991 完成工程预检后取消，正式训练步数为零。三条的来源为作业 9093243.0 的正式日志及停止记录。
- 查档早期四标量配置时，W&B API 未返回这四个 run。本地归档保留原实验 ID、来源配置、可验证指标和训练状态，活动 leaderboard 排除这四条。[冻结指标](runs.json)

- 三条六系数消融的逐点验证曲线已与同步缓存逐点核对，终点和成本来自 HTCondor 回传的正式训练摘要。各自的原配置保留了原 group、run ID 和 baseline。离线归档由 [runs.json](runs.json) 和 configs/ 下的配置驱动。

<!-- agent:begin -->

<!-- reviewed: 57ba73d6@已完成@1000013824@3.19776@-0.09489,9c3d50ec@已完成@1000013824@3.20808@—,62357991@主动终止@None@—@—,da4e2ef3@主动终止@None@—@—,961d9458@已完成@1000013824@3.26773@0.06050,33c7f10a@已完成@1000013824@3.26184@0.05462,e1579b8b@已完成@1000013824@3.20997@0.00275 -->

**实测结论** 在 1B、seed 42 的零初速配对中，完整切向 Chord 的 CE 为 3.207224。固定 μ=0 高 0.002750，仅前瞻高 0.060501，仅球面高 0.054617。更早的四标量查表版最终 CE 为 3.197763。

**待验证** 在相同零初速下，固定 μ=0 对 ambient 完整 Chord 的验证 CE 有多大影响？

<!-- agent:end -->
