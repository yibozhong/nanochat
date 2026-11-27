# Nanochat 1B 研究总览

正式研究使用 [nanochat-1b](https://wandb.ai/yibozhong657-none/nanochat-1b)。默认从零、seed 42，使用同一 packed 数据，30,518 步 × 32,768 tokens。

[每日看板](index.html) · [实验总表](RESULTS.csv) · [同协议排名](LEADERBOARD.md) · [实验规范](../AGENTS.md)

## 研究方向与状态

<!-- sync:begin -->

- [chord-initial-velocity](chord-initial-velocity/REPORT.md) · 已完成 2 · 已复核
  - **实测结论** 在相同协议和 seed 42 下，三种配置均完成了 1,000,013,824 个训练 tokens。从归一化 h₀ 投影生成初速度的基线（2e37a95d）最终验证 CE 最低，为 3.196963，独立速度表为 3.199604（ΔCE +0.002641），零初速为 3.207224（ΔCE +0.010261）。本题缺少 h₀ 基线的中途验证点，只能在终点与它比较。
  - **待验证** 在匹配参数预算和优化器设置的多种子配对实验中，h₀ 投影能否在 1B tokens 时稳定取得比独立速度表更低的验证 CE？
- [chord-velocity-geometry](chord-velocity-geometry/REPORT.md) · 已完成 2 · 待 Agent 复核结论
- [legacy-chord-1b](legacy-chord-1b/REPORT.md) · 主动终止 2，已完成 5 · 已复核
  - **实测结论** 在 1B、seed 42 的零初速配对中，完整切向 Chord 的 CE 为 3.207224。固定 μ=0 高 0.002750，仅前瞻高 0.060501，仅球面高 0.054617。更早的四标量查表版最终 CE 为 3.197763。
  - **待验证** 在相同零初速下，固定 μ=0 对 ambient 完整 Chord 的验证 CE 有多大影响？
- [procedural-pretraining](procedural-pretraining/REPORT.md) · 主动终止 5，等待同步 1
- [residual-mechanisms](residual-mechanisms/REPORT.md) · 已完成 5 · 已复核
  - **实测结论** 5 个配置均以 seed 42 完成 1,000,013,824 tokens。在零输出初始化的 vanilla 主干上，基线最终验证 CE 为 3.29265，Chord Next、AttnRes 和 mHC 分别降至 3.19696、3.20451 和 3.23794，对应 ΔCE 为 −0.09569、−0.08814 和 −0.05472。Native 基线的 CE 为 3.16445，但其主干和注意力协议不同，本组缺少可单独衡量 residual 贡献的 native 配对对照。
  - **待验证** 若给 Chord 和 AttnRes 相同的 H100 训练时长，并用多个随机种子做配对复验，Chord 的最终验证 CE 是否仍稳定更低？

<!-- sync:end -->

## 核心发现与待验证问题

上方按方向汇总已复核的结论和待验证问题。首次实验前可打开对应 REPORT 查看假设、对照和判据。
