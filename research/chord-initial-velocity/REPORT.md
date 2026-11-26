# Chord 的初速度需要独立词表吗？

## 假设与实验设计

研究对象为六系数 token-dependent Chord。新实验默认查独立 token 速度表，也可选从归一化 h0 生成或从静止开始。本组以相同训练预算比较这三种选择。

本轮以已完成的 h0 生成版（2e37a95d）为参照，新增零初速、独立速度表两个完整 Chord 实验。假设是初速度的来源会改变早期及最终 CE。三者共用 vanilla 主干、六系数 token-dependent 更新、数据顺序、seed 42 和 1B 学习率日程。

判据是完整预算下的 validation CE 差值，并检查差距在训练中的变化。并列报告参数量、训练 FLOPs、同型号 GPU 上的吞吐及峰值显存。两种初速度的参数量和优化器分组均有差异，结论适用于这一整体替换。

## 实验事实

<!-- sync:begin -->

| 配置 | 状态 | 验证 tokens | 最近 CE | ΔCE | 基线 |
|---|---|---:|---:|---:|---|
| [chord-token-dependent-v0-table-s42-1b](configs/7337b20d.json) | 排队 | — | — | — | 2e37a95d |
| [chord-token-dependent-v0-zero-s42-1b](configs/aa44a2d7.json) | 排队 | — | — | — | 2e37a95d |

[完整验证曲线数据](curves.csv)

<!-- sync:end -->

## 分析、结论与待验证问题

截至 2026-11-26，residual-mechanisms 组的 h0 生成版已完成 1B，最终 validation CE 为 3.196963。该协议下的六系数 token-dependent 查表版和零初速版尚无正式结果。[已完成实验](https://wandb.ai/yibozhong657-none/nanochat-1b/runs/2e37a95d)

待验证问题：与 h0 生成相比，查表和零初速在同一 1B 日程下的 CE 曲线、最终 CE 和达到相同 CE 的时间如何变化？
