# 全局 leaderboard

按协议分榜。配置和配对基线都完成全部训练后，按最终 validation CE 升序排列。

## 08e694cd0517647a · 1,000,013,824 tokens

native · 8 层 × 512 · context 2048 · seed 42

| 配置 | 研究问题 | 最终 CE | ΔCE | 基线 |
|---|---|---:|---:|---|
| native-baseline-zero-s42-1b (7e37ab98) | [residual-mechanisms](residual-mechanisms/REPORT.md) | 3.16445 | 0.00000 | self |

## 09c92a717b71cb92 · 1,000,013,824 tokens

vanilla · 8 层 × 512 · context 2048 · seed 42

| 配置 | 研究问题 | 最终 CE | ΔCE | 基线 |
|---|---|---:|---:|---|
| chord-next-bf16-zero-s42-1b (2e37a95d) | [residual-mechanisms](residual-mechanisms/REPORT.md) | 3.19696 | -0.09569 | a2793a7e |
| vanilla-attnres-zero-s42-1b (d38f2bd0) | [residual-mechanisms](residual-mechanisms/REPORT.md) | 3.20451 | -0.08814 | a2793a7e |
| vanilla-mhc-zero-s42-1b (b4e0525d) | [residual-mechanisms](residual-mechanisms/REPORT.md) | 3.23794 | -0.05472 | a2793a7e |
| vanilla-baseline-zero-s42-1b (a2793a7e) | [residual-mechanisms](residual-mechanisms/REPORT.md) | 3.29265 | 0.00000 | self |
