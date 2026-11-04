# FLA Triton moment attention验证与吞吐报告

GPU job7802199.0，H200，source `bc77aba06ec52e1e284123340b5bd3e414a815b1`。17/17检查通过；无W&B、无正式训练、无权重回传。

## 完整训练step

8层D512/H4，2K、micro16/accum1、32768tokens/step，BF16，原MuonAdamW和完整词表CCE。每组20步warmup、30步timed；新后端正序/反序各一轮，表中为两轮均值，显存为两轮峰值。包含前向、反向、梯度finite检查和optimizer，不含编译及定期validation/checkpoint。GB为十进制。

| 配置 | ms/step | 相对新baseline | allocated GB | reserved GB |
|---|---:|---:|---:|---:|
| none | 63.988 | 1.000x | 5.771 | 5.916 |
| second_moment | 71.939 | 1.124x | 6.329 | 6.375 |
| variance | 72.154 | 1.128x | 6.329 | 6.384 |
| jensen_silu | 75.125 | 1.174x | 6.329 | 6.375 |

同卡旧实现另测一轮：cuDNN baseline 61.377ms，独立FP32 variance 240.972ms，Jensen 228.541ms。
新variance比旧实现快3.34倍，Jensen快3.04倍。新baseline比cuDNN慢4.25%；统计分支相对同后端baseline开销分别为12.76%和17.41%。

## 正确性与计算结构

独立PyTorch dense oracle使用BF16输入/phi/输出mean，FP32 attention累加；对照的是BF16计算语义，不是逐位相同，也不宣称所有输入分布误差均相同。17/129/2048长度覆盖非整块尾部；D32/64/128、输出/QKV梯度、有符号/零gap、因果prefix及零future VJP均通过。

实际2K/D128：

| 配置 | mean相对L2 | 统计量相对L2 | Q/K/V最大相对L2 |
|---|---:|---:|---:|
| none | 0.2138% | 0.0000% | 0.2749% |
| second_moment | 0.2138% | 0.0883% | 0.3569% |
| variance | 0.2138% | 0.0886% | 0.4054% |
| jensen_silu | 0.2138% | 0.1098% | 0.4334% |

完整小模型compile/eager的梯度相对误差分别为baseline 0.1970%、variance 0.2897%、Jensen 0.2209%；三者CE最大绝对差 3.34e-06，且optimizer更新后全参数finite。

Profiler确认每个attention调用只有一次forward kernel，随后一次preprocess、一次dQ、一次dKV。扩展前向共享片上P/在线softmax，直接累计mean和Ephi并center；没有完整A或phi(V)的HBM张量。BF16TensorCore dot+FlashAttention标准FP32累加，不再运行独立FP32 attention。

独立算子的事件计时包含输出标量loss及反向，不能等同纯kernel时间：none 1.510ms, second_moment 2.588ms, variance 2.619ms, jensen_silu 3.009ms。详细逐kernel调用次数及样本见numerical-audit.json。

## 修复与边界

首轮7802187.0的D128扩展dKV需要246784bytes共享内存，超H200上限232448；改num_stages=1后全通过，数值阈值未调整。普通baseline kernel保持上游launch设置。

初版仅固定长度、全因果MHA，head_dim32/64/128；不支持KV-cache解码、GQA、变长或SWA，显式拒绝。native/mHC和旧reference路径保留。未做极端偏置/极小gap的全面误差界、长期训练稳定性或正式恢复测试，后续正式训练前应验证新后端恢复。

现有三组正式训练已停止，其checkpoint按用户要求删除；W&B/日志/报告保留。不得恢复旧job。后续效果实验需要从零训练同新后端baseline及变体，不能把旧cuDNN曲线直接当作严格匹配对照。本轮未启动新正式训练。
