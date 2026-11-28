# 加深 vanilla 主干时，Chord 能否改善训练稳定性？

## 实验设计

固定宽度 256、2 个 Attention 头（每头 128 维）、FFN 宽度 1024、全长 2048 的 causal attention。在 32、64、128 层各跑 vanilla、static Chord、默认 token-dependent Chord。每个深度的两个 Chord 都与同深度 vanilla 配对。两个 Chord 共用独立 token 初速度表、切面投影、六系数、BF16 状态及 τ=0.05 的 soft force。两者只改变系数是否依赖当前 token。

九条均从零、seed 42，使用相同 FineWeb-Edu packed 数据顺序和 tokenizer，Attention 和 FFN 输出投影零初始化。每步 32,768 tokens，microbatch 8 × 梯度累积 2，共 30,518 步、1,000,013,824 tokens。复用 nanochat 的 LR 与 weight decay 缩放；同深度的主干优化器相同。宽度固定后，增加层数也增加参数和每 token FLOPs；两个 Chord 的速度表各增加 8,388,608 个参数。

## 假设与判据

假设 Chord 对深层训练的帮助会随深度增加。分别记录三种深度的终点 validation CE、最后一亿 tokens 共同验证点的平均 ΔCE、训练中断或非有限梯度发生的步数。同时检查整条 CE 曲线和突发 loss spike。固定 seed 的单次轨迹用于识别现象。

第 1 步及每 250 步，记录首、中、末 block 输出 h 的 NTP 梯度 RMS 和全模型参数梯度范数。三个 h 的梯度来自该步首个 microbatch、按有效 batch 缩放后的 backward；采样沿用训练图。若早层与末层的梯度比值持续极端，结合 loss 和参数梯度定位。比较完整预算的吞吐、峰值显存和理论 FLOPs，速度按同 GPU 型号解释。作业内先在实际形状下检查零初始化梯度、4 个训练步和中途 checkpoint 恢复，通过后同卡从零正式开训。

## 进度

正式结果待同步。[W&B 实验组](https://wandb.ai/yibozhong657-none/nanochat-1b/groups/chord-depth-scaling)

<!-- sync:begin -->
<!-- sync:end -->

<!-- agent:begin -->
待完整配对结果后复核。
<!-- agent:end -->
