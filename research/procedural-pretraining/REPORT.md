# Procedural pretraining

检验在相同 NanoChat 主干、初始化、训练步数和理论训练 FLOPs 下，将开头一段 FineWeb-Edu 换成程序数据，能否用更少 FineWeb tokens 达到基线的验证 CE。对照组为同一架构、seed、注意力后端和上下文的纯 FineWeb 训练。

## 数据与监督

使用 Jiang 等人公开生成器的 9 个 token 任务。每个输入输出已经是现有 RustBPE 词表内的 ID；模型两端与语言训练共用同一套参数。[论文](https://arxiv.org/html/2601.21725v2) · [原始生成器和配置](https://github.com/zlshinnick/procedural-pretraining/tree/e97ffe3)

| 任务 | 生成和训练目标 |
| --- | --- |
| `identity`、`reverse`、`sort` | 复制、反转或升序排列，只监督答案 |
| `set`、`union` | 单序列或两段序列按首次出现顺序去重，只监督答案 |
| `delete` | 输入给出去除的符号，按公开代码删除其第一次出现，只监督答案 |
| `stack` | 先偏向 push、后偏向 pop，栈内符号唯一，从栈顶到栈底输出，只监督答案 |
| `dyck`、`dyck_shuffle` | 64 对括号，嵌套或按未闭合类型采样。监督同一序列内每个有效下一 token |

输入输出任务默认 64 个输入符号。每例从原生词表随机选取并按 ID 排序的 99 个普通候选符号，保留作者代码的重复概率和排序关系，同时覆盖整个符号池。0 编码 POP，最后两个词表 ID 用于分隔和填充。Dyck 默认使用完整上下文，括号配对在整次训练中固定。第一篇的 Rule 110 实验使用专用二值输入和输出投影，公开的 token 生成器不含该任务。

为配对 packed2K 基线，多条独立生成的完整样本放入同一因果上下文，样本间留一个边界位置，尾部填充。后面的样本可以看到同一行中前面的样本。边界、填充和输入输出任务的提示均不贡献 loss。读出层在所有位置上计算完整词表，梯度按整个优化器 batch 的有效 target 数归一化。

## 两阶段训练

两阶段都使用既有 NanoChat 优化器和 LR 配方。阶段切换时在同一模型上新建优化器，不复制或重置任何模型参数，语言日程从第 0 步按剩余步数运行，FineWeb 从训练缓存第 0 行开始。中断恢复保留所在阶段的模型、优化器、随机状态和数据位置。[整模型迁移协议，附录 A.2](https://arxiv.org/html/2609.38764v1#A1.SS2)

现有训练入口为 `python -m scripts.connection_train`，调度入口为 `research/connections/native/chord/submit.py`。在同一 baseline recipe 上添加以下参数即可启用实验组，省略这些参数即为纯 FineWeb 对照。该示例占用 100 个原有训练步。

```text
--procedural-task union --procedural-tokens 3276800 --procedural-input-length 64
```

总步数和上下文与基线一致。warm-up 预算按实际模型位置计费，向上取整到每步 32,768 个，计入提示和填充。配置记录请求与实际预算。训练记录分别保存 `tokens`、`procedural_tokens`、`procedural_supervised_tokens`、`fineweb_tokens`；总 tokens 对应基线的训练 FLOPs 估计，实测耗时另计。以相同总 tokens 对齐 dev CE 曲线；数据效率以两条曲线首次达到同一基线 CE 所用的 FineWeb tokens 衡量。训练前、切换点和共同的验证步均评估同一份 FineWeb dev。

## 验证

使用作者公开实现 `e97ffe3` 进行同种子逐例对照。7 个输入输出任务各检查 384 个样本的序列、答案和监督位置，两种 Dyck 各检查 128 条 2,048 长序列，全部吻合。完整收益需用配对训练测量。
