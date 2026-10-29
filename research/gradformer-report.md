# Gradformer 探索总结与归档

2026-10-28。用户决定本轮探索告一段落，归档研究结果，并将 nanochat、llama2.c 的 main 恢复为原生 Transformer 实现。保留 uv 环境、Git 作者、W&B 项目约定、原始实验输出和可恢复的实验代码历史。

## 研究问题与结论

出发点是将深度方向的 hidden 更新理解为优化过程。最初让 block 预测最终 NTP 对其输入的梯度，用 SGD/Adam 更新 hidden，同时用 detached true-gradient targets 的 MSE/cosine 训练预测方向。之后依次检查辅助强度、梯度范数配平、纯 cosine、EMA teacher、严格 local CE、读出端 NTP 训练，以及每层 learned energy 的真实梯度。

现有实测未建立稳定的 NTP 质量或总算力优势。提高梯度对齐并不自动改善任务损失；补足输出端训练、调节参数 LR 和 hidden eta 可以显著改变结果，但调过 LR 的普通 Transformer 仍优于本轮最好的 per-layer energy 配置。暂时结束这条架构主线，不继续追加 optimizer/energy 变体。

这不是方向不可能有效的证明。主要结果为单 seed、小模型短程探索，部分早期实验训练200M tokens；不同训练预算的 CE 不直接比较。特定方法未穷尽所有 LR/eta 组合或参数化，理论解释与新颖性判断也未获得系统证明。

## 关键实证

### 最初的 direct-gradient 监督：200M-token 预算

llama2.c 为6层、D288、15.19M参数；同一 H200、seed1337、200,015,872 tokens。原始低权重联合目标的有效 MSE/cosine 系数为0.1/0.01。

| 配置 | 最终验证 CE | 训练累计时间 |
| --- | ---: | ---: |
| 普通 Transformer | 1.40670 | 9.59 min |
| Hidden Adam，无辅助 | 1.49114 | 12.11 min |
| Hidden Adam + MSE | 1.48571 | 20.77 min |
| Hidden Adam + MSE + cosine | 1.48293 | 20.69 min |
| Hidden Adam，MSE/cosine均1 | 1.46629 | 见原始报告 |
| 总辅助/NTP 参数梯度范数比控制为1 | 1.57330 | 见原始报告 |

范数配平确实实现了所设强度约束，最终 cosine约0.067，但任务CE更差。Hidden SGD 的三个200M-token对照CE为1.40596/1.40512/1.40750，接近普通baseline1.40670；单seed小差异不支持稳定收益，监督版本增加计算。

nanochat 单卡200M-token对照：8层D512、约123.47M参数，同一 RTX PRO6000 Blackwell Server Edition。普通baseline CE1.12642、训练8.84min；hidden Adam+MSE+cosine CE1.16373、16.89min。原定八卡作业未执行完成，不存在已验证八卡预训练结论。

代表记录：[llama2 baseline](https://wandb.ai/yibozhong657-none/nanochat/runs/6lgzi4ia)、[joint](https://wandb.ai/yibozhong657-none/nanochat/runs/0jts2o6n)、[范数配平](https://wandb.ai/yibozhong657-none/nanochat/runs/d4bdrk4q)、[nanochat baseline](https://wandb.ai/yibozhong657-none/nanochat/runs/avn3q7te)、[nanochat joint](https://wandb.ai/yibozhong657-none/nanochat/runs/vuar8pzy)。

### 后续诊断：1000步 /32,768,000 tokens

| 路线 | 主要结果与含义 |
| --- | --- |
| 纯 cosine，head没有NTP | CE9.081894、cosine0.583240；高对齐未换来好预测，且混入输出端未训练的问题。 |
| cosine blocks + NTP head | 8组扫描最好CE2.678559；固定末层eta对照2.808591，NTP-only对照1.932952。补足head后仍有差距。 |
| EMA targets | joint CE1.921512，对照online1.925177；pure CE7.677125。小改进未证明稳定收益；精确suffix重算的同卡成本约baseline5.3–5.8倍。 |
| 严格 local SGD vs connected e2e | 各13组等额扫描；最优local2.020580、e2e1.787003。完整验证集分别2.060319/1.828854。此e2e是hidden-SGD，不等同普通residual baseline。 |
| Local CE + gradient监督 | 用户提前取消；中心只保留最近500步CE2.590430/cosine0.035335等部分记录。不能称六组扫描完成。 |
| 每层条件 energy 真梯度 | 8组energy +1个普通baseline全部完成；最佳energy-SGD CE1.959035。解析导数、混合二阶outer导数、因果性、编译/CCE、恢复检查通过。 |

纯cosine的targets随各自模型改变，对齐指标不是共同固定teacher任务上的统一分数。高辅助loss拟合能力不等于能提高NTP泛化或减少达到相同CE的计算。

代表记录：[cosine+head](https://wandb.ai/yibozhong657-none/nanochat/runs/h749wjta)、[pure cosine](https://wandb.ai/yibozhong657-none/nanochat/runs/s85axjjl)、[EMA joint](https://wandb.ai/yibozhong657-none/nanochat/runs/qe1k8k3q)、[local SGD](https://wandb.ai/yibozhong657-none/nanochat/runs/g67o54rx)、[connected SGD](https://wandb.ai/yibozhong657-none/nanochat/runs/k1pg6pt6)、[最佳energy](https://wandb.ai/yibozhong657-none/nanochat/runs/vhnzpfdz)。

### 普通 Transformer 的 LR 扫描

保持同一1000步预算、模型、数据、seed、全词表CCE和训练日程；4个新run +复用已有.0005 baseline，全部正常结束并与W&B核验。

| 参数峰值 LR | 最终验证 CE |
| --- | ---: |
| .00005 | 2.988640 |
| .0005 | 1.793137 |
| .0015 | **1.680402** |
| .005 | 1.696200 |
| .015 | 3.059543 |

最佳普通baseline为[LR.0015 / v0jjrscm](https://wandb.ai/yibozhong657-none/nanochat/runs/v0jjrscm)。本轮最佳energy与之相差0.278634 CE。该LR只作为这套1000步pilot的参考，不声称长程训练也最优。

同卡中心对照的energy-SGD/普通Transformer训练耗时为81.642/75.251秒，约增加8.5%；它们的CE为2.335076/1.793137。最佳energy运行于L40，最佳baseline运行于RTX PRO6000 Blackwell Server Edition，不能直接比较两者秒数。参数量、tokens与单步速度相近也不等于达到相同质量的成本更低。

## 解释、限制和未做的实验

以下是讨论得到的假设，不是经消融证明的失败原因。

- 标签条件下的纠错梯度与推理时所需的表示计算未必相同。以单位置CE为例，logits梯度为p-one_hot(y)；当p匹配真实条件分布，梯度的条件均值为0，而计算p仍需要非零表示变换。不能把此例外推成整个架构的不可能性定理。
- 在线target随模型和hidden改变，并包含当前block及后续计算的Jacobian；梯度拟合与NTP训练存在耦合。EMA实验只检查一种teacher平滑方式，没有解决全部可能问题。
- 当前energy在RMSNorm(z)上定义；固定context、忽略epsilon时，z与其energy梯度正交，限制了径向更新。这来自本次参数化。将归一化尺度作为条件的消融尚未执行。
- 各层独立energy不意味着在下降同一个全局目标。局部energy下降也不保证最终CE下降。
- 近端/隐式更新、共享energy和低秩二次约束只是讨论候选，没有实现或提交。优化器式hidden更新和energy层已有相关工作，具体组合的新颖性未得到系统查重确认。
- 没有完成八卡验证、跨seed显著性评估、所有方法的完整LR交叉网格或大规模外推。

## 停止状态、代码回退与保留材料

用户明确结束本轮探索。此前单独取消的local-gradient批次保持取消；最后仍排队的nanochat八卡任务7310803.0也于本次收尾取消。取消前JobStatus=1，launcher为eight.sh；它未产生八卡训练结论。不会自动重提Gradformer训练。

| 仓库 | 实验前原生代码基点 | 回退前实验tip | 可恢复归档tag |
| --- | --- | --- | --- |
| llama2.c | 350e04f | 9373a45 | gradformer-archive-2026-10-28 |
| nanochat | 92d63d4 | 41eca4c | gradformer-archive-2026-10-28 |

使用新增回退commit保留完整Git历史，不重写main历史。原生模型/attention/export路径恢复到上述基点；Gradformer实现、专用训练/测试/提交脚本从main移除。独立保留uv环境、Python3.11、W&B依赖/项目约定、通用本地文件与凭据ignore规则，以及本总结。移除Gradformer专用依赖组。

实验代码归档：[llama2.c tag](https://github.com/yibozhong/llama2.c/tree/gradformer-archive-2026-10-28)、[nanochat tag](https://github.com/yibozhong/nanochat/tree/gradformer-archive-2026-10-28)。本地还保存两个经过git bundle verify的完整历史bundle，以及逐commit清单，位于/home/yzhong239/gradformer/archive/closeout-2026-10-28。

全部既有数据、容器、不可变源码bundle、checkpoint归档、日志、W&B记录和报告保留，不因代码回退而删除。结果根目录为/home/yzhong239/gradformer/results，报告根目录为/home/yzhong239/gradformer/reports。旧路径的实验脚本应在归档tag中查找，不从恢复后的main运行。

## 证据索引

工作区reports目录中的主要材料：report-initial-2026-10-28.md（初轮原始报告）、gradient-balance.md、ema-pilot.md、local-ce-pilot.md、local-lr-sweep.md、cosine-head-sweep.md、local-gradient-cancellation.json、layer-energy.md、energy-final-audit.json、baseline-lr.md、baseline-lr-audit.json。每份详细报告保留其配置、commit、指标和对应W&B链接。

## 回退验证

恢复实现的commit：llama2.c为0d597a6220d521bab41f35292e6b29c7f7490d24，nanochat为008b6b6051eddecc96782d93ef1fa0908d54c344。之后仅补充本报告，不改变已验证的代码。

- 静态核对：除明确保留的通用环境文件外，两个仓库的上游文件与实验前基点逐字一致；活动Python代码及依赖定义无Gradformer引用。Python语法解析、git diff --check、两个uv.lock的离线一致性检查通过。
- GPU验证：作业7337307.0在NVIDIA L40、Python3.11/Torch2.9.1+cu128的固定uv-v4容器中正常退出，exit 0。两个原生模型均通过BF16 forward/backward、有限梯度、固定batch四步loss下降、因果前缀和checkpoint恢复后logits完全一致检查；llama2.c原生模型导出正常。
- nanochat原有测试：SDPA fallback、engine/KV cache及optimizer选取测试共15项，全部通过（0失败、0错误、0跳过）。没有运行八卡或长程预训练验证。
- [W&B恢复验证 nrh0qrrd](https://wandb.ai/yibozhong657-none/nanochat/runs/nrh0qrrd)为finished；服务端summary与传回结果一致。报告、恢复清单和验证JSON/XML已随该run归档。此run仅用于代码恢复检查，不作为新架构实验结果。

验证细节保存在archive/closeout-2026-10-28中的static-verification.json、rollback-verification.json、native-pytest.xml和wandb-verification.json；完整返回文件为results/restoration-check-7337307.0.tar。
