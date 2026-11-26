# Chord 历史配方

本目录供点名某个历史变体或恢复历史实验时使用。

- **四标量 Scalar BF16**。旧版取 delta = soft_force − h，两个速度写入增益均为 1 − β。保留速度表的原参数供复现，历史 from-rest 消融只将初速度替换成零。[四标量参数](chord_scalar_four.args)

- **Next BF16 + 独立 token 速度表**。从仓库根目录向 `scripts.connection_train` 传入 `@research/connections/ngpt/legacy/chord_next_token.args`。这里固定保存当时的完整模型参数，与收录时的 Next 主配置仅差初速度来源。原提交名为 `vanilla_next_soft`。[参数文件](chord_next_token.args)
- **其他历史 Chord 消融**。用具体的旧配方名或 run ID 定位原始 manifest 和对应代码。日常入口收拢前的提交器可用 `git show 5c6d5ba:research/connections/native/chord/submit.py` 查看，原 Chord 迁移说明和各消融的参数见 `git show 5c6d5ba:research/connections/ngpt/CHORD.md`。从旧 checkpoint 恢复时使用该 run 自己的配置、源码版本和训练进度。

10B 时代的提交器、监控、测速及消融预检保存在清理前的提交 `bd86a04`。可用 `git ls-tree -r --name-only bd86a04 research/connections/ngpt research/connections/native/chord` 定位，再用 `git show bd86a04:<路径>` 读取单个文件。要复现某一条 run，以其 manifest 记录的 commit 为准。[早期 nGPT 实验记录](ngpt-10b.md)
