# Chord 历史配方

本目录供点名某个历史变体或恢复历史实验时使用。

- **Next BF16 + 独立 token 速度表**。从仓库根目录向 `scripts.connection_train` 传入 `@research/connections/ngpt/legacy/chord_next_token.args`。这里固定保存当时的完整模型参数，与收录时的 Next 主配置仅差初速度来源。原提交名为 `vanilla_next_soft`。[参数文件](chord_next_token.args)
- **其他历史 Chord 消融**。用具体的旧配方名或 run ID 定位原始 manifest 和对应代码。日常入口收拢前的提交器可用 `git show 5c6d5ba:research/connections/native/chord/submit.py` 查看，原 Chord 迁移说明和各消融的参数见 `git show 5c6d5ba:research/connections/ngpt/CHORD.md`。从旧 checkpoint 恢复时使用该 run 自己的配置、源码版本和训练进度。
