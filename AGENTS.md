# Chord residual

新实验中，“Chord”或“ChordRes”默认指 token-dependent 六系数 BF16 版本。每个子层根据当前 token 的 h 调制系数。Static 六系数版本供显式指定的消融使用。两版共用 vanilla 主干、τ=0.05 的 soft force 和同一个 update-based 球面更新。

| 用户称呼 | 1B 提交参数 | 初速度 |
| --- | --- | --- |
| Chord、ChordRes、token-dependent 或 Next Chord | 默认，可显式指定 `--recipe chord_next_bf16` | 从归一化的 h0 生成。[默认模型参数](research/connections/ngpt/chord.args) |
| static 或 scalar 消融 | `--recipe chord_scalar_bf16` | 默认独立 token 速度表。[模型参数](research/connections/ngpt/chord_scalar.args) |

两个入口均接受 `--velocity-init zero|token|projected_h0`。比较控制方式时显式指定相同初速度。速度表沿用输入 embedding 的 AdamW 组，W_v 沿用主干矩阵 Muon 组，六系数及 token 调制使用 nanochat routing AdamW 组。历史四标量实验使用各自的 manifest 和归档配方。[Chord 说明](research/connections/ngpt/CHORD.md)

## 1B 实验工作流

本仓库新授权的正式训练默认从零、seed 42、1B tokens。所有新 1B 配方默认将每层 Attention 输出投影和 FFN down 投影置零。历史作业按其保存的配置恢复。packed FineWeb-Edu 下每步 32,768 tokens，共 30,518 步、1,000,013,824 tokens，warmup 和衰减使用 30,518 步日程。作业恢复沿用自身正式 checkpoint 和 run ID。GPU 诊断仍在同一分配内独立执行，只写本地文件。

- 正式研究只使用 W&B 的 [nanochat-1b 项目](https://wandb.ai/yibozhong657-none/nanochat-1b)。旧 nanochat 项目归档在原处。新协议的同步、分析和排名仅读取新项目。
- 一个 group 对应一个研究问题，使用稳定的英文短名，如 `chord-initial-velocity`。一个 run 对应一个配置，名称说明架构、改动和 seed。提交前在 `research/<group>/REPORT.md` 写清假设、配对基线、改变的变量和判据；先查已有报告，保留停训及负结果。
- 正式入口为 `research/connections/native/chord/submit.py`。指定 `--recipe`、`--group` 和 `--baseline-run`，可用 `--name` 定制名称。基线填 `self`，实验组填写该项目基线的 run ID。新配方复用这一入口和 `scripts.connection_train`。仅在获得训练授权后提交。来自旧项目的 CE 不参与新协议对照。残差研究组用 `--recipe residual_suite --group residual-mechanisms --baseline-run self` 提交一个顺序作业。先预检六个配置，再逐条正式训练，各自保存 checkpoint 和 W&B run。
- W&B 和本地配置快照记录完整模型、训练和优化器超参数、实际参数量、数据及 tokenizer 版本、seed、GPU、Git commit 和预算。终止作业时在作业目录写入含对应 `run_id` 的 `STOPPED.json` 与原因，以区分主动终止、失败和分段恢复。
- [总览](research/OVERVIEW.md)记录核心发现、状态和待验证问题；[总表](research/RESULTS.csv)保存客观指标；[leaderboard](research/LEADERBOARD.md)按协议分榜。每个方向在自己的 REPORT 内保留假设、实验设计、结果、分析和结论。报告写中文，技术术语可保留英文。区分实测事实、解释和未验证假设。
- 每次成功同步后，原子更新[单页研究看板](research/index.html)，作为 nanochat-1b 的日常阅读入口。Overview 汇总研究方向与结论，Leaderboard 沿用同协议完整预算排名，Experiments 内嵌专题报告和曲线，Running 显示调度状态与已训练 tokens，Archive 区分运行失败、主动终止、中断和完整配对后最终 CE 高于基线的结果。结论尚未复核时注明状态。页面使用同步器的同一批数据，可离线打开；同步失败时保留上一次成功生成的页面。
- 用 `uv run --no-project --script scripts/research_sync.py --review-ended` 刷新。脚本通过 W&B API 分页读取全部 validation CE，以训练 tokens 为横轴，生成每组的曲线图、原始曲线、配置快照及客观结果。AP 定时执行这一命令；组内已登记作业均结束后，Agent 解读完整曲线并更新 REPORT 与 OVERVIEW。数值同步失败、离线未上传、分析待复核应单独标记。已结束组的同一结果只复核一次。需要全量重读历史曲线时加 `--refresh`。
- ΔCE = 当前配置验证 CE − 基线在同一 token 点的验证 CE，负值表示改善。协议匹配包含数据划分、tokenizer、主干、输出初始化、精度、训练预算、LR 日程、主干优化器、seed 和验证口径。缺基线或缺同 token 验证点则留空。leaderboard 只列达到完整预算、通过相同协议核验的最终验证结果。早停或失败仍保留已观测曲线。
- 算力栏记录累计理论训练 FLOPs 及估算范围，吞吐量为训练 tokens 除以累计训练步耗时，显存为各恢复分段中最大的 allocated bytes。性能对比需要相同 GPU；不同协议分别解释。CSV 空值表示尚无有效观测。
- 自动事实区和已复核的分析区有 HTML 标记。手工备注写在标记外；新报告可沿用 [Chord 初速度报告](research/chord-initial-velocity/REPORT.md) 的结构。集群 job/PID、授权与停止操作继续记入 home 的 RESEARCH_HISTORY.md。
