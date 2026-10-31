# Native nanochat: three formal 10B-token runs

User authorized all three on 2026-10-31 after native gate 7636681.0 passed.
Run native baseline, NVIDIA mHC, and NVIDIA no-Sinkhorn independently from seed42.
Model/trainer/runtime source is pinned to the tested1855eb4 commit; launcher and
submission/monitor tooling have a separate recorded commit.

- Data: the existing FineWeb-Edu sample-10BT cache, raw train000..012 / heldout013,
  revision87f09149, RustBPE32K. Same rows as the three previous vanilla runs.
- Budget per model:305176steps ×32768tokens =10,000,007,168 model tokens. The
  native best-fit loader reached source epoch2 while making this cache, so this
  is not10B distinct source tokens. No diagnostic checkpoint is reused.
- Native D512/8layers/4heads/2K, SSSL, all VE/smear/x0/scaling/backout retained.
  Baseline125,829,354params; either connector126,624,415params. Shared native
  initial parameters match bitwise; mHC/no-Sinkhorn parameters all match.
- Single Hopper GPU, micro16, accumulation1, BF16, activation checkpointing off.
  Native automatic FA3/SDPA dispatch preserves sliding windows. It differs from
  the previous full-attention experiment's forced cuDNN SDPA; this is not an
  isolated speed comparison between those two backbone architectures.
- Native MuonAdamW CLI-default learning-rate rules, batch/width/depth scaling,
  40-step warmup,65% warmdown,5% final LR; native scalar and VE groups preserved.
  Connector AdamW uses unembedding LR, zero decay. CCE covers every token and
  vocabulary entry. No clipping or auxiliary objective.
- Dev every250steps plus step50/final; final and dev-best test on2,097,152 tokens.
- W&B project`yibozhong657-none/nanochat`, group`mhc-native-pretrain-20261031`.
  One saved ID per formal experiment, reused on every resume. Offline fallback
  retains records for later synchronization. No diagnostic W&B runs.
- Resume: atomic overwrite of`latest.pt` every500steps; dev-selected`best.pt`
  independent. Every1800seconds exit85 uploads the latest state through the
  already validated scheduler mechanism. Preserve model/optimizer/RNG/step/data
  position and run ID. No numbered resume checkpoint history.
- Scheduling: public slots preferred by Rank, research-group backfill allowed.
  2CPUs/16GiB RAM/80GiB scratch, one whole SM90 GPU with at least22,000MiB.
- Outputs: unique home result/checkpoint archives for each job, about1.25GB of
  latest+best per model (under4GB for all three), within the current home quota.
  Reuse the existing staging data/runtime; no new full data cache is created.

`submit.py WORKFLOW_DIR` checks the completed GPU gate, immutable source equality,
existing marked jobs and an on-disk submission intent, then queues exactly three
formal jobs. It never modifies the earlier vanilla jobs or their inputs.
`watch.py WORKFLOW_DIR` monitors the same jobs, audits final config/metrics,
checkpoint member sizes and W&B summaries, and writes results in the workflow
folder. It does not submit any additional runs.
