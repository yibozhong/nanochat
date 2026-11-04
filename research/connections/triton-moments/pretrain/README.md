# Triton attention: three matched 10B runs

User authorized exactly `none` (new baseline), `variance`, and `jensen_silu`.
All use `attention_kernel=fla_triton`; no second-moment formal run is submitted.
The production implementation is identical to the validated `bc77aba` source
(17 correctness checks and H200 throughput gate `7802199.0`).

Each run starts from seed 42 with matched common parameters, vanilla full
attention, 8 layers, D=512, 4 heads, 32K vocabulary, 2048 context, microbatch 16,
accumulation 1. There are 305176 steps / 10000007168 training tokens with the
existing 2K FineWeb-Edu cache, dev/test splits, default MuonAdamW and full 10B
learning-rate schedule. Alpha is fixed at 1, additional statistic projections
start at zero. No changes to native or mHC.

The only additional gate checks exact optimizer/RNG restoration and shared
initialization (4 tests), followed by a real 25 -> exit 85 -> fresh process 50
restart for each mode. Diagnostics use no W&B credentials and return no weights.
The gate uses the full 305176-step LR horizon; it is not a short-budget pilot.

After audit and publication, the collector submits three single-Hopper jobs
(>=22GB, public soft priority, backfill permitted). Formal runs alone log to
`yibozhong657-none/nanochat`, group `triton-moments-10b-20261103`. Latest checkpoint
overwrites every 500 steps; best remains separate. Every 1800 seconds exit 85
persists checkpoints and logs, then resumes model, optimizer, RNG and data cursor
under the same run ID. The collector stops after all three execute a finite-loss
training step. It does not monitor the entire 10B training or restart old runs.

Workflow root: `/home/yzhong239/triton-moments-10b-20261103`.
`workflow.py ROOT submit` and `workflow.py ROOT monitor` are guarded against
duplicate submissions and `STOPPED.json`. Results are in `recovery-validation.json`
and the workflow root's immutable source, submission audits and first-step file.
