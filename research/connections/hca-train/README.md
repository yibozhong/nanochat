# Fixed / Dynamic HCA: authorized paired 10B runs

User authorization (2026-11-12): after validation passes, automatically run one
Fixed HCA and one Dynamic HCA nanochat experiment, each 10B tokens. No baseline
training, extra run, or hyperparameter sweep is authorized by this plan.

Common recipe: vanilla backbone, baseline additive residual, 1 stream, 8 layers,
D512, 4 heads, 32768 vocabulary, context 2048, seed 42, BF16, existing MuonAdamW
schedule, no activation checkpointing, explicit Triton HCA kernels. FineWeb-Edu
sample-10BT and the existing packed2K cache/tokenizer are unchanged. Each run is
305176 updates of 32768 tokens = 10000007168 tokens. Both use SWA256, target B/R16,
temperature 1, and training Gumbel noise. Dynamic has a one-sided soft compression
loss with coefficient 1; this is not an exact count or achieved-ratio guarantee.
See [implementation semantics](../hca/README.md), including open-prefix summaries
and the per-microbatch global compression loss under gradient accumulation.

## Conditional execution

1. Freeze one committed source archive and two persistent W&B IDs; only Fixed is
   initially submitted, held/audited/released as a combined validation/training job.
2. Run the 24 Triton output/VJP/determinism/causality/cache checks, then rerun
   the 18 reference core checks and the 2K full/fixed/dynamic cache comparison.
   Model changes invalidate reuse of the original reference-only evidence.
   Use PyTorch math SDPA for the baseline's masked cached prefill.
3. On that same GPU, test the full 8-layer production shape, 3 finite updates plus
   exact model/optimizer/RNG/data restore and next-update replay under the
   recorded actual CCE gradients for each mode. Actual CCE backward was isolated
   as nondeterministic in the earlier diagnosis; its replay loss must still be
   exact, and its uncontrolled update differences are reported. Check nonzero
   pool/router gradients and measure memory/time. Choose the largest **common**
   microbatch from 16,8,4,2,1 within 90% of 78000 MiB. Only CUDA OOM/capacity
   overflow permits a smaller candidate; other failures stop the process.
   Separately measure three warmed updates per mode and baseline on this GPU,
   without replay recording. Stop if either HCA path is >=10x the baseline;
   this engineering guard rejects the previously diagnosed 184x/260x overhead,
   not a quality threshold or claimed architecture speedup.
4. After both pass, Fixed starts a clean formal process in the same allocation.
   A background controller validates the streamed success report and submits the
   one Dynamic job from the same frozen source, with exactly the chosen microbatch.
   Dynamic verifies its production shape/replay on its own allocation, then starts
   clean there. No diagnostic weights/RNG/data position enter formal training.

Initial GPU requirement is a complete H100/H200 >=78000 MiB, with soft public-pool
preference and backfill allowed. Other architectures are not yet validated; the
controller does not silently broaden resource constraints to them. Each job asks
for 2 CPUs, 32 GiB host memory and 80 GiB scratch, including input archives, unpacked
20GB data, runtime/container, compile caches and checkpoints. Startup validation
is bounded by 3600 seconds; failure preserves the allocation for 600 seconds for
manual diagnosis, then holds. The controller never releases or replaces a failed job.

Formal W&B only: `yibozhong657-none/nanochat`, group `hca-10b-20261112`. Diagnostic
steps have no W&B run and do not count as formal progress. The first finite step
after the exact formal marker is required before reporting training started.
The startup probe records measured train-only ETA; no speed/quality benefit is
assumed for the kernels before measurement. CE, total objective, compression loss and
soft/hard training keep rates are logged separately. Dev/test and best selection
use CE alone; the target R does not establish matched realized KV budgets.

## Persistence and operation

`train.sh` retains the validated successful startup marker across exit85 segments.
Every 1800 seconds the trainer saves model/optimizer/RNG/step/data position and the
same run ID. Latest and best are separate. Atomic archive failure returns 74,
never 85. Actual scheduler checkpoint transport was verified by the existing
checkpoint-safety eviction test; model-specific replay is tested at startup.
Completed summaries prevent a restored completed job from training again.

AP-side code imports stdlib only. No content-hash scans or remote HEAD audits.
Credentials are transferred encrypted from the existing 0600 netrc; they never
enter source, logs, or output archives. The controller collects small result/config
reports and verifies checkpoint member sizes without loading tensors on the AP.

```bash
python -m unittest discover -s research/connections/hca-train -p test_control.py -v
python research/connections/hca-train/submit.py initialize
python research/connections/hca-train/submit.py fixed_hca
python research/connections/hca-train/controller.py
```

The final command is intended as a detached background process. Plan, job states,
logs and controller PID live in `/home/yzhong239/hca-10b-20261112`. Persistent
submission intents, per-run locks, queue checks and frozen IDs prevent duplicate
submissions. A `STOPPED`/`STOPPED.json` marker stops new controller actions; the
operator must also cancel the two corresponding Condor jobs when stopping this
research direction.

## Verified Triton launch, 2026-11-12

The user requested interactive debug instead of repeatedly submitting failing
batch jobs. After repairing the interactive session lifetime, allocation
8526447.0 stayed active through all kernel fixes and validation. The actual GPU
was an H100 80GB (UUID d9edf92c-50c2-1c6a-fcb3-fdfeff4a708a).
Frozen training source `3ed347020ec65e8a4c59dc91419ed965aa4e7199` passed all 24
Triton tests, 18 reference tests, 2K cache comparison, and both full 8-layer
production/recovery checks at the common microbatch 16. The same-card warmed
updates were 0.08109 s baseline, 0.11359 s Fixed, and 0.16834 s Dynamic: 1.40x
and 2.08x baseline time. Train-only 10B estimates are 9.63 and 14.27 hours,
excluding evaluations, checkpoint transfer, initialization, and scheduling.
This validates engineering behavior, not 10B stability or compression quality.
Untrained Dynamic is not yet at the target ratio of 16.

Fixed starts a fresh formal process in that same interactive allocation, retaining
run ID `f6dd86fc`. After its first 1800-second segment, the debug parent archives
outputs and exits normally. `bridge.py` verifies exit85 from the trainer, source,
run ID, progress, recipe, validation evidence, and both checkpoint members before
releasing the existing held Fixed job 8492173.0. Initial bridge archives have
distinct filenames so subsequent scheduler checkpoints take precedence. The
container URI remains in TransferInput. Any failed/missing transfer blocks the
handoff. Dynamic retains run ID `6e68ed48` and uses the same frozen source.
The AP controller has separate operational commits; these do not change the
model source saved in the two formal checkpoints.
