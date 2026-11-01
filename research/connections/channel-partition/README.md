# Channel partition and expanded head, 10B tokens each

User authorization (2026-11-01): stop the vanilla mHC vocabulary-partition
orthogonal-mixing run, then train two fresh vanilla+mHC models for 10B tokens.
The stopped run is 7645934.0 / d340fc37. Its last durable step28500 checkpoint
and reports are preserved under mhc-orthogonal-mix-10b-20261031/stop-20261101;
that is 933,888,000 tokens, not necessarily its final in-worker step.

Both models retain the NVIDIA mHC backbone: 8 full-attention blocks, D512,
4 attention heads, 4 persistent streams, 20 Sinkhorn iterations per connector,
2K context, 32K vocabulary. There is no learned final collapse or final mixer.
Each original D-dimensional stream is RMS-normalized before readout routing.

* `channel_partition`: partition the vocabulary into four equal groups with
  the same private seed20261031 permutation as earlier experiments. Group g
  reads channels [128*g:128*(g+1)] from all four streams, concatenated in stream
  order into a 512-dimensional vector. Its head maps512 to8192. Feature routing
  is an exact coordinate permutation with no learned parameters. All four
  local CCE/LSE results form ONE full-vocabulary softmax, including denominator
  gradients for all groups. Total parameters59,507,120; head16,777,216.
* `dense_expanded`: concatenate all four normalized512-wide streams in stream
  order and apply a single2048->32768 head. Total parameters109,838,768;
  head67,108,864. This is4x the head parameters and matrix arithmetic, not4x
  whole-model training cost. CCE avoids materializing the entire logit tensor.

The channel layout preserves the original partition head's parameter/GEMM
counts; coordinate movement can still affect measured runtime. Independent
stream states allow a pre-softcap linear readout rank up to2048 in both new
models. If all streams coincide, channel partition can use only128 distinct
features per vocabulary group (512 globally). This capacity penalty does not
prove that training cannot collapse. No orthogonality loss or projection is
added. Logit softcap15 is retained and rank claims concern the pre-softcap map.

Both runs use seed42, identical same-shaped backbone and connector parameters,
and the existing tokenizer/packed10B FineWeb-Edu cache and held-out splits.
Channel head copies the previous vanilla head initialization. Expanded head
uses independent CPU seed43 with standard deviation0.001/sqrt(4)=0.0005,
matching the increased fan-in in expectation. Its head is necessarily a
different shape and does not claim exactly matched initial predictions.
All optimizer groups, including the head LR, keep the original D512-based
MuonAdamW settings. No LR sweep is included in this authorization.

Each formal run has305176 steps x32768=10,000,007,168 model tokens, microbatch16
and no accumulation, the original10B LR schedule, no activation checkpoint,
and the same cuDNN attention backend as the previous formal mHC comparison.
This is a same-token comparison with differing compute and parameter budgets;
report wall-clock cost alongside CE. The plain mHC10B comparison already exists.

Before release, a single GPU gate runs27 checks: independent FP64 coordinate/
logit/gradient oracle, rank-under-collapse example, BF16 CCE input/weight
gradients, target and denominator routing, matched initialization/optimizer,
actual2K causality, KV cache, compile/recompute, optimizer/RNG resume, and old
readout regression. It profiles collapse/channel/expanded on the same GPU and
does separate25->exit85->fresh-process50-step restart checks for both new modes
with the full10B LR schedule. Gate/profiling have no W&B and return no weights.

Formal training alone records to yibozhong657-none/nanochat, group
mhc-channel-partition-10b-20261101. Latest checkpoints atomically overwrite every
500steps; dev-best is independent. Exit85 every1800seconds persists checkpoints,
optimizer, RNG, run ID and exact step-derived data position through scheduler
transfer. Public GPU slots have soft priority; backfill is allowed. Each run
requests one full Hopper GPU with at least22000MiB, subject to measured memory
gate. Source, launchers and input hashes are immutable after submission.

Workflow: submit.py creates two held formal jobs and one GPU gate. watch.py
audits the gate, commits/pushes the report and implementation, then releases
both formal jobs. Failure leaves training held with needs_attention. Check
state.json, process/lock and scheduler before restarting; STOPPED.json disables
continuation. See protocol.json for exact hashes and settings.
