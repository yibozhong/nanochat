# Same-Transformer chunk recomputation

The two cross-attention Lookback variants have been retired. Their runtime
modules, tests and launchers are removed; historical reports are retained in
`research/connections/archive/lookback-cross-attention-20261102`. Original
source remains in Git history and immutable training bundles. Do not restart
either stopped Lookback workflow. The vanilla16K baseline continues with its
original input snapshot; `baseline_watch.py` only collects that existing job.

The replacement design requested by the user reuses the same Transformer and
its per-layer Q/K/V projections, with no learned Lookback module or shared
extra KV projection. For C1,C2,C3:

1. Process C1 normally and preserve its original predictions and per-layer KV.
2. Process C2 reading C1; preserve C2's original predictions and per-layer KV.
3. Rerun C1 from its original final block output using the same Transformer
   weights. At every layer, expose C2's completed per-layer KV in addition to
   C1's ordinary causal local attention and any available preceding chunks.
4. Replace C1's history KV with its recomputed per-layer KV. C3 reads revised
   C1 and the original C2. Do not recompute or replace any earlier prediction.
5. After C3 finishes, revise C2 once in the same way; older revised chunks
   remain unchanged. There is exactly one rerun per boundary, not iteration.

The original token/RoPE positions are retained. Recomputed states may include
information from the successor chunk but only future chunks may consume them.
Use functional bank replacements so prior outputs and saved autograd inputs
remain immutable. Keep end-to-end NTP gradients and reset at sequence start.

`nanochat/chunk_recompute.py` implements this path as `lookback_recompute`.
The model has the identical 58,720,256 parameters, seed42 initialization and
MuonAdamW parameter groups as vanilla. No projection or trainable tensor is
added. `recompute_previous=False` is the no-lookback sequential KV baseline;
its outputs and gradients must equal full vanilla up to numerical rounding.
The existing native and NVIDIA mHC paths remain unchanged.

The rerun starts directly from the original final block output, BEFORE the
LM-head RMSNorm. It does not re-embed tokens, add an entry norm, detach the
hidden or introduce a learned projection. The first block still applies its
ordinary pre-norm internally. Original absolute RoPE positions are retained.
Its history excludes its own obsolete cache, includes
all earlier revised chunks, and includes the completed successor. Every history
token is visible, while local attention remains triangular. There is one joint
attention softmax over history plus local tokens, not two added attentions.
Each layer stores its ordinary input-projected K/V. First-layer revised KV can
now differ because its input is the original final hidden. The first rerun
attention output and subsequent KV can incorporate the successor. The original
final hidden remains connected to its original stack's autograd graph. Only the
latest unrevised chunk retains its final hidden in the logical bank; older
revised chunks retain only KV.
The last boundary also executes a rerun, although this final revised
cache has no later loss consumer. Ordinary Transformer residuals are retained.

Training uses only original-pass next-token loss. Revised KV remains connected
to autograd. Old bank versions remain immutable until their backward consumers
finish; replacing the logical bank does not release all saved activations.
The live BF16 KV bank at B2/T16K/D512/L8 is 512 MiB, plus4 MiB for one final-hidden
chunk; peak training VRAM is measured separately. FLOP estimates include extra
reruns and omit activation/norm/optimizer
costs and any activation-checkpoint overhead. They count allowed attention
pairs, not hardware work on masked entries.

The authorized experiment is one new 10B-token run, 305176 updates of32768 tokens,
16K context with2K chunks, seed42, standard full attention, original32K tokenizer,
FineWeb-Edu sample-10BT, and the same fresh16K packing/dev/test as baseline
7669762.0 (`0bb03c6e`). That baseline continues with its immutable original source.
No additional baseline or sweep is launched. Reuse the existing full16K archive;
do not reconstruct the deleted2K caches. Runtime and optimizer settings remain
the proven cuDNN/BF16 single-Hopper configuration.

`recompute_workflow.py WORKFLOW submit` submits the GPU gate;
`recompute_workflow.py WORKFLOW watch` holds a process lock, audits results,
publishes source/report, and submits exactly one formal run after success.
Sixteen checks cover an independent FP64 formula/gradient oracle, vanilla
equivalence, versioned-cache availability, full16K causality and isolation,
compiled/CCE/checkpoint gradients and optimizer/RNG restoration, plus an isolated
gradient path from revised first-layer V through the original final hidden into
the original last MLP. Two opposite-order
same-card timing repetitions choose micro2/accum1 if it fits90% of22,000 MiB,
otherwise micro1/accum2. The gate additionally validates a25→exit85→fresh-process50
restart with the full305176-step LR horizon. Diagnostics never receive W&B
credentials or return weights. Formal training uses W&B nanochat and latest500
overwrite, separate dev-best, plus1800-second exit85 durable checkpoints.

The active workflow/group is `lookback-final-hidden-10b-20261102`. The previous
embedding-start formal job7674729.0 (`b8a6f136`) is stopped, with no durable
checkpoint returned before cancellation. Its reports are archived under
`research/connections/archive/lookback-embedding-recompute-20261102`; its code
remains in Git. Checkpoint metadata explicitly records `recompute_input=final_hidden`;
loading the retired embedding-start model through the new path is rejected.
The replacement run starts from zero with a new W&B identity.

Validation status and job identities live in the workflow state and, after
successful GPU validation, `recompute-validation.json`. Implementation alone
does not establish correctness, feasibility or a quality benefit.
