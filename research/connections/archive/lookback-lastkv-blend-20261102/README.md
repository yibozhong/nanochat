> Retired on 2026-11-02 at the user's request. This is a historical report, not an active workflow. Executable code was removed; use Git history for reproduction.

# Last-attention KV reuse and previous-chunk blending with Lookback

This replaces the stopped independent shared-projection model. History now
reuses each completed chunk's **last attention layer's existing K/V**, computed
from that layer's normalized input, before its attention and MLP updates.
There is no extra memory projection of the final block output.

The last-KV reuse and preceding-chunk blend follow the official LastKV LLM
implementation, pinned in `lastkv-reference.json`. For normal processing of Cn,
layer l reads older chunks' shared last-layer KV and a mixture for Cn-1:

    a[l,h] = sigmoid(prev_chunk_alphas[l,h])
    K_previous = a * K_last + (1-a) * K_local[l]
    V_previous = a * V_last + (1-a) * V_local[l]

Each layer/KV head has one learned gate, shared between K and V. Logits start
at zero (50/50). Original RoPE positions and existing Q/K normalization stay
unchanged. The requested changes do not include LastKV's deep-V scaling,
Marin/Llama backbone, or incremental generation interface.

For C1,C2,C3:

1. Run C1 normally; keep its prediction, raw final output H1, all its local KV
   pairs and a reference to its last pair M1 (no copy or new projection).
2. Run C2 with causal local self-attention and the per-layer blend of C1's
   local KV and M1. Keep H2, C2 local KV and its last pair M2.
3. Refine C1 once starting directly from H1 through the same Transformer.
   Each layer reads its own causal local KV, shared M2 and older revised history.
   The successor M2 is NOT blended with C2's local per-layer KV: the previous-
   chunk blend applies only to the chronological left neighbor in normal passes.
4. Replace M1 with the last attention KV produced during C1's rerun. Discard
   C1's logical per-layer bank and original final hidden. This is replacement,
   not KV-plus-delta. C1/C2 predictions remain unchanged.
5. C3 reads revised M1 plus the per-layer blend for C2, and its own causal KV.
   Then only C2 is revised, once. Older C1 is never recomputed again.

All memory/local tokens use one joint attention softmax. No detach, auxiliary
loss, additional entry normalization or embedding reinjection. Only original-
pass NTP supervises the model. Each independent sequence starts empty. The
last boundary still executes an unused rerun, as in the previous experiment.

There are 58,720,288 parameters at L8/D512/H4: the unchanged vanilla backbone
plus 32 gate logits. Gates use the existing connector AdamW LR/betas/epsilon,
zero weight decay. All backbone/head initial values and RNG remain matched to
seed42 vanilla; their optimizer settings are unchanged. Old Lookback checkpoints
are rejected explicitly rather than silently changing semantics on resume.

Only the latest original chunk retains layer-local KV for the next blend and
its raw final hidden for the next rerun. Older chunks keep only deep KV.
At B2/T16K/C2K/D512, deduplicated logical KV is 120MiB: 64MiB shared history
plus 56MiB additional predecessor layer KV; raw predecessor hidden is 4MiB.
Autograd retains extra versions/activations, so actual training VRAM is profiled.
Matrix FLOPs omit elementwise blending and other non-GEMM operations; the last
rerun layer's output/MLP has no downstream gradient consumer because only its
attention KV is retained.

`recompute_previous=False` retains LastKV-style chunk recurrence and blending.
It is a diagnostic control, not a separately authorized formal training run.
The 21-check GPU gate covers independent FP64 forward/all-parameter gradients,
closed-form gate derivatives, the two-chunk local-only limit against vanilla,
last-attention KV provenance, exactly-once/immutable memory scheduling, future
availability, production 16K causality/isolation/reset, CCE/compile/checkpoint
gradients and exact optimizer/RNG restore. Same-card reverse-order profiles
compare vanilla, memory-only and Lookback; a fresh-process 25→50 restart checks
real training recovery with the full 305176-step LR horizon.

The replacement is ONE new 10B run from zero, seed42, 305176×32768 tokens,
16K context/2K chunks, existing FineWeb-Edu cache/tokenizer/dev/test and default
MuonAdamW schedule. Completed vanilla7669762.0/W&B0bb03c6e is unchanged.
Micro2/accum1 is preferred; micro1/accum2 is selected only if measured reserved
memory exceeds 90% of 22000MiB. Tests never use W&B credentials or export weights.
Formal training alone logs nanochat/group lookback-lastkv-blend-10b-20261102.
Latest checkpoint overwrites every500 steps; best is separate; exit85 every
1800 seconds persists model/optimizer/RNG/data for preemption recovery.

Workflow: `/home/yzhong239/lookback-lastkv-blend-10b-20261102`.
`recompute_workflow.py WORKFLOW submit` submits validation; `watch` audits,
publishes and submits exactly one formal run after success. Inputs are immutable;
STOPPED.json blocks old workflows. Previous shared-projection results are in
`research/connections/archive/lookback-shared-projection-20261102`.
