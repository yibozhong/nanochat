# Shared final-hidden KV with Transformer Lookback

Every completed chunk has one memory pair, projected from its final block
output by the same two learned, bias-free D→D matrices. These matrices are
shared across all layers, chunks, original passes and Lookback passes.
There is no learned cross-attention module or per-layer historical KV bank.

For C1,C2,C3:

1. Run C1 normally; keep its original prediction and raw final hidden H1.
   Build M1=SharedKV(H1).
2. Run C2 using M1 at every layer and ordinary local causal self-attention.
   Keep its original prediction and raw final hidden H2; build M2=SharedKV(H2).
3. Refine C1 once, starting directly from H1 before the LM-head RMSNorm,
   through the SAME Transformer stack. Every layer attends jointly to its
   local C1 tokens, M2 and any earlier completed/refined chunk memories.
4. Build M1_updated=SharedKV(H1_refined) with the same matrices. Replace M1;
   do not add a KV delta. M2 is reused, not projected again.
5. C3 reads M1_updated and M2 at every layer plus its own causal self-attention.
   After C3 completes, refine only C2 once. Previously refined C1 is not rerun.

Both normal and Lookback passes use one attention softmax over historical
memory plus local K/V. Local Q/K/V still use the current layer's original
projections. All historical memory is visible; the local block is triangular.
Lookback is therefore not future-only cross-attention. Its local K/V are
transient activations, not persistent history. Obsolete cache for the chunk
being refined is excluded from the external memory list.

SharedKV uses RMSNorm(H), Wk/Wv, original absolute token positions for key RoPE,
and the backbone's1.2-scaled key RMSNorm. H remains the raw final block output;
refinement starts from H directly, without an additional entry norm or detach.
The LM head retains its existing final RMSNorm. All projections and both passes
are trained end-to-end by original-pass NTP only. Prior predictions and old
bank tensors remain immutable; new memories become visible only to later
chunks. Each independent sequence starts with an empty bank. The final boundary
also performs its one refinement, even though it has no future loss consumer.

The new model has59,244,544 parameters: vanilla58,720,256 plus524,288 shared K/V
parameters. Existing backbone/head initialization and RNG remain matched to
seed42 vanilla. New matrices join the existing Muon group with unchanged LR
and decay settings. Native and NVIDIA mHC math remains unchanged. Only the
latest unrevised chunk retains its raw final hidden in the logical bank.
At B2/T16K/D512, BF16 history is64MiB KV plus4MiB previous hidden. Actual training
VRAM is measured separately; autograd retains old versions and activations.

`recompute_previous=False` is the shared-memory baseline. It is NOT equivalent
to full vanilla, since its history comes from final hidden. The GPU gate checks
it against an independent dense oracle. With only one chunk, backbone outputs
and gradients still match vanilla. Eighteen checks cover shared KV provenance
and reuse, exactly-once replacement, joint local/history softmax, independent
FP64 model/gradients, final-hidden gradient paths,16K causal isolation/reset,
compiled/CCE/checkpoint gradients, optimizer ownership and exact restoration.

Two opposite-order same-card profiles compare full vanilla, shared-memory-only,
and shared-memory-plus-Lookback. These are local diagnostics, not additional
formal experiments. Profile memory chooses micro2/accum1 if under90% of22,000MiB,
otherwise micro1/accum2. The new model also passes25→exit85→fresh-process50 recovery
using the full305176-step LR horizon. No diagnostic W&B credentials or weights
leave the worker.

The authorized replacement is ONE new10B run from zero:305176×32768tokens,
16K context/2K chunks, seed42, original32K tokenizer/FineWeb-Edu16K packed cache,
matched dev/test, default MuonAdamW and LR. Reuse completed vanilla7669762.0
(W&B0bb03c6e); no new formal baseline or sweep. Only formal training records to
nanochat/group lookback-shared-kv-10b-20261102. Latest500 overwrite, separate
best and1800-second exit85 persistence/recovery remain enabled.

Active workflow: /home/yzhong239/lookback-shared-kv-10b-20261102. `recompute_workflow.py
WORKFLOW submit` launches validation; `watch` audits, publishes, submits exactly
one formal run and monitors it. Validation results appear in
`recompute-validation.json` only after success. Source and every job input are
immutable. Old embedding-start and per-layer-KV checkpoints are explicitly
rejected by the new configuration. Prior runs are STOPPED; their reports and
available checkpoints remain archived. Do not restart them.
