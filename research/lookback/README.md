# Same-Transformer chunk recomputation: design requested2026-11-02

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
3. Rerun C1 from its original token embeddings using the same Transformer
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

This file records the clarified design, not an implemented or validated model.
No new recomputation training job has been submitted. Fresh16K data preparation
scripts and validated cache provenance remain reusable.
