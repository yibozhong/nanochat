# 2K depth utilization diagnostic

Read-only evaluation of the completed 10B-token vanilla (6a07f085), native
(84ed009b), and vanilla+mHC (048c708c) **latest** checkpoints, all step 305176.
No fine-tuning, optimizer, auxiliary objective, or W&B run. Production model code
is unchanged. Native is not native+mHC. All have 8 blocks, D512 and 4 heads;
native has 125.83M parameters and SSSL/native extras, versus 58.72M/59.52M for
vanilla/mHC. Therefore only vanilla versus mHC isolates the connection change
reasonably well; single seed and different historical source revisions remain limitations.

## Protocol

Use all 64 original FineWeb-Edu dev sequences, 2048 NTP targets each (131072
tokens), same tokenizer, rows, and targets across all conditions. Reuse PackedData,
the existing model loader, evaluate(), NVIDIA mHC operators, BF16 activations,
softcap15 and full-vocabulary Apple CCE. Use cuDNN attention consistently; native
retains its SWA masks. Report historical validation CE differences, since native
training selected the automatic attention backend. Do not use the test set.

Layer index is 1..8, normalized depth l/8. A layer comprises attention and MLP.
Define h at the **pre-layer boundary**, before native resid_lambda/x0 injection.
The output boundary is after MLP, before the next layer's affine routing or final
readout/backout. Flatten [D,K] for mHC; never collapse streams for these metrics.

- Total update = h_after - h_before.
- For each sublayer, compute the pure-routed result by writing a zero branch using
  the **same routing coefficients** as the ordinary forward. The effective
  computational increment is actual_output - pure_routed_output; routing increment
  is pure_routed_output - sublayer_input. This includes the actual BF16 write rounding.
  Separately report deviation from the unrounded H_post*branch increment.
- Whole-layer compute is the sum of its two local effective increments. Whole-layer
  routing is both local routing increments plus native affine injection/scaling.
  They telescope to the total update. This is local additive accounting in fixed
  residual coordinates, **not** a causal attribution or a transported attention
  contribution through the later routing matrix.
- Report mean tokenwise ||update||/||h||, global Frobenius ratios, residual norm and
  coordinate variance, hidden cosine and update-residual cosine. Report adjacent
  total-update and compute-update cosines; undefined zero-vector cosine is excluded
  with valid-token counts. Sublayer denominators use that sublayer's own input.
- Skip exactly one block's attention and MLP computations. Keep both mHC routing
  operations; the second route is recomputed on its intervened input, not frozen
  to baseline coefficients. Preserve all native affine routing and final backout,
  whose saved middle representation comes from the intervened pass. No KV cache.
  Report signed delta CE and relative CE increase, without clipping negative effects.
- Save per-sequence metrics and losses, bootstrap sequences (4000 draws, seed42)
  for pointwise 95% intervals. Packed sequences are not necessarily independent
  documents; intervals describe this dev sample, not seed/model uncertainty.
  Early=layers1..4, late=5..8; additionally report late excluding layer8. Positive
  importance share uses clipped positive *mean* delta CE only as a descriptive score.

## Validation

Five independent synthetic checks cover full-stream norm, known ratios/cosines,
nonidentity routing with disabled compute, identity skip for additive residuals, and explicit undefined-cosine handling.
For each real checkpoint: exact hidden parity with the original eager forward;
all eight skipped forwards equal an independent module-output-zeroing hook oracle;
disabled branches are not called; layer decomposition identity; full-dev CCE
matches the existing evaluate(); historical CE discrepancy <.002; weights and
buffers have identical before/after fingerprints. No diagnostic weights are returned.

## References and scope

1. [Csordás et al., Do Language Models Use Their Depth Efficiently?](https://arxiv.org/html/2505.13898v3)
   The supplied incomplete ID is corrected to 2505.13898. Critically, late layers
   can remain important for the current prediction while contributing little to
   reusable future-token computations. All-position NTP skipping does not reproduce
   their prefix-only intervention, multihop tasks, or shallow/deep learned mappings.
2. [Liu et al., Inverse Depth Scaling From Most Layers Being Similar](https://proceedings.mlr.press/v306/liu26bt.html)
   ([full text](https://arxiv.org/html/2602.05970v2)). Weak adjacent update correlation
   is part of their evidence against smooth depth dynamics; the title does not
   predict that adjacent update vectors must have high cosine. Three same-depth
   checkpoints cannot test an inverse-depth scaling law or establish ensemble averaging.
3. [Sun et al., The Curse of Depth in Large Language Models](https://arxiv.org/html/2502.05795v6)
   The reported pruning/variance pattern is testable here, but these probes do not
   measure the block Jacobian or causally establish Pre-LN as the cause. The theory
   invokes distributional/independence assumptions and growth bounds; an exponential
   upper bound alone does not prove exponential growth for every trained model.

Our 8-layer FineWeb models are a small-scale applicability check, not a full
replication of these papers' large/deep models or proof of compositional computation.
