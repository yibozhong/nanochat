# Static full-width residual mixers

Two optional vanilla+mHC variants preserve the original NVIDIA H_pre/H_post
read/write, branch pre-norm, attention/MLP, and final learned stream collapse.
Only the residual-mixing function changes. Each attention/MLP sublayer owns its
own static parameters, shared across token positions; they are learned during
training. Persistent tensor order is `[batch,time,channel,stream]`.

* `mhc_static_linear`: flatten2048 -> bias-free linear2048, identity initialized.
* `mhc_static_mlp`: flatten2048 -> bias-free linear512 -> ReLU -> bias-free
  linear2048. Both matrices are nonzero, down variance2/2048, up variance1/512.

The output is `reshape(mixer(flatten(S))) + H_post * branch_output`.
There is no additional identity bypass, mixing norm, bias, gate, auxiliary
loss, dynamic matrix generator, or Sinkhorn. The identity passed to NVIDIA's
final expand-combine kernel acts on the already mixed tensor; it never adds
original S. The bottleneck version cannot preserve an arbitrary2048-vector
at initialization and may discard information; this is part of the experiment.

New mixing matrices use nanochat's existing Muon parameter groups and defaults.
Original read/write routing parameters remain in their original AdamW group.
Backbone weights, optimizer hyperparameters, routing initialization and RNG
sequence are matched. No changes are made to existing running jobs or bundles.
At d512/depth8, total parameter counts are126,099,621 and92,545,189 respectively.
The MLP variant halves the mixer's GEMM FLOPs, not total model FLOPs. FLOP
estimates exclude Muon optimizer work; measured steps include it.

The GPU gate covers20 checks, including a separate FP64 formula/VJP with explicit
BF16 forward boundaries, no-bypass/normalization checks, initialization and
optimizer grouping,2K causality, decoding, CCE, compilation, recomputation and
exact optimizer/RNG restore. Same-GPU timing compares original mHC, current
channel-local mixing and both new variants. Each new variant resumes from a
real exit85 at25 to128 steps, with the full10B LR horizon; initial/trained
per-sublayer scales and gradients are recorded. Diagnostics never log to W&B
and never return weights. Passing a short gate is not a long-run quality claim.

After validation and push, exactly two10B runs use the existing FineWeb-Edu2K
cache, seed42, microbatch16/accum1, original FA3 and default optimizer schedule.
Latest checkpoints overwrite every500 steps, best is separate, and scheduler
checkpoint exit85 persists both archives every1800 seconds. Only formal
training logs to nanochat/static-mixing-10b-20261105. Monitoring stops after
the first finite training step of each run.
