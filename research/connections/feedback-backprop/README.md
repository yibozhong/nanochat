# Cross-pass gradient diagnostic

One bounded GPU job compares `feedback_detach=True` (existing default) with
`False` for `embedding_qkv` and `per_layer_qkv`. Full gradients retain the
earlier backbone graph and propagate through the shared normalized layer bank.
The fused Triton backward writes dQ and dBank without atomics, including zeros
for the unused last source token. Reassociated full-width Q/K/V/O stays enabled.

Both arms use only final-pass CE, seed 42, the same first 2,048 packed training
batches (67,108,864 tokens/model), original 10B optimizer schedule, 15% P1 / 85% P2,
BF16, FA2, CCE and compiled blocks/readers. No activation checkpointing. Dev
P1/P2/P4 is measured initially and every 256 updates. Four comparators run one
at a time on the same GPU; arm order reverses between architectures.

Report complete optimizer-step latency by pass count, sampled mean latency,
peak allocated/reserved VRAM, training CE averaged over the final 128 steps and
paired dev curves. Latency excludes first 128 updates and first post-eval steps.
No W&B, permanent model checkpoints or automatic formal continuation.

Run `submit_debug.py` from a clean committed tree. It freezes inputs, submits
held, audits resources/container transfer, then releases exactly one job.
`debug.sh` reuses the frozen runtime and extracts the unchanged data prefix on
the worker. `check.sh` runs operator/model/causal regressions, the comparison,
and CLI exit85/restore checks for both full-gradient modes. Pipeline timeout
is 2,400 seconds; failure preserves the allocation for 900 seconds. A read-only
`collect.py` persists small results and never releases or resubmits jobs.

New CLI flag: `--no-feedback-detach`. The default and old checkpoint semantics
remain detached. Full gradients are currently supported for the two projected
layer-bank variants. Inference forward and exact causal cache recurrence are
unchanged; the switch only changes training gradients.

The paired initialization tensors are checked bitwise. Initial FP32 CE reductions
are compared within one float32 epsilon scale; repeated reduction order can move
the reported mean by a fraction of an ULP. This does not change the gradient
oracle tolerances. The Torch reference uses matching compilation and checkpoint
settings so BF16 compiled/eager rounding is not conflated with kernel accuracy.
