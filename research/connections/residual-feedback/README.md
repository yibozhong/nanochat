# Detached per-layer attention over residual updates

`--feedback-source per_layer_qkv --feedback-detach --feedback-bank residual_update`

For each previous-pass block, store `attention_update + mlp_update` instead of
the accumulated output hidden state. Read branches directly and sum in FP32
before casting to BF16, avoiding cancellation from subtracting rounded hidden
states. The bank excludes both the identity path and the feedback injected
before the block. There are still eight entries per token for the eight-block
model, with the same per-entry RMS normalization and independent full Q/K/V/O
at every consuming block. Queries still read only the previous token's bank.

The existing zero-initialized output projections make every update zero at
initialization. The first step therefore behaves like the baseline; backbone
learning creates nonzero updates on subsequent steps. No initialization or
warmup changes are introduced. The original hidden bank remains the default.

The preflight runs in the same GPU allocation as the authorized single 10B:
regressions and new collection/zero-bank/gradient/causal/cache checks, paired
512-step hidden/update-bank timing and finite training, then CLI exit85 and
exact state/optimizer recovery checks. A P2 slowdown above 30% stops for review.
No diagnostic W&B runs or reused diagnostic weights. On success a new process
starts formal training from seed42; a per-family validation marker is reused
on resume. The preflight is bounded to 2400 seconds by the launcher.
