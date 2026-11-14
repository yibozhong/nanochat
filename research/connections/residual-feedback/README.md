# Per-layer attention over residual updates

`--feedback-source per_layer_qkv --feedback-detach --feedback-bank residual_update`

Use `--no-feedback-detach` for full backpropagation through earlier passes and
their residual banks. Both settings keep the same forward, initialization,
independent full-width Q/K/V/O and final-pass-only CE. The existing Triton bank
gradient and compiled branch-sum backward handle the connected case; no new
projection or materialized KV bank is introduced.

For each previous-pass block, store `attention_update + mlp_update` instead of
the accumulated output hidden state. Read branches directly and sum in FP32
before casting to BF16, avoiding cancellation from subtracting rounded hidden
states. The bank excludes both the identity path and the feedback injected
before the block. There are still eight entries per token for the eight-block
model, with the same per-entry RMS normalization and independent full Q/K/V/O
at every consuming block. Queries still read only the previous token's bank.

The existing zero-initialized output projections make every update zero at
initialization. The first forward therefore matches the baseline; backbone
learning creates nonzero updates on subsequent steps. Full-gradient training
also differentiates through the zero bank to its producing branches, so its
first backward differs from the detached variant. No initialization or
warmup changes are introduced. The original hidden bank remains the default.

The preflight runs in the same GPU allocation as the authorized single 10B:
regressions and new collection/zero-bank/gradient/causal/cache checks, paired
512-step hidden/update-bank timing and finite training, then CLI exit85 and
exact state/optimizer recovery checks. A P2 slowdown above 30% stops for review.
The full-gradient gate adds 17 checks (187 total), including zero-bank
derivatives, joint hidden/update backward, every earlier bank's gradient,
matching compiled/checkpointed Torch references, CCE/dense gradients and cache
causality. It times hidden and residual banks with full gradients and a third
detached residual arm, each for 512 updates on the same GPU. The 30% gate
compares residual and hidden banks at the same gradient setting.
No diagnostic W&B runs or reused diagnostic weights. On success a new process
starts formal training from seed42; a per-family validation marker is reused
on resume. The preflight is bounded to 2400 seconds by the launcher.

The authorized 10B keeps seed42, d8/D512/T2048, microbatch16/global32768,
15% P1 / 85% P2, eval P4 plus P1, FA2/CCE and compiled blocks. The full-gradient
residual run completes the hidden/residual by detached/full-gradient factorial
comparison; the other runs retain their frozen code and configuration.
