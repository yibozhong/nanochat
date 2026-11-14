# Original top feedback with cross-pass gradients

`--latent-feedback --feedback-source top --no-feedback-detach`

Keep the original top-hidden RMS normalization, one full-width fusion matrix,
previous-token shift, and input-only additive feedback. Retain the earlier
backbone graphs and pass their output directly into fusion. There is no depth
attention and no new Q/K/V projection. Only the final pass computes CE.

This completes the existing 2×2 comparison: top versus per-layer QKV, each
with detached versus full cross-pass gradients. All runs use seed42, the same
FineWeb-Edu packed2K data, optimizer/schedule,15%P1/85%P2, P4 evaluation and10B
tokens. The diagnostic and formal processes initialize independently.

The same GPU allocation runs53 targeted tests, paired512-step detached/full
gradient checks and timing, and CLI exit85/recovery validation before starting
formal training. The gate has a2400-second bound, no W&B, and a per-family
success marker that is reused on resume. Failures stop the launch chain.
