# Frozen HC channel diagnostics

User requested a mechanism test against **HC without Sinkhorn**, not manifold HC.
Compare final seed-42 10B checkpoints: shared `6a494b4f` and mix-only static
channel scale `6df116ef`. Production model, kernels and active training stay unchanged.

One GPU job, no optimizer, W&B, new training or returned weights. Use the same
64 held-out 2K dev sequences and attention backend for both checkpoints. Verify
checkpoint SHA, metadata, historical CE within .002, exact instrumented forward,
and identical parameter/buffer fingerprints after the experiment. Five synthetic
checks precede real-checkpoint evaluation. Diagnose at all 16 connectors.

Measurements use every 32nd token: stream-centered energy, average pair cosine,
effective/stable rank before scale, after scale, after pure mix and after branch
write; H singular values, negative coefficients and row/column sum errors.
Geometry uses FP64 on sampled tensors; the original forward executes unchanged.
No contraction or double-stochastic assumption is made. Branch write remains
rank one in both selected models.

Frozen gate interventions: all ones, per-stream channel mean, train-calibrated
least-squares scalar, train-calibrated energy-preserving scalar, three fixed
channel permutations, and 16 single-site energy-scalar interventions. Calibration
uses only the first 32 sequences of the existing training calibration pack.
Energy matching is on original fixed calibration inputs, not a claim that all
intervened model outputs retain their norms. Channel permutations preserve the
distribution and cross-stream structure of scale vectors. Scalars remove all
channel variation while retaining a separate number for each stream/site.

Report sequence-paired CE deltas with 4,000 deterministic sequence bootstraps,
plus per-sequence raw measurements. This is a single trained seed and a frozen
intervention; dependence on learned scales does not establish that a scalar
control retrained from scratch would lose, nor that stream similarity is the
cause of the original loss gap. Test data is not evaluated. Training data hashes
and protocols are recorded rather than silently assumed identical.

`submit.py ROOT` submits held, audits, then releases one Hopper/A100 diagnostic.
`collect.py ROOT` only collects/validates; failures stop at `needs_attention`.
