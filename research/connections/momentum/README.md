# Heavy-Ball with approximate residual normalization

Status: implementation and bounded GPU validation only. No new formal training
is authorized. The five old plans in `stopped-runs.json` remain retired. Their
small result summary is `/home/yzhong239/nanochat-residual-runs-stopped-20261116.json`.
The sphere route, its launchers and geometry tests were removed at the user's
request; historical code remains in Git at `d2f106d`. Old sphere checkpoints
raise an explicit retirement error rather than silently changing architecture.

## Source and scope

- [anGPT paper](https://arxiv.org/html/2505.22014), section 4 and Appendix E.
- [Authors' repository](https://github.com/automl/anGPT), checked 2026-11-17 via
  the GitHub contents API: only `.gitignore`, `LICENSE`, and `README.md`; no model
  implementation was published there. We inspected the available repository,
  but cannot claim code-level equivalence to the authors' experiments.
- Appendix E gives the residual factor `1/sqrt((1-alpha)^2+alpha^2)`.
  Section 4.3's displayed factor omits the inverse square root. This implementation
  follows the second-moment derivation and Appendix E, not that inconsistent line.

Borrow the normalized branch plus analytically scaled residual idea; keep
nanochat's original pre-RMSNorm, Attention/RoPE/QK normalization, ReLU-squared
FFN, zero output-projection initialization, final norm/head, NTP loss, native
Muon/AdamW groups, warmup and LR schedule. No weight projection, new large
matrix, auxiliary loss, exact hidden normalization, tangent projection or
transport is added. This is **anGPT-inspired Heavy-Ball**, not full anGPT.

## Recurrence

The retained `heavy_ball` mode is unchanged:

```
u = beta*v + gamma*F(h)
h_new = h + eta*u
v_new = u
```

New `an_heavy_ball`, applied at each Attention and FFN sublayer:

```
z = F(h) / sqrt(mean_channels(F(h)^2) + floor^2)
u = beta*v + gamma*z
q_new = beta^2*q + gamma^2
cross = beta*c
scale = 1 / sqrt(1 + 2*eta*cross + eta^2*q_new)
h_new = scale*(h + eta*u)
v_new = u
c_new = scale*(cross + eta*q_new)
```

Here `q = E[v_i^2]`, `c = E[h_i*v_i]` are **nominal**, coefficient-derived
moments; the model assumes `E[h_i^2]=E[z_i^2]=1` and new z uncorrelated with
past h/v. The covariance produced by momentum is retained, including the
`2*eta*beta*c` term. Ignoring it systematically underestimates residual scale
even with independent innovations. These are two scalars per forward, not
statistics measured across tokens/batches, not EMA state and not learned matrices.
They reset at every forward and propagate differentiably through depth.

For beta=0 and gamma=1 this reduces to
`(h + eta*z)/sqrt(1+eta^2)`, which is the scalar anGPT interpolation correction
with `alpha=eta/(1+eta)`. No explicit alpha is needed in addition to eta.

The statistical assumptions are approximate in a trained network: F(h) can
correlate with h/v and learned embeddings can change their scale. This does not
guarantee fixed/bounded hidden norm. The validation report records actual hidden
and velocity RMS and their correlation separately from nominal moments.

## Stability and state

- Initial h is the unchanged nanochat RMS-normalized token embedding. v is an
  independent token velocity embedding, normal std 0.02, by default. q0=0.02^2,
  c0=0; explicit zero velocity remains available for diagnostics/old HB loads.
- beta=sigmoid(raw), gamma/eta=softplus(raw), three scalars per sublayer.
  Default HB `(0.5,0.5,1)` is preserved. New mode uses `(0.5,0.5,0.1)`.
- `floor=0.1` is a smooth RMS regularizer: derivative at a zero branch is 10,
  so the initial direct branch gain gamma*eta/floor is 0.5. This preserves
  nanochat's zero-initialized output projections without a singular normalizer.
  Branch RMS is <=1 and only approaches 1 for sufficiently large outputs;
  nominal moments therefore overestimate innovation power near initialization.
- Carry unscaled u, never eta*u or the hidden displacement. h/v are ordinary
  Euclidean vectors; velocity is not normalized, projected or detached.
- FP32 residual/moments, original BF16 branches; FP64 preserved in math tests.
- No cross-token state or added KV cache. Full prefill, chunked and token decoding
  use the same recurrence. Activation checkpointing carries the scalar moments.
- Checkpoints preserve model/optimizer/RNG/progress/config. Moment scalars are
  recomputed from coefficients each call, so there is no extra persistent state.
  Existing pure-HB parameter keys and optimizer grouping remain compatible.

Select `--residual-dynamics heavy_ball` or `--residual-dynamics an_heavy_ball` in
`scripts.connection_train`. Both default to token velocity embeddings. New-mode
knobs are `--an-force-rms-floor` and existing `--momentum-{beta,gamma,eta}-init`.
Vanilla remains the overall default. This directory has no formal-run launcher.

## Validation

GPU checks cover independent basis/covariance oracles, zero-force gradcheck and
second gradients, retained HB formula/state, seed-matched backbone, scalar and
embedding gradients, CCE/dense agreement, compile/checkpointing, causality/batch
independence, cached inference, optimizer/RNG roundtrip and CLI exit85 resume.
The bounded debug profiles same-card vanilla/HB/new-mode throughput and memory,
and checks finite training, held-out CE and actual per-layer residual scales.
Only local JSON/XML logs, no W&B. Theoretical FLOPs exclude scalar special
functions; profiler FLOPs are explicitly partial, not hardware totals.
