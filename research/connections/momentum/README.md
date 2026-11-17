# Heavy-Ball with nGPT-inspired residual interpolation

Status: implementation and bounded GPU validation only; no new formal training.
The user redirected this task from anGPT to nGPT before GPU tests. The pending
anGPT debug job was held during environment preparation; no anGPT validation
result is claimed. The interim approximate-moment implementation is archived
in Git at `63a20cc` and removed from the active model.

The five old formal plans in `stopped-runs.json` remain retired. Their summary
is `/home/yzhong239/nanochat-residual-runs-stopped-20261116.json`. The old
SpheretNorm/tangent/transport route, its launchers and tests are removed;
historical code is in Git at `d2f106d`. Old sphere checkpoints raise an explicit
retirement error; existing pure-HB parameter/optimizer keys remain compatible.

## Reference inspected

- [nGPT paper](https://arxiv.org/html/2410.01131), section 2.2.2, equations 10/11.
- [Official model.py](https://github.com/NVIDIA/ngpt/blob/main/model.py),
  `Block.justnorm` and both interpolation sites in `Block.forward`, lines 126–157.
  Official code uses normalized h and branch output, absolute-valued per-channel
  alpha, initial effective alpha 0.05, and normalization after interpolation.
- [Official train.py](https://github.com/NVIDIA/ngpt/blob/main/train.py),
  `normalize_matrices`: weights are normalized after optimizer updates in full
  nGPT. We intentionally retain nanochat's weights/optimizer instead.
- Checked 2026-11-17. The anGPT repository named in its paper contains only
  README/license/gitignore, no model source; it is not the implementation reference.

Borrow only normalized-target residual interpolation. Keep nanochat Attention,
RoPE/QK norm, ReLU-squared FFN, original pre-RMSNorm, zero output-projection init,
final norm/head, NTP loss, MuonAdamW groups, warmup and schedule. This is
**nGPT-inspired Heavy-Ball**, not a reproduction of full nGPT.

## Recurrence

Retained `heavy_ball` is unchanged:

```
u = beta*v + gamma*F(h)
h_new = h + eta*u
v_new = u
```

New `ngpt_heavy_ball`, at every Attention and FFN sublayer:

```
z = F(h) / sqrt(max(mean_channels(F(h)^2), force_floor^2))
u = beta*v + gamma*(z-h)
h_new = normalize_RMS(h + eta*u)
v_new = u
```

h starts at RMS=1 (radius sqrt(width)). Every subsequent h is already normalized,
so no redundant h normalization is needed before the interpolation. beta=0
reduces exactly to the scalar nGPT LERP formula with alpha=eta*gamma whenever
the branch is above its zero-safety floor. The independent unit test transcribes
NVIDIA's unit-radius expression and checks the radius-rescaled result.

**Velocity is an ambient Euclidean vector**, not a tangent vector. All layers
share the same coordinate basis. Normalizing h is a model update, not a basis
change. v is not normalized, projected, detached, or transported; carry u, not
eta*u or h_new-h. Radial components of velocity are allowed and affect later
mixing. This is projected Euclidean Heavy-Ball, not Riemannian Heavy-Ball.
No covariance model or approximate-normalization statistics remain.

## Stability and compatibility

- Both momentum modes default to independent token velocity embeddings,
  normal std 0.02. Explicit zero is retained for diagnostics/legacy HB loads.
- Three learned scalars/site: sigmoid beta and softplus gamma/eta. Pure HB keeps
  `(0.5,0.5,1)`. New mode starts `(0.5,0.5,0.1)`, effective alpha=0.05 like nGPT.
- Official `justnorm` divides by zero on nanochat's initial zero branch. We use
  `force_floor=0.1`, clamping squared RMS before rsqrt. The derivative at zero
  is 10, and initial direct force gain gamma*eta/floor is 0.5. This leaves
  original backbone initialization untouched and matches exact normalization
  once force RMS exceeds the floor. There is a clamp kink at the threshold.
- Hidden normalization has only a numerical RMS floor of 1e-6. A precisely zero
  update remains zero (no direction can be assigned); diagnostics check actual
  nonzero hidden norms and finite derivatives. No hard clipping is added.
- FP32 h/v/residual arithmetic, BF16 original branches; FP64 for math checks.
- Token-local depth states only; no additional persistent KV state or cross-token
  statistics. Cached decoding and activation checkpointing carry h/v normally.
- Checkpoints retain model/optimizer/RNG/data position/progress/config. Pure HB
  initialization, state dict names and optimizer membership remain unchanged.

Use `--residual-dynamics ngpt_heavy_ball` (new) or `heavy_ball` (retained).
New-mode option: `--ngpt-force-rms-floor`; existing momentum coefficient flags
remain available. Vanilla remains the overall default. No formal-run launcher
is provided in this directory.

## Validation

The bounded GPU gate checks official nGPT scalar-limit equivalence, zero-force
first/second derivatives, velocity semantics, 256 sublayer recurrence, seed-matched
backbone, compile/checkpoint gradients, CCE/dense agreement, causal/batch independence,
chunk/token KV decoding, a 32-layer model, optimizer/RNG roundtrip and CLI exit85
recovery. Same-card measurements: 32 vanilla steps, 32 HB steps, 128 new-mode steps,
actual hidden/velocity scales, CE, throughput and peak memory. Local JSON/XML only.
Profiler FLOPs are partial; analytic FLOPs exclude scalar special functions.
