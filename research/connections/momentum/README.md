# Heavy-Ball with nGPT-inspired residual interpolation

## Current revision: tangent velocity, RMS control and transport

The user authorized fixing velocity scale and transport on 2026-11-17,
including bounded validation. No additional formal run is authorized.
Existing jobs use frozen source. GPU validation of this revision is pending;
results in the historical section below concern the ambient implementation.

New `ngpt_heavy_ball` configs default to `--ngpt-velocity-mode transport_rms`:

```
N(x, floor) = x / max(RMS(x), floor)
P_h(x) = x - mean_channels(h*x)*h       # RMS(h)=1, radius=sqrt(d)
h0 = N(token_embedding, 1e-6)
v0 = N(P_h(velocity_embedding), 0.1)
z = N(F(h), force_floor)
u = P_h(beta*v + gamma*z)              # P_h(z-h) = P_h(z)
w = N(u, velocity_floor)
h_new = N(h + eta*w, 1e-6)
v_new = w - mean(w*h_new)/(1+mean(h*h_new)) * (h+h_new)
```

This is normalized-target Riemannian HB, not exact scalar nGPT LERP at beta=0.
Pure `ngpt` retains exact LERP. Tangent retraction and speed normalization change
the mixed model's finite-step rule. Three learned scalars/site remain: sigmoid
beta and softplus gamma/eta, initialized `(0.5, 0.5, 0.05)`. Gamma weights the
new direction against memory; eta controls step size. Gamma*eta is no longer
nGPT alpha in this mode. The initial angle is at most atan(.05), about2.86 degrees.

Velocity normalization is scalar, without centering or affine gains/biases,
preserving tangency. `--momentum-velocity-rms-floor .1` bounds its zero-region
derivative by10; zero speed remains zero and per-token velocity RMS is at most1.
This is separate from the force floor (1e-6 for nonzero initialization). There
is a clamp kink at the threshold, with ordinary autograd semantics. The
[official HB implementation](https://github.com/gaijingchu/Momentum-Streams-for-Optimizer-Inspired-Transformers/blob/main/formers/hb/model.py)
uses velocity LayerNorm. We borrow scale control, using scalar RMS because
centered/affine LayerNorm need not preserve the sphere tangent space.

Transport is exact parallel transport along the short great-circle arc between
endpoints. Tangent retraction gives dot(h,h_new)>=0, so its denominator is >=1.
Transport preserves tangent-space norms and inner products. Carry w, not eta*w
or the chord h_new-h. All operations differentiate through the entire geometry;
FP32 state, BF16 unchanged branches, O(d) work/storage, no d x d tensor.
See [Boumal's manifold optimization text](https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf)
for sphere retractions and vector transport.

Checkpoints store velocity mode/floor. Missing mode in old checkpoints resolves
to `ambient`, retaining recurrence and optimizer layout. CLI resume rejects
config mismatches; reproduce old runs with `--ngpt-velocity-mode ambient`.
The existing three nonzero-run launch recipes explicitly pin ambient.
Vanilla, pure nGPT and pure HB are unchanged. Token velocity embedding stays
the default; zero is used only for compatibility and diagnostic tests.

`submit_debug.py` submits one bounded diagnostic. Tests cover dense independent
rotation-reference transport, tangent/norm/isometry invariants, first/second
derivatives including zero speed, 256 updates, 32-layer gradients, compile,
checkpointing, CCE, causality/KV decoding and exact optimizer/RNG/CLI recovery.
Production nonzero-init timing:128 new-mode steps plus32 each of vanilla, pure
nGPT and ambient HB, all timed over steps17-32. Local JSON records speed,
tangency and discarded radial energy. Require time <1.35x ambient and <1.75x
vanilla, allocated memory <90% capacity. Timeout2400s/failure debugging900s;
no W&B or automatic formal continuation.

## Historical ambient implementation and validation

Earlier status: bounded GPU validation **passed** on 2026-11-17, job `8837389.0`,
H100 80GB, model commit `22023c3`. No new formal training has been started.
The subsequently authorized pure nGPT residual run (without Heavy-Ball) is
documented separately in [the historical nGPT report](../ngpt/legacy/ngpt-10b.md).
The user redirected this task from anGPT to nGPT before GPU tests. The pending
anGPT debug job was held during environment preparation; no anGPT validation
result is claimed. The interim approximate-moment implementation is archived
in Git at `63a20cc` and removed from the active model.

The five old formal plans in `stopped-runs.json` remain retired. Their summary
is `/home/yzhong239/nanochat-residual-runs-stopped-20261116.json`. The old
SpheretNorm route and its launchers remain removed;
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

## Validated result (2026-11-17)

[Machine-readable report](validation.json): 52 GPU tests passed, no failures or
skips. New mode completed 128 updates / 4,194,304 training tokens with finite
loss/gradients; dev CE 10.398012 -> 5.901073. The 32-layer test, cache/causality,
compile/checkpoint, scalar nGPT limit and zero-force derivative checks passed.
CLI exit85 resume preserved optimizer/model/RNG/config/data position and passed
the exact fixed-gradient next-update check. Independent FA2/CCE backward runs
are not bitwise deterministic; this is recorded separately in the report.

Same H100, batch 16 x 2048, compiled steady steps 17–32:

| Mode | Median step | Peak allocated | Parameters |
|---|---:|---:|---:|
| Vanilla | 67.04 ms | 5.37 GiB | 58,720,256 |
| Heavy-Ball + token velocity | 72.69 ms | 6.53 GiB | 75,497,520 |
| nGPT residual + Heavy-Ball + token velocity | 75.70 ms | 7.79 GiB | 75,497,520 |

The new residual adds 4.14% step time versus HB and 12.91% versus vanilla in this
measurement. The bounded diagnostic is an engineering check; it is not a trained
quality comparison. No formal run or W&B was created.
