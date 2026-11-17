# Riemannian momentum residual stream

The original zero-velocity Riemannian 10B run remains frozen and running.
On 2026-11-16 the user additionally authorized exactly three 10B runs:

| Preset | Residual update | Initial velocity | Total parameters (d8/D512) |
| --- | --- | --- | --- |
| `riemannian_embed` | Heavy-Ball, sphere, exact parallel transport | Independent token embedding, tangent-projected | 75,497,520 |
| `heavy_ball_embed` | Euclidean Heavy-Ball | Independent token embedding | 75,497,520 |
| `sphere` | Tangent force plus radial retraction | No velocity state | 58,720,272 |

Each uses the same seed, backbone initialization, data, optimizer rules and
10,000,007,168 tokens. These are token-budget-matched, not parameter-matched.
Each embedding is V-by-D, untied, normal(0, .02), BF16 like the input embedding,
and uses the existing input-embedding AdamW group. There is no additional
absolute position embedding; nanochat keeps RoPE. This is the token-only
adaptation of the paper's velocity token+position embeddings. Velocity is
looked up per token, never carried across tokens or cached as sequence state.
The initial spherical velocity is P_h(E_v[token]); the Euclidean version uses
E_v[token] directly. Neither adds a velocity LayerNorm.

Sphere-only has one positive learned alpha per sublayer, initially .5 (the
combined mechanism's initial gamma*eta), fixed radius sqrt(d), and no momentum,
transport, or depth-decay schedule. It uses the same geometry as the current
combined implementation, rather than changing to the paper's full recipe.
Heavy-Ball retains beta/gamma/eta=(.5,.5,1), u=beta*v+gamma*F(h),
h_new=h+eta*u, v_new=u, with FP32 residual accumulation and unchanged pre-norm.
The full combined option retains the exact transport described below.

Default configuration retains vanilla. `--riemannian-momentum` still loads
legacy zero-velocity checkpoints; new flags are `--residual-dynamics` and
`--velocity-init`. No running source bundle or checkpoint is modified.

The backbone is the existing full-attention vanilla d8/D512 model, not the
native SSSL/extras model. Attention, ReLU-squared FFN, NTP/CCE, optimizer rules,
data order, LR schedule and head are unchanged.

## Geometry and the momentum consistency issue

At each Attention and FFN site, r = sqrt(d), h lies on the sphere, and v is a
tangent vector at h. Input embeddings are projected to that sphere; v starts
at zero independently for every token's traversal of depth.

    g = F(h) - h * dot(h, F(h)) / r²
    u = beta * v + gamma * g
    h_new = r * (h + eta * u) / norm(h + eta * u)
    v_new = u - dot(u, h_new) / (r² + dot(h, h_new)) * (h + h_new)

The last expression is exact parallel transport along the short great-circle
arc. It preserves the norm and inner products of tangent vectors. Retraction
has dot(h, h_new) > 0, so the denominator stays away from the antipodal
singularity. No d-by-d matrices, trigonometric operations, normalized velocity
or division by velocity norm are needed; zero velocity has a smooth backward.

The implementation combines the linear tangent projection as
`u = P_h(beta*v + gamma*F(h))`. For tangent v it is the same recurrence; it also
removes FP32 radial roundoff from the accumulated velocity. P_h uses the actual
dot(h,h), equal to r² on the sphere. Transport then uses the unscaled u. Carrying
eta*u or the chord h_new-h would change momentum when eta varies; simply
projecting u to the new tangent plane would shorten it and add damping.
All state, projections, normalization and transport retain autograd. No detach.

Each site learns three FP32 scalars: beta=sigmoid(b), gamma=softplus(c),
eta=softplus(e), initially (0.5, 0.5, 1). This starts with unit steady-state
gain gamma/(1-beta), rather than amplifying a constant force. Gamma and eta
are partly redundant in their scale; their values are logged by sublayer.
They use the existing connector AdamW group with no weight decay. There are
48 additional scalar parameters at depth 8 for either momentum variant. The original zero output-projection
initialization makes the initial stream an identity on the sphere.

This is a coherent geometric feature-update mechanism, not an exact optimizer
of NTP loss with respect to hidden states: F is a learned feature map, not an
NTP gradient oracle. Tangent geometry resolves the changing tangent plane, but
does not guarantee that different layers' learned updates align semantically.
Fixed-radius propagation removes radial information and step angles approach
90 degrees for very large velocities; this can still affect conditioning.
The identity is on tangent perturbations, not a full Euclidean identity.

## Precision, inference and cost

Hidden and velocity states stay FP32. The existing pre-norm Attention/FFN see
BF16 inputs; the final normalization/head retain their original BF16 path.
Ordinary PyTorch autograd and compiled whole blocks fuse the geometry. FP64
helpers support independent gradcheck. KV caching remains the original cache;
velocity is local across depth, never carried between tokens or batches.
Both states are supplied to activation checkpointing if it is enabled.

Block compilation uses `emulate_precision_casts=True` to preserve the BF16
branch rounding at the FP32 geometry boundary. Default cast elimination caused
an 8.9% relative error in a small scalar-gradient vector despite near-identical
loss; preserving casts reduced the observed maximum parameter-gradient error
below 0.5%, within the unchanged 4% BF16 check. Cached multi-token chunks use
PyTorch's structured `causal_lower_right` bias so Flash SDPA handles the correct
cache-offset causality without materializing an unsupported dense mask.

Cost is O(B*T*d) per sublayer, versus O(B*T*d²) projections. For one site the
geometry has approximately 21*d-5 channelwise FLOPs, plus scalar operations.
Two live FP32 states use 8*B*T*d bytes; saved backward intermediates are extra.
The gate measures single-model peak VRAM and same-card production throughput.
PyTorch profiler's supported-operator FLOP sum is recorded separately and
explicitly marked incomplete for FA2, CCE and fused kernels. The analytical
training cost uses an approximate 2x-forward backward, not hardware counters.

## Validation and authorized run

GPU tests cover FP64 independent rotation/transport, first/second derivatives
including zero velocity, long FP32 norm/tangency, preserved backbone weights,
optimizer membership, compiled/eager/checkpoint gradients, CCE/dense, causality,
batch isolation, chunk/token KV decoding and exact checkpoint/optimizer/RNG.
For each new preset, 36 shared tests plus 8 preset-specific checks cover
embedding gradients/optimizer state, exact backbone initialization, compiled
and checkpointed gradients, causal/KV inference and checkpoint recovery.
Preflight additionally runs 512 finite NTP updates and 32 vanilla timing-only
updates at the production shape (no ablation research run), then a real CLI
exit85/resume check. Diagnostics write local JSON only. A predeclared 1.75x
median step-time ceiling and numerical invariants stop failures before 10B.

The formal process starts afresh at seed42, 305176 updates x 32768 tokens,
microbatch16, T2048, original packed FineWeb-Edu sample-10BT/tokenizer, FA2/CCE.
It never loads diagnostic weights or data position. Latest and best checkpoints
and same-run recovery use the existing 1800-second exit85 transport. GPU-family
validation is tied to the frozen source. Failure stops the pipeline and keeps
a bounded debug allocation; it does not submit another training run.

## Sources and differences

- [Momentum Streams for Optimizer-Inspired Transformers](https://arxiv.org/abs/2605.24425)
  motivates learned momentum streams; this implementation uses heavy-ball,
  without lookahead, velocity normalization or triple-momentum machinery.
- [Sphere Retraction Normalizations](https://arxiv.org/abs/2608.02668)
  supplies the projected tangent/retraction construction. We retain the user's
  fixed radius sqrt(d) and learn site coefficients without importing the
  paper's learned radius or prescribed depth-decay rule.

Their individual results do not measure this combined nanochat architecture.

## Submission

`submit_train.py --experiment riemannian_embed`, `heavy_ball_embed`, or `sphere`
submits exactly one held-audited job for that preset, with per-root locks and
existing-run detection. The same GPU allocation runs the 3600-second bounded
gate, then launches an independent seed42 formal process. Each maintains its
own run ID, checkpoints, source metadata and W&B run. H100/H200 first, A100
after one continuous idle hour with the same gate; no automatic failed-run
resubmission. The source bundle is frozen at submission.
