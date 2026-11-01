# Depth-wise Fast Weight Residual (DFWR)

DFWR replaces **both** additions in every full-attention vanilla Transformer
block. Attention and MLP still compute ordinary pre-norm branches. No ordinary
Block.forward is called, and no branch or previous hidden is added to the readout.

For each sublayer, compute delta=F(h) and q=k=RMSNorm(W_k h). W_k is one shared
slow projection, with no separate query or value projection. Each (batch, token)
has its own runtime gate/up/down matrices and momentum buffers. They persist
across all16 sublayers of the8-layer model, and reset at the next forward pass.
Causal attention is the only path connecting token positions. Cached decoding
also initializes the new token's own FW state; it never inherits another token's
FW state. Initial implementation uses a configurable512->8->512 SwiGLU.

The inner objective per state is0.5 times the sum of squared output errors:
prediction=FW_old(k), target=stop_gradient(prediction)+delta. It is **not** an
auxiliary outer loss. `binding_gradient` calculates all three partial matrix
gradients analytically, treating key/target as fixed arguments to the inner
derivative while retaining their outer computational paths. In particular,
`prediction-(prediction.detach()+delta)` must not be simplified to `-delta`:
the forward values agree but the outer derivative does not. Delta is attached.

Inner Muon is functional EMA momentum(beta=.95), Nesterov direction,
five Newton–Schulz steps with coefficients(3.4445,-4.7750,2.0315), rectangular
shape scaling, then a positive learning rate. The three rates start at.01 and
their logs are shared meta-learned scalars. LR is applied **after** NS. There is
no in-place optimizer step, state detach, hidden residual, gradient clipping,
inner weight decay, or post-update weight normalization. Initial implementation
uses FP32 FW states/NS for stability; FP64 is used for mathematical tests. The
config exposes BF16 NS, but the initial formal protocol keeps it off.

The analytic SwiGLU update follows the structure of
[VideoChat3 FastWeightSwiGLU](https://github.com/lululu39/VideoChat3/blob/004d5eff3fbd20c80a5bdbb60f838825c8cb683e/xtuner-videochat3/xtuner/v1/model/compose/videochat3/modeling_vision_lact.py).
Momentum/Nesterov and NS orientation follow
[MiniWorld muon.py](https://github.com/lululu39/MiniWorld/blob/4e043041341038bfc9933ad3e052e206725e99b7/miniworld/muon.py).
VideoChat3's inner update has no momentum and normalizes fast weights after
updating FP32 master weights. Those choices are not copied: the user's corrected
DFWR specification requires momentum and the stated functional matrix update.
Its optional modified/clipped backward is also not used.

To retain the token embedding without a hidden bypass, initialization explicitly
binds h0 into W_down. If phi is the initial SwiGLU feature at k0, write
W_down += (h0-FW_initial(k0)) outer phi / (||phi||²+1e-6), then read h0 entirely
from this bound FW. Gate/up/down initial parameters remain shared meta-learned
slow weights; only their expanded runtime copies evolve with depth. Momentum
starts at zero. This binding is differentiable and is checked for reconstruction
error. It is an initialization operation, not a per-layer residual addition.

Shared embedding/backbone matrices use vanilla's seed42 initialization except
attention/MLP output matrices, which use small nonzero std0.1/sqrt(fan_in).
Vanilla's zero outputs would give a zero initial inner direction, where NS norm
normalization is extremely sensitive. The difference is recorded, not presented
as bitwise matching every baseline parameter.

The outer objective is unchanged full-token/global-vocabulary NTP CE, with the
original final RMSNorm, D->V head, softcap15 and CCE. It differentiates through all
analytic inner gradients, momentum updates and NS iterations. Transformer
parameters, shared W_k, initial FW matrices and inner log rates all receive outer
gradients. Outer optimizer remains nanochat MuonAdamW: matrix parameters use Muon,
the head/embedding retain their groups, and log inner rates use unembedding AdamW
without decay. There are no per-layer learned FW initial matrices.

For one2K sequence, full D512 bottleneck512 needs12GiB for just live FP32 FW and
momentum; retaining states across depth is infeasible on a22GB budget. Bottleneck8
needs192MiB per live state. With checkpointing each two-sublayer block, saved input
states have a conservative state-only estimate1.6875GiB at microbatch1. Activations,
recomputation, gradients, model optimizer and allocator storage are extra and must
be measured. Microbatch16 would already require27GiB for those checkpoint input
states alone. Start microbatch1/accum16 to retain batch32768.

The GPU gate covers independent FP64 inner and outer autograd oracles, finite
differences with frozen stop-gradient anchors, token/example state isolation,
momentum carry and immutability, initial embedding binding, no-residual routing,
real2K causality/cached decoding/reset, gradients to all slow parameters, CCE,
compile/recomputation, optimizer ownership and exact optimizer/RNG restoration.
Then profile baseline/DFWR on the same GPU and run25 steps, exit85, and restore in
a fresh process to50 using the full305176-step LR horizon. Tests/profiles never
log to W&B and diagnostic weights stay on the worker. Gate failure keeps the
formal job held; it must not silently change state sharing or truncate gradients.

Formal target retains the established FineWeb-Edu10B packed cache, seed42 and
305176steps (10,000,007,168 model tokens), default outer LR schedule, dev/test and
single-GPU setup. Group `dfwr-10b-20261101`, project `nanochat`. One formal run;
historical vanilla is the reference. Latest overwrites every500 steps, dev-best
is separate, and exit85 every1800seconds durably transfers checkpoints; restart
uses the same W&B ID and exact data position. Runtime FW state is reconstructed
per forward and is not a cross-sequence training checkpoint state.

No quality/efficiency benefit is established. A single nonlinear update need not
fit the incremental target, and finite FW capacity does not guarantee retention
of all earlier-layer information. GPU memory and stability gate results are
required before releasing the full training run.
