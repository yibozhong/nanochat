# SiLU Jensen Gap Attention

User-confirmed first configuration: phi(x)=SiLU(x) with no MLP, alpha fixed at 1.
The nonlinear function has no parameters. The value projection and independent
output projection W_j remain learnable. This is not yet a learned-phi variant.

    mu = A V
    J = A SiLU(V) - SiLU(A V)
    output = W_o mu + W_j J

`--attention-statistic jensen_silu` in `scripts.connection_train` selects this
configuration for vanilla full-attention models. Existing Q/K/V, attention
temperature, RoPE, QK norm and causal support are preserved. Both occurrences
of phi are identical elementwise SiLU. SiLU is not globally convex, so J may
be negative: never clamp it as a variance. No detach or auxiliary loss.

Reuse the moment-attention reference: original efficient BF16 mean attention,
FP32 causal probabilities and both nonlinear terms in a compiled statistics
branch, query tiles256 with activation recomputation, then BF16 projection.
Mathematical attention weights are shared; rounded intermediate probabilities
are recomputed and need not match cuDNN bitwise. No new custom kernel or MLP.

W_j is stored as `attn.c_stat.weight`, shape D x D, zero-initialized without
consuming RNG, using the existing Muon group. Alpha1 and W_j0 preserve the
baseline at initialization and allow nonzero W_j gradients. Existing W_o also
uses the baseline's zero initialization. Shared weights and initial RNG match
exactly. D512 x8 adds2097152 parameters, total60817408, identical to variance.
Statistical-branch matrix FLOPs equal the reference variance branch; SiLU
pointwise cost is additional and excluded from matrix-only FLOP accounting.

Initial scope was implementation and bounded GPU validation. The user subsequently
authorized one formal 10B-token run on2026-11-03; see the launch protocol below.
The two existing second-moment/variance jobs keep their immutable inputs.
Validation25 checks: eight new SiLU tests plus the17 retained checks, including
FP64 formula/VJP, compiled FP32 VJP, positive/negative/zero gaps, nonzero
projection gradients, live2K causality/cache, compile/recompute parity,
optimizer/RNG recovery and config compatibility. Same-card mean/variance/SiLU
profiles (20 warmup+30 measured steps), and25 -> exit85 ->50 fresh-process
recovery with the305176-step horizon. No W&B, credentials or returned weights.

Run `python research/connections/jensen/workflow.py ROOT submit`, then `... ROOT
monitor`. The monitor collects and audits, publishes source/results, and exits.
It never launches formal training. GPU validation is pending until
`validation.json` is published. Reuse existing runtime/data and the single
Hopper>=22GB/public-soft-priority/backfill resource policy.

## Authorized formal training (2026-11-03)

`pretrain.py ROOT submit` starts exactly one `jensen_silu` run from the GPU-tested
7d1b18c immutable bundle, after verifying gate7801189.0 and the published report.
305176 updates x32768=10000007168tokens,2K,micro16/accum1,seed42, original
FineWeb-Edu packed data and MuonAdamW/LR horizon. From scratch, alpha fixed1,
phi=SiLU, W_j zero initialization. Same `nanochat` W&B project and
`attention-moments-10b-20261103` comparison group as the two existing runs.
Latest overwrites every500steps, best remains separate,1800second exit85 durable
transfer/resume. Single Hopper>=22GB, public-slot soft priority/backfill allowed;
80GB scratch and no new data cache. Added persistent checkpoint budget <1GB.
`pretrain.py ROOT monitor` stops after a finite-loss training step; it does not
monitor full10B or submit additional jobs. No existing runs are stopped.
