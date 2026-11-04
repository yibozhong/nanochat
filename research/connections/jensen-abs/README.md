# Absolute Jensen gap attention

`--attention-kernel fla_triton --attention-statistic jensen_abs` computes
`mu = A V`, `J = A abs(V) - abs(mu)`, and `Wo(mu) + Wj(J)`.
Alpha is fixed at 1. Wj is an independent bias-free D x D matrix initialized
to zero and trained with the existing Muon optimizer. Q/K/V, masks, normalization,
and common initialization are unchanged. The default D512 eight-layer model has
60,817,408 parameters, the same as the SiLU/variance variants.

The existing FLA-derived Triton kernel is specialized with `MODE=4`.
One forward launch computes both statistics using shared on-chip attention
weights, without materializing A or abs(V). The analytic backward uses
`dmu = dout - dJ sign(mu)` and `dV = A.T dmu + sign(V) * (A.T dJ)` alongside
the Q/K softmax derivatives. `sign(0)=0` agrees with PyTorch's abs subgradient.
There is one preprocess, one dQ, and one dKV backward launch.

Q/K/V, dot operands, saved moments, and outputs are BF16. Online softmax,
dot accumulation and centering retain standard FlashAttention FP32 accumulators.
Like the other variants, centering uses the returned BF16 mean. No additional
clamp is applied: the exact mathematical gap is nonnegative, but BF16 rounding
can leave small signed errors. The zero kink also makes gradients sensitive
to tiny differences in means; the gate records mean-sign disagreements and
negative-value frequency instead of claiming bitwise mathematical equality.

The compiled PyTorch reference also supports `nonlinearity='abs'`. Tests include
independent FP64/compiled FP32 formula/VJPs, BF16 fused forward and Q/K/V VJPs at
17/129/2048 tokens and D32/64/128, exact cancellation and zero subgradients,
same-sign zero gap, causal prefix/no future gradients, shared initialization,
zero-Wj gradient/Muon membership, compiled full-model updates, and optimizer/RNG
restoration. Together with the 17 existing kernel regressions there are 28 checks.
The output relative L2 bound is <2.5%; full-model compile/eager gradient <2%,
loss difference <.002. The first gate failed one of 28 checks: at T17/D32,
one of 1088 mean signs differed between online BF16 and dense attention,
giving 5.47% Q-gradient and 5.32% K-gradient relative error (2K maximum 2.38%).
That unrestricted dense-VJP failure is retained in `failed-attempts.json`.

The revised abs-only check independently differentiates dense attention at the
sign selected by the fused returned mean, requiring full Q/K/V VJP relative L2
<3.5% without dropping any tokens. It also records the unrestricted dense VJP,
sign-disagreement count/magnitudes, and negative roundoff; it does not assert
uniform <3.5% agreement when the two forwards cross the abs kink. Explicit
zero-subgradient tests remain. Production kernels and all other-mode thresholds
are unchanged. This separates backward correctness from nonsmooth sensitivity;
it does not eliminate that sensitivity or prove it harmless for training.

The bounded H200 gate also traces operator launches, profiles baseline/SiLU/abs
in two reversed rounds (2K, micro16, 20 warmup +30 timed steps), and validates
25 -> exit85 -> fresh process50 with the 305176-step LR horizon. It returns no
weights and records no W&B. This diagnostic gate never submits formal training.
The three existing formal jobs keep their immutable source/configuration.

Workflow root: `/home/yzhong239/jensen-abs-gate-20261103`. Results are published
to `validation.json` after successful audit; submission alone is not a pass.

## Authorized 10B run

After validation completed, the user explicitly requested one formal 10B run.
`pretrain.py ROOT submit` reuses the exact GPU-verified source bundle `7684123`,
the existing Triton training launcher, and the original 2K FineWeb-Edu cache.
The run starts from seed42 with vanilla full attention, 8 layers/D512/4 heads,
micro16/accum1, default MuonAdamW and 305176-step LR horizon (10000007168 tokens).
It uses the `triton-moments-10b-20261103` W&B group in the nanochat project, alongside
the three earlier matched runs. No additional baseline or sweep is submitted.

Root: `/home/yzhong239/jensen-abs-10b-20261103`. Single complete Hopper >=22GB,
public soft priority/backfill allowed, 2CPU/16GB RAM/80GB scratch. Latest saves
overwrite every500 steps, best remains separate, and exit85 every1800 seconds
persists state for resumption with the same W&B ID. `pretrain.py ROOT monitor`
stops after the first finite-loss training step. Source, queue and restart
configuration are audited in the workflow root; do not duplicate submissions.
