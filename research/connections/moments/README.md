> Current supported attention statistics: `none`, `variance`, `jensen_silu`. Other statistics are retired as of 2026-11-04. Older modes/results below are historical. This research direction is closed unless explicitly reopened.

# Attention value moments, 2K / 10B

The experiment tests whether explicitly retaining value dispersion improves NTP
over mean-only attention. It does not claim a larger LM-head Jacobian rank,
calibrated uncertainty, or that standard Transformers cannot learn moments.
Attentive Statistics Pooling (Okabe et al., 2018,
https://arxiv.org/abs/1803.10963) is a related mean/standard-deviation pooling
precursor; novelty of this causal LM configuration is not established.

Two new runs, each 305176 updates x 32768 tokens = 10000007168 tokens:

- `second_moment`: original Wo * mean + Ws * E_a[v^2].
- `variance`: original Wo * mean + Ws * (E_a[v^2] - E_a[v]^2).

Alpha is fixed at one, Ws starts at zero, no bias, extra normalization, learned
gate, or additional Q/K/V projections. Moments are computed per head/channel,
then concatenated for a D x D output projection. Both models have 60817408
parameters, 2097152 more than the 58720256-parameter vanilla baseline. New
matrices use existing Muon groups; old LR/weight-decay scaling is unchanged.
All shared initial parameters and initial RNG state match vanilla exactly.

Reuse the completed 2K vanilla 10B baseline (7562105.0, W&B 6a07f085), rather
than starting an extra training run. Same seed42, full-attention 8 blocks,
D512/H4, RMS/QK norm, RoPE, ReLU-squared MLP, softcap15, RustBPE32K, original
FineWeb-Edu packed train/dev/test and 10B LR horizon. New 2K cache splits are
identical to the original data. Micro16, accumulation1, no block checkpointing.
Single-seed effects are descriptive; extra parameter capacity is a confound
against vanilla. The two new branches isolate centering at identical parameters.
Do not tune on test. Compare matched-token dev curves and final test losses,
as well as measured wall time; reference-kernel throughput is not an optimized
architecture efficiency claim.

## Precision-first implementation

The original BF16 cuDNN mean path remains unchanged. The new branch recomputes
the same mathematical causal softmax in FP32 and evaluates the moments in FP32,
including centering, before casting the statistic to BF16 for Ws. Thus these
branches share mathematical weights, not bitwise-identical rounded kernel
intermediates. FP32 negatives caused by cancellation are clamped to zero;
the gate records raw negative frequency/magnitude and error against FP64.
No gradient detach and no auxiliary objective.

Compiled PyTorch query tiles (256 queries, all causal keys) plus non-reentrant
checkpointing bound the temporary attention matrix memory. This does not truncate
attention, add recurrent chunks, or change causality. All sublayer and backbone
parameters remain trained jointly with original NTP. Prefix KV caches reuse
the original K/V; no squared-value cache is required. Uniform batch cache
positions follow the existing engine convention.

This first version recomputes QK/softmax and variance's mean; it costs more
than an eventual shared-softmax fused moment kernel. The matrix FLOP estimate
explicitly includes dense masked statistics matmuls and one recomputation.
Pointwise operations, optimizer and softmax are excluded. No custom efficient
kernel will be developed before a quality signal.

## Validation and execution

`python research/connections/moments/workflow.py ROOT submit` submits one GPU
gate only. `... ROOT monitor` collects and audits it, publishes the validated
code/report, submits the two formal runs once, and stops at evidence that both
have executed a finite-loss training step. Existing stopped experiments are
never resumed. STOPPED.json and submission-intent files prevent duplication.

Gate: 14 new checks plus 3 retained initialization/resume regressions, FP64
weighted-central-variance and gradient oracle, compiled FP32 oracle, live 2K
causality and cached decoding, nonzero Ws gradient from zero initialization,
compiled BF16 full-model gradient/recompute parity, exact optimizer/RNG restore.
Same-GPU two-order full-step timing for mean-only/second-moment/variance, each
20 warmup + 30 measured updates. Both new branches also do real process
25 -> exit85 -> 50 recovery with the full 305176-step LR horizon.

Tests/profile do not receive W&B credentials, log to W&B, or return weights.
Formal jobs use `yibozhong657-none/nanochat`, group
`attention-moments-10b-20261103`; latest overwrites every500, best separate,
exit85 every1800 seconds for durable transfer/resume. Public slots have soft
priority, backfill allowed. Single full Hopper >=22000MiB; measured reserved
memory must fit 90% of that minimum. Home budget for two final checkpoint
archives <3GB, no new staging datasets. GPU validation is pending until the
workflow publishes `validation.json`; submission alone is not a passing gate.
