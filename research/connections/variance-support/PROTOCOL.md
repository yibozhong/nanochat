> Retired on 2026-11-04: all attention-statistics implementations, checkpoints and launchers have been removed. Only ordinary vanilla attention (including the FLA mean-only backend), native and NVIDIA mHC remain. This document preserves historical results and commands; it is not an active experiment.

> Archived on 2026-11-04 by user request. Implementation and launchers have been removed; the text below records the historical experiment only. No automatic continuation.

# Effective Support Gated Variance Attention

One authorized 10B run, `attention_statistic=variance_support`, using the fused
FLA-derived Triton backend. Same vanilla full attention D512 / 8 layers / 4 heads /
context 2048 / micro16 accumulation1 / seed42, FineWeb-Edu packed cache, default
MuonAdamW and 305176-step schedule as the existing Triton variance run. Total
10,000,007,168 training tokens. No additional baseline or sweep.

For each query and head, mu=AV, var=max(A(V²)-mu²,0),
log_support=-log(sum(A²)), g=2*sigmoid(a_h*log_support+b_h),
output=Wo(mu)+Wvar(concat_h(g_h*var_h)). The gate acts BEFORE the dense Wvar:
a headwise scalar cannot be moved through a projection mixing all heads.
No visible-length division, extra norm, detached support, learned alpha or new
Q/K/V. Alpha=1; Wvar is independent, bias-free, zero initialized. a=b=0 starts
exactly at g=1 and matches variance initial outputs and shared weights/RNG.
There are 64 additional slow parameters: 60,817,472 total. Gate vectors use the
existing scalar/connector AdamW rule (lr .008*batch_scale*width_scale, betas
.8/.96, eps1e-10, no decay); matrix and other optimizer groups are unchanged.

The one fused forward maintains sum(P²) with the SQUARED online max-rescale.
log_support=2*log(sum(P))-log(sum(P²)). QKV, dot operands and moment outputs
remain BF16; online softmax, accumulators, and log_support remain FP32, as
required for stable reductions. No extra GEMM or materialized attention matrix.
The backward reuses variance preprocess/dQ/dKV and adds
2*dlog_support*A*(1-exp(log_support)*A) to the existing score adjoint. No direct
support derivative to V. The singleton support adjoint is exactly zero.
The small pointwise gate uses PyTorch compilation and full autograd.

Existing variance BF16 centering/clamping and kernels are unchanged. This adds
concentration information, not a guarantee of quality or useful uncertainty.
Gate learning begins once the zero-initialized variance projection activates.

Validation: 17 existing + 16 new checks. Dense PyTorch formula and derivative,
BF16 shapes through real 2K, individual output VJPs, uniform/concentrated and
singleton support, online pivot rescaling, per-head/value independence, strict
causality, gate placement and gradient flow, initialization, optimizer groups,
compiled/recompute gradients and exact optimizer/RNG restoration. Support max
absolute log error <1e-4; moments relative L2 <.025, QKV/gate VJP <.035; model
CE difference <.002 and parameter gradient error <.02. Thresholds fixed before
GPU results. Same-card two reversed rounds of baseline/variance/new complete
training-step profiles plus operator trace. Report FLOPs, throughput, peak
memory, gate/support values and gradients after 50 updates. Matrix FLOPs equal
variance; reported FLOPs omit support reductions and pointwise gate work.
Real process restart: 25 steps -> checkpoint exit85 -> new process ->50, retaining
305176-step LR horizon. Tests/profile have no W&B credentials or weight returns.

Only successful GPU checks, numerical/profile audit and Git publication release
one formal run to W&B nanochat / group triton-moments-10b-20261103. Source bundles
are immutable. Single complete Hopper >=22GB, public soft priority with backfill
allowed, 2 CPU /16GB RAM, gate40GB / formal80GB scratch. Resume latest overwritten
every500 steps, independent best, exit85 persistent transfer every1800 seconds.
Monitor ends at first finite-loss formal training step. Failure blocks training.
Existing runs, including the separately authorized third-moment run, are untouched.
