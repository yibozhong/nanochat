> Current supported attention statistics: `none`, `variance`, `jensen_silu`. Other statistics are retired as of 2026-11-04. Older modes/results below are historical. This research direction is closed unless explicitly reopened.

# FLA-derived BF16 causal attention and fused moments

Upstream: https://github.com/fla-org/flash-linear-attention/blob/cf9f520fcfef873534132e65e61e12d36b77ab62/fla/ops/attn/parallel.py
MIT license, fixed original source, reference/test/helper context, checksums in
`upstream/`. `generate.py` reproducibly copies the baseline kernels and creates
moment extensions; no FLA package, GLA, gating, or linear-attention substitution.

The baseline uses the original FLA online-softmax forward/preprocess/dQ/dKV
kernels with optional features disabled. One moment forward kernel accumulates
mean and raw E[phi(V)] using the SAME on-chip P and normalization. It writes
mean and centered statistic (or raw second moment). Q/K/V/outputs and tensor
core dot operands are BF16; softmax/LSE/dot accumulators use the standard FA
FP32 precision. This is not the previous standalone FP32 attention path.
Phi(V) is generated on chip and rounded to BF16 for the extra dot. Centering
uses the BF16 output mean and the same BF16 phi evaluation; raw accumulators
remain FP32 until subtracting, avoiding subtraction of two rounded raw moments.
Variance clamps negative roundoff to zero; SiLU gaps retain signs. Alpha is1;
independent Ws/Wj is zero-initialized and uses the original Muon optimizer.

Backward chain rule: dm=do-ds*phi'(mean), draw=ds; variance additionally masks
clipped outputs. Delta=sum(mean*dm+raw*draw), dp=dm V^T+draw phi(V)^T;
dv=P^T dm+phi'(V)*(P^T draw). Both dQ and dKV loops share probabilities across
branches. This retains FLA's three backward launches; it is not a claim that
forward and backward together are a single launch. Saves only O(BTHD) tensors,
no full attention matrix or transformed V cache. BF16 saved raw moments and
rounded dot operands imply approximate derivatives as in ordinary BF16 FA;
independent PyTorch VJP checks quantify errors instead of claiming bit equality.

`--attention-kernel fla_triton` supports fixed-length causal full-attention MHA
for vanilla (`variant=baseline`) with head_dim32/64/128. Native/mHC retain their
existing implementation. No GQA, packed variable lengths, SWA or KV-cache decode
in this initial port; unsupported use raises instead of silently changing the
backend. `original` remains the default for checkpoint compatibility and as an
evaluation reference. All future formal quality comparisons must retrain the
baseline with the same new kernel. No formal training is launched by this gate.

Validation: independent PyTorch formula and Q/K/V gradients at17/129/2048 tokens
and D32/64/128, signed/zero gaps, exact causal prefix and no future VJP influence,
full-model compile/eager and an optimizer step. Predeclared relative L2 bounds:
operator output2.5%, gradient3.5%; full-model compile gradient2%, loss.002.
These bounds do not guarantee accuracy for arbitrarily tiny gaps or extreme
value offsets; constant/first-token cases are checked separately.

Same H200 single-job operator forward+backward timing and kernel trace;
full-model 2K/micro16/accum1,20warmup+30timed steps, forward/reverse order for
baseline/second_moment/variance/jensen, and original baseline/variance/Jensen
profiles. No W&B, no diagnostic checkpoints returned, no new data cache.
Results remain pending until `validation.json` is published. The workflow
collects/audits/pushes and exits, never submits or resumes formal training.
