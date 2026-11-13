# Final-FFN narrow readout

User-authorized implementation and bounded GPU validation, 2026-11-13.
No new formal training run is authorized by this task. HCA/Dynamic HCA are
retired; their small closeout is in `../archive/hca-closed-20261113/summary.json`.

## Architecture

For the last Transformer block only:

```
x = residual after last attention                         # D
u = ReLU(W_up RMSNorm(x))²                                # 4D, existing FFN
[a, b] = W_down_packed u                                  # D + R, one GEMM
h = x + a                                                # BF16 residual boundary
s = rsqrt(mean(h²) + 2^-23)                               # FP32, D main channels
features = [h*s, b*s]                                    # D + R
logits = W_vocab_packed features                          # V, one GEMM
loss = CE(15*tanh(logits/15), target)                      # unchanged objective
```

The branch reads the expanded FFN features before the original down projection.
It has its own output directions and ordinary end-to-end gradients. It shares
the main residual's normalization scale; including the new channels in that
denominator would change the baseline function even with a zero branch.
There is no additional attention, FFN up projection, vocabulary softmax,
auxiliary loss, or separately materialized sequence-by-vocabulary tensor.

With D=512, V=32768 and R=64, this adds 2,228,224 parameters, including the
64 x 2048 branch down projection. Vocabulary GEMM arithmetic grows 12.5%; whole
model matrix arithmetic grows about 4.4% at 2K/8 layers. Hardware timing must be
measured. Baseline config R=0 preserves its implementation and parameter shapes.

## Initialization

Baseline weights and caller RNG are preserved. New down rows start at zero;
new vocabulary columns start nonzero so the new down rows receive a gradient
on the first update. Both must not start at zero.

`random` uses the original head initialization scale. `gradmax` performs one
bounded, deterministic calibration before the optimizer is created, on at most
256 positions from the first training sequence. It estimates the covariance of
the pre-softcap vocabulary error and the RMS-scaled FFN features, projects out
the existing vocabulary directions, and selects R directions by randomized
SVD with one power iteration. It uses implicit products rather than storing
the full V x 4D covariance. This projected LM-head application is our adaptation
of [GradMax](https://arxiv.org/abs/2201.05125), not a reproduced LM experiment
from that paper. Calibration does not advance training data or RNG, and is
skipped on checkpoint resume.

## Kernels

`nanochat/ffn_readout_triton.py` fuses residual addition, normalization, and
packing into one forward kernel, with one backward kernel computing both
input gradients. Separate D/R tiles avoid padding 576 channels to 1024.
The kernel saves only inputs and one FP32 reciprocal RMS per token; no FP32
activation-sized temporary, atomics, or gradient split/concatenation kernels.
Packed GEMMs use the existing library GEMMs; the full-vocabulary loss keeps
the existing unfiltered [Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy)
Triton implementation. `reference` is an independent PyTorch oracle.

## Configuration and validation

Model fields: `ffn_readout_width=64`, `ffn_readout_backend='triton'`, and
`ffn_readout_init='gradmax'`. CLI flags have the corresponding hyphenated names.
Supported backbone: vanilla baseline, one stream. Width 0 disables the branch.

GPU checks: `python -m pytest tests/test_ffn_readout.py -q`.
The benchmark in this directory compares the same production 8-layer, D512,
2K, microbatch16 configuration, using the locked runtime and existing packed
FineWeb-Edu cache. It records numerical checks, finite training updates,
allocator peak, warm timing, and gradient activity. Diagnostics do not create
W&B runs. See `results.json` for actual validation and performance when ready.
