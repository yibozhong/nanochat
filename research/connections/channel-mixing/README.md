# Channel-local signed residual mixing

`--variant mhc_channel_no_sinkhorn` adds a vanilla-only option. Existing vanilla,
native, NVIDIA mHC and shared-matrix no-Sinkhorn models are unchanged.

For every token and channel, take its four residual stream values `x_c`:

```
v_c = x_c / sqrt(mean(x_c**2) + float32_epsilon)
M_c[i,j] = bias_c[i,j] + mix_scale * sum_r W_c[i,j,r] * v_c[r]
S_out[c,j] = sum_i S[c,i] * M_c[i,j] + H_post[j] * F(H_pre @ S)[c]
```

Each attention/MLP connector owns independent `W[channels,4,4,4]`,
`bias[channels,4,4]`, and one learned scale. Matrix axes follow NVIDIA's
input-stream/output-stream convention. Initial weights are N(0,0.5²), bias is
identity, and scale is 0.01. There is no Sinkhorn, softmax, coefficient clipping,
or stochastic-matrix constraint. The four-value RMS normalization affects only
the coefficient generator. Residual values themselves are not normalized here.

NVIDIA's original global H_pre/H_post generation, aggregation, branch pre-norm,
attention/MLP and final learned stream collapse are retained. Their initial
parameters and RNG sequence match the shared no-Sinkhorn model. New parameters
use the existing connector AdamW group, without changing LR/weight decay.
The initial local implementation explicitly rejects native and TE's shared
fused-gradient-buffer mode instead of silently altering those paths.

The Triton forward fuses normalization, coefficient generation, local mixing,
and branch distribution. Backward analytically differentiates both direct and
input-dependent coefficient paths, including normalization, and recomputes the
coefficients. It never materializes or saves [B,T,C,4,4] matrices. BF16
activations and FP32 coefficients/reductions match the existing mHC precision
policy. Backward uses a temporary deterministic partial parameter-reduction
workspace (about 40.5 MiB at B16/T2048/C512); it is not a saved forward activation.
Second-order derivatives through the custom backward are not supported.

GPU validation compares all input/parameter VJPs against independent FP64
PyTorch, tests zero/tiny/ragged inputs, locality, initialization, 2K causality,
decoding, CCE, activation recomputation and optimizer/RNG restore. It profiles
vanilla, original mHC, shared no-Sinkhorn and local no-Sinkhorn on the same GPU,
and separately compares a single connector with compiled PyTorch. A fresh
process resumes at step25 and completes128 steps using the full10B LR horizon.
These checks do not establish long-run stability or quality.

After validation and push, one authorized10B run uses the existing 2K FineWeb
cache, seed42, microbatch16/accum1, original FA3 and default MuonAdamW settings.
Only formal training logs to W&B. Latest checkpoints overwrite every500 steps;
best is separate; durable scheduler checkpoints occur every1800 seconds.
Monitoring checks every300 seconds and exits at the first finite training step.
