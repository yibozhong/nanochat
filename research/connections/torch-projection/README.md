# Ordinary PyTorch connection projection

The user asked whether the slow mHC projection can use ordinary PyTorch.
`projection_backend=official` remains the default. Explicit `torch` and
`torch_compile` alternatives replace only the connection projection and RMS:

```python
flat = x.flatten(-2).float()
out = F.linear(flat, phi) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
```

Both alternatives use PyTorch autograd; neither defines a custom backward.
The compiled option applies `torch.compile(fullgraph=True, dynamic=True)` to
these same operations. Input activations remain BF16 and projection weights
and intermediates FP32. The diagnostic disables TF32 for PyTorch matmuls;
the official kernel retains its own explicit TF32 rounding. Consequently this
compares the same mathematical map at different arithmetic precision, not
bitwise equality or a lower-precision speedup. It does not change backbone
precision, the model's parameters, initialization or optimizer.

The choice also applies to the final stream head. Evaluation bypasses the
official fused preprocessing operation when a PyTorch backend is selected,
so no_grad does not silently revert to the original projection. Sinkhorn,
coefficient generation, stream read/write and expansion remain official ops.

The bounded same-GPU experiment first runs the existing mHC tests and new
FP64 output/input-gradient/weight-gradient comparisons, official-kernel
comparisons, seed equality, real 2K causality, cache, CCE and model-gradient
checks. It then measures the real 32768-token projection shape with 10 warmup
and 30 event-timed iterations per backend. Finally it profiles vanilla,
official mHC, eager PyTorch mHC and compiled PyTorch mHC in separate processes
on the same GPU, using the existing 66-update profiling protocol. Every run
records to `nanochat`, group `mhc-torch-projection-20261030`. No checkpoint or
1B pretraining is created.

First attempt `7547152.0`, source `bbc1da3`, passed 19 checks and failed the
compiled model's variable-length cache check: static shape compilation hit
Dynamo's eight-version recompilation limit. It never reached timing. The
retry uses dynamic shape compilation with unchanged projection mathematics;
the failure archive remains in the workflow results directory.

After validation, an explicit training selection is available as
`--projection-backend torch` or `--projection-backend torch_compile` in
`scripts.connection_train`; existing commands retain the official path.

## Results

Retry `7547170.0`, source `ab693dc39574950e929c3ea4dbed8505b8fef69c`, exited 0
after one execution. **20 tests passed**, none failed or skipped. The numerical
checks against FP64 had maximum output relative-L2 error 2.04e-7 and weight
gradient error 1.64e-7. Input gradients compared with the BF16-cast FP64 oracle
had maximum relative-L2 error 3.97e-5. Differences against the official path
were at most 2.08e-4 for output and 1.05e-3 for either gradient. These are
tolerance checks, not bitwise equality.

All four timing trials ran sequentially on one RTX PRO 6000 Blackwell Server
Edition (SM120/188 SMs), UUID `7883cf28-2434-c0bd-78d6-ed843c2845bf`.
Each made 66 diagnostic optimizer updates, 2,162,688 tokens; the 20 ordinary
timed updates follow 20 warmup updates and exclude profiler instrumentation.
All five successful W&B runs were independently verified. The separate failed
attempt remains recorded and contributes no training tokens.

| Full model | Mean ± sample SD, ms/step | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Vanilla | 80.588 ± 0.859 | 5.375 | 5.389 |
| Official mHC | 152.170 ± 0.434 | 8.280 | 8.475 |
| PyTorch eager projection | 169.970 ± 0.409 | 13.248 | 13.420 |
| PyTorch compiled projection | **129.410 ± 0.319** | 12.629 | 12.836 |

Eager PyTorch is **11.70% slower** than the official implementation. Compiled
PyTorch reduces step time by **14.96%** (22.76 ms), but remains **60.58% slower
than vanilla**, versus 88.82% for official mHC. It also increases reserved
memory by 4.36 GiB. Memory is PyTorch allocator usage in the ordinary timed
region, not whole-device peak; it does not establish compatibility with smaller
GPU architectures. Trials may reuse compiler caches, so startup-inclusive wall
times must not be interpreted as steady throughput.

The isolated projection measurement at the real `[16,2048,4,512]` activation
shape and `[24,2048]` weight shape gives:

| One projection invocation | Forward ms | Backward ms | Total ms |
| --- | ---: | ---: | ---: |
| Official | 0.248 | 2.649 | 2.897 |
| PyTorch eager | 1.118 | 2.697 | 3.815 |
| PyTorch compiled | 0.584 | **0.869** | **1.453** |

These are 30 CUDA-event samples after ten warmups. Forward includes forming a
scalar probe loss; both intervals include stream gaps. The backward includes
the whole projection/RMS derivative, not just a GEMM. A single isolated-call
measurement must not be confused with the earlier sum of 17 projection
backward kernels per training step.

The compiled path cuts isolated backward time by about 67%, while its forward
is slower. In full-model event samples, forward increases from 26.56 to
32.88 ms, while backward drops from 116.95 to 88.06 ms. Eager has many separate
FP32 multiply, add, square and conversion kernels: the trace shows that replacing
the GEMM without fusing its surrounding work does not improve total performance.
The compiled path fuses that surrounding work using PyTorch-generated kernels;
there is still no hand-written backward.

Trace audit confirms **zero official projection kernels** in both PyTorch
trials, including the final head. The other official connection operations
remain present. FP32 SGEMM kernels appear in the new paths; these measurements
did not enable TF32 or lower the projection precision to obtain a speedup.
They do not establish the best achievable speed at matched TF32 precision.

This experiment validates a usable optional implementation, not the paper's
6.7% overhead or long-run model quality. All three mHC choices retain 59,515,317
parameters and the same initialization, data, batch and optimizer. Default
selection remains official; the alternatives are explicit flags.

- [Full measurements](results.json), [numerical and trace audit](audit.json).
- Raw traces/operator averages/logs:
  `/home/yzhong239/mhc-torch-projection/results/gate-7547170.0.tar`.
  Archive and per-trace SHA-256 values are in `audit.json`.
- [Successful parent](https://wandb.ai/yibozhong657-none/nanochat/runs/9a4d1afb),
  [vanilla](https://wandb.ai/yibozhong657-none/nanochat/runs/947244fc),
  [official mHC](https://wandb.ai/yibozhong657-none/nanochat/runs/77157433),
  [eager](https://wandb.ai/yibozhong657-none/nanochat/runs/c3b88aae),
  [compiled](https://wandb.ai/yibozhong657-none/nanochat/runs/b8d2c948).
- [First failed attempt](https://wandb.ai/yibozhong657-none/nanochat/runs/e1614753),
  archive `gate-7547152.0.tar` in the same workflow directory.
