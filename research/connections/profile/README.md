# mHC overhead diagnosis — 2026-10-30

The largest measured bottleneck is the pinned official connection-projection
backward kernel: **42.55 ms/update**, about **62% of the additional GPU kernel
time**. Its launch has only **16 CUDA blocks** on a GPU with **188 SMs**.
Sinkhorn costs 3.35 ms/update. Removing the manifold constraint leaves the
dominant bottleneck intact. Using official operators did not reproduce the
paper's optimized training system or its 6.7% overhead. These measurements
diagnose our implementation and shapes, not an inevitable cost of mHC.

## Measurement and audit

Job `7547053.0` exited 0 after one execution, source
`89969596b9f43101f9fd3dd6196ca79c59e6a92c`. All variants ran sequentially on the
same RTX PRO 6000 Blackwell Server Edition, SM120, 188 SMs, GPU UUID
`c651d100-2cec-b117-23c2-52c709249570`. Fixed FineWeb-Edu data, seed42,
D512/eight layers/2K/full attention, four streams, micro16/accum1, native
MuonAdamW and LR schedule were unchanged. Each variant made 66 diagnostic
updates (2,162,688 tokens), without saving checkpoints or launching full
pretraining. All four W&B runs finished and were verified through the API.

Each variant used 20 warmup updates, 20 ordinary timed updates, 20 updates
temporarily omitting finite-check, four CUDA-event phase samples and two
CPU/CUDA profiler samples. Parameters and gradients were independently checked
for finiteness after the temporary control. Production finite-check, model,
kernel and optimizer implementations were unchanged. The finite-check control
follows ordinary timing sequentially: this is an approximate diagnosis, not a
repeated randomized benchmark.

| Variant | Ordinary ms/update, mean ± sample SD | Finite-check-off ms/update |
| --- | ---: | ---: |
| Vanilla | 80.724 ± 0.065 | 79.255 |
| mHC | 152.561 ± 0.125 | 148.656 |
| No manifold | 149.310 ± 0.283 | 145.333 |

mHC adds 71.837 ms, or 88.99%. Removing Sinkhorn saves 3.251 ms, 2.13% of
mHC step time, leaving no-manifold 84.96% slower than vanilla. This independently
agrees with the prior 50-step measurement (2.22% saved, 85.57% slower).

## Where the time goes

These are **GPU kernel durations**, averaged over two profiled updates. They
exclude launch gaps and identify large additional operations; they are not an
exhaustive decomposition of the uninstrumented wall-time difference.

| Additional operation in mHC | GPU ms/update |
| --- | ---: |
| Projection backward GEMMs plus input/RMS gradient | **42.550** |
| Stream read, forward + backward | 6.689 |
| Stream write/residual mixing, forward + backward | 7.839 |
| Separate BF16 input-gradient merge | 4.528 |
| Sinkhorn, forward + backward | 3.346 |
| Projection forward GEMM | 1.828 |
| Separate projection normalization kernels | 0.657 |

Total kernel time is 77.812 ms for vanilla and 146.471 ms for mHC, a difference
of 68.658 ms. Projection backward alone is 61.97% of that difference. The
no-manifold projection backward still costs 42.542 ms. Attention is effectively
unchanged (8.071 vs 8.099 ms), as is fused vocabulary loss (40.050 vs 40.051 ms).
Vocabulary loss is a substantial baseline cost but does not explain this gap.

CUDA-event phase samples provide a separate check. They include stream gaps;
do not add these values to the kernel table.

| Phase | Vanilla ms/update | mHC ms/update |
| --- | ---: | ---: |
| Forward | 18.826 | 26.608 |
| Backward | 56.382 | 117.380 |
| Finite-check | 1.353 | 2.058 |
| Optimizer | 3.531 | 5.880 |

The optimizer interval includes host dispatch; profiler operator accounting
gives about 3.01 vs 3.10 ms of device work in the optimizer. Omitting
finite-check saves 1.47 ms for vanilla and 3.90 ms for mHC overall, far below the
71.84 ms gap. The apparent 102 ms of CPU time inside mHC finite-check mostly
waits for queued backward work. It is not time spent computing the check.

## Concrete implementation causes

1. **Insufficient parallelism in projection backward.** The pinned
   [official kernel](https://github.com/deepseek-ai/TileKernels/blob/36d9e45d38e204ebb87e6f6e833821eee0482fe5/tile_kernels/mhc/norm_fn_kernel.py#L208)
   `_mhc_pre_norm_fn_bwd_mul` launches `n_rms_group × ceil(rms_group_size/128)`
   blocks. Here these are 1 and 2048/128: the trace reports grid `[1,16,1]`.
   Each block serially traverses 32768/128 = 256 token tiles while accumulating
   the projection-weight gradient and computing the input gradient. There are
   17 launches/update: attention and MLP connections in eight blocks, plus the
   final head, averaging 2.50 ms/launch. Sixteen blocks can occupy at most
   sixteen of 188 SMs at a time. This is a parallelism limit, not a hardware
   occupancy measurement or a claim of proportional achievable speedup.

2. **Four streams incur substantial memory traffic.** The official read and
   post/residual kernels already contain fusion, yet combined forward/backward
   costs 14.53 ms at our shapes. This remains after removing Sinkhorn.

3. **Our autograd integration leaves a separate gradient merge.** Trace
   correlation links all 17 large BF16 additions/update to
   `autograd::engine::evaluate_function: MHCPreNormFnBackward`, 4.53 ms total.
   We set `fuse_grad_acc=False`: official fused mode requires an external
   `grad_from_mhc_post` storage buffer that the current adapter does not install.
   Our integration therefore does not reproduce that fused gradient path.
   The 4.53 ms is measured addition time, not a guaranteed saving from simply
   switching the flag; buffer ownership and graph dependencies require care.

## Why the paper's 6.7% is not the applicable expectation

The [paper, Section 4.3](https://arxiv.org/html/2512.24880v2#S4.SS3), reports
6.7% for its optimized large-scale training system. It describes tuned fused
projection/RMS kernels, selective recomputation and communication overlap.
It explicitly excludes the projection GEMM/RMS scan from the kernels implemented
with TileLang. We use the older public TileLang projection implementation with
complete autograd wrappers; this is not demonstrated to be that performance path.

The [model table](https://arxiv.org/html/2512.24880v2#A1.T5) uses 3B/9B/27B MoE
models, widths 1280/1920/2560 and 4K context, versus our 59M dense D512/2K model.
The denominator and operation shapes differ. We did not measure a size or
hardware sweep, so cannot quantify their separate effects or promise that
changing size reaches 6.7%. The inspected description does not specify the
exact hardware/microbatch underlying that percentage. Our single GPU has no
pipeline communication; missing DualPipe overlap cannot explain the local
42.55 ms projection kernel.

The first optimization target is projection backward's token-axis parallelism
and reduction, followed by stream traffic and fused gradient integration.
Deleting Sinkhorn targets only a small fraction of current cost. No such
kernel or training-path changes were made in this diagnosis.

## Reproducible records

- [Full results and timing samples](results.json), [derived attribution](analysis.json).
- Raw archive: `/home/yzhong239/mhc-profiling/results/gate-7547053.0.tar`,
  containing each variant's `trace.json.gz`, `operator-averages.json`, config,
  summary and logs. Archive/trace SHA-256 values are in `analysis.json`.
- Recompute with standard Python:
  `python research/connections/profile/analyze.py ARCHIVE OUTPUT.json`.
- [Parent audit](https://wandb.ai/yibozhong657-none/nanochat/runs/797e6438),
  [vanilla](https://wandb.ai/yibozhong657-none/nanochat/runs/522bfe32),
  [mHC](https://wandb.ai/yibozhong657-none/nanochat/runs/4038ffef),
  [no manifold](https://wandb.ai/yibozhong657-none/nanochat/runs/d352ebe0).

Profiler spans are 82.91/158.00/154.80 ms, larger than ordinary timing; use the
ordinary table for throughput. A timeline with some GPU kernel active does
not imply all SMs are occupied. This bounded diagnosis does not assess
long-run model quality.
