# Official mHC kernels in nanochat

HC was removed at the user's request on 2026-10-30. Its code remains in Git tag
`hc-archive-2026-10-30`; the old measurements under `archive/` describe the
retired PyTorch reference implementation, not this kernel implementation.

The active variants are `baseline`, `mhc` and `mhc_no_manifold`. They retain the existing full
attention backbone: eight layers, D512, four heads, 2K context and 32K vocabulary.
The baseline has 58,720,256 parameters; mHC has 59,515,317. The default is
microbatch 16, accumulation 1, no whole-block activation checkpointing and
32,768 tokens/update. Native nanochat GPT/SSSL remains unchanged.

An explicit [ordinary PyTorch projection option](torch-projection/README.md)
has also been validated: `--projection-backend torch` or `torch_compile`.
Only projection/RMS and its automatic backward change; other mHC operators
remain official. Same-GPU step time was 152.17 ms for official, 169.97 ms for
eager and 129.41 ms for compiled PyTorch. Compiled saves 14.96% step time but
uses 12.84 vs 8.47 GiB reserved and remains 60.58% slower than vanilla.
The default remains `official`; these FP32 alternatives do not reproduce
the official projection's explicit TF32 rounding.

## Removing the manifold constraint

`mhc_no_manifold` is the requested unconstrained hyper-connection ablation.
Its only architectural change from current mHC is to bypass Sinkhorn and feed
the raw residual mixing coefficients directly to the official `mhc_post` kernel.
Coefficients can be negative and their row/column sums are not constrained.
Projection, pre/post sigmoid gating, final head, four streams, all parameter
initial values, optimizer and LR settings remain identical to mHC. In particular,
the identity residual bias becomes a raw identity bias rather than Sinkhorn logits.
This is not the previously archived LayerNorm/tanh HC parameterization.

Both training and inference bypass Sinkhorn. The fused mHC inference operation
contains Sinkhorn internally, so this ablation uses the official individual
projection/split/read/write kernels during inference as well. No kernel source
is modified. Tests compare outputs and every input/parameter gradient to the
official PyTorch reference with its Sinkhorn operation omitted; they also
exercise signed mixing, 2K causality, cache/CCE, identical mHC parameter seeds,
and ensure neither the explicit nor fused Sinkhorn path is called.

The bounded comparison runs vanilla, mHC and no-manifold for 50 updates each on
one physical GPU, at the existing 32,768-token batch and seed42. Steady steps
11–50, dev loss and peak memory will be reported together. W&B remains project
`nanochat`, group `mhc-manifold-ablation-20261030`. This does not launch a 1B run.
The ablation completed successfully in job `7547005.0`, one execution, source
`7812450`. **18 tests passed**, zero failures/errors; 10 native FA3
comparison tests were skipped on SM120. All four W&B runs were verified through
the API. The collector checked identical data, seed, batch/LR protocol and mHC /
no-manifold parameter counts (59,515,317). The official source-file hashes
matched; GPU used the native CUDA 13.1 driver on RTX PRO 6000 Blackwell.

| Variant | Step-50 dev CE | Steady seconds/step | Peak PyTorch reserved GiB |
| --- | ---: | ---: | ---: |
| Vanilla | 6.496122 | 0.080279 | 5.389 |
| mHC | 6.498739 | 0.152356 | 8.475 |
| No manifold | 6.536396 | 0.148973 | 8.443 |

Removing Sinkhorn reduced steady step time by **2.22%**, leaving the
no-manifold variant **85.57% slower than vanilla**. Its reserved memory was
only 32 MiB lower. Thus most of the measured overhead remains without the
manifold projection in this implementation and model size. The subsequent
[same-GPU operator profile](profile/README.md) identifies the main cost:
42.55 ms/update in the official projection backward with only 16 CUDA blocks,
plus 14.53 ms for stream read/write and 4.53 ms for separate gradient merges.
Sinkhorn itself costs 3.35 ms. The projection backward accounts for about 62%
of added GPU kernel time; current official-operator integration does not
reproduce the paper's optimized training path or its reported 6.7% overhead.
No-manifold dev CE was 0.037657 higher than mHC after 50 steps;
this is a single short run, not a long-run quality conclusion.

Timings average steps 11–50; no steps within that range were filtered. Per-step
standard deviations were 0.208/0.296/0.285 ms respectively. Trials run
sequentially and can reuse compilation caches, so their total startup-inclusive
wall times must not be interpreted as steady training speedups. Memory refers
to PyTorch reserved memory rather than all GPU allocations. Each variant saw
1,638,400 training tokens and the same 131,072-token dev set; test was not used.

- [Audited results](no-manifold/validation.json), [kernel errors](no-manifold/kernel-errors.json),
  [all measured step times](no-manifold/timings.json)
- [Correctness run](https://wandb.ai/yibozhong657-none/nanochat/runs/3e3e6eea),
  [vanilla](https://wandb.ai/yibozhong657-none/nanochat/runs/91c75be2),
  [mHC](https://wandb.ai/yibozhong657-none/nanochat/runs/32cc59fd),
  [no manifold](https://wandb.ai/yibozhong657-none/nanochat/runs/7f4ce283)

Run this variant with `python -m scripts.connection_train --variant mhc_no_manifold
--steps 50 --data-dir data --output-dir outputs/no-manifold` after activating the
locked runtime/toolchain described below. No full 1B training was submitted.

## Official code used directly

With the default `official` projection backend, mHC computation calls the unmodified functions in
[DeepSeek TileKernels at 36d9e45](https://github.com/deepseek-ai/TileKernels/tree/36d9e45d38e204ebb87e6f6e833821eee0482fe5).
This pinned official release retains the complete prenorm forward/backward and
PyTorch autograd wrappers. Later v2 releases removed those modeling wrappers
and moved the prenorm GEMM into DeepGEMM. We use the complete earlier API rather
than mix incompatible releases.

The imported operations cover embedding expansion, projection plus RMS scaling,
pre/post/comb coefficient generation, 20-iteration Sinkhorn with its custom
backward, stream read/write, and final stream head. Evaluation uses the official
fused pre-processing kernel. The default four streams, post multiplier 2 and
epsilon 1e-6 match the official V4 configuration. No locally implemented kernel
or silent PyTorch fallback is used.

The nanochat adapter disables the official optional `fuse_grad_acc` mode because
it assumes an externally installed storage gradient buffer that nanochat does
not own. Ordinary PyTorch gradient accumulation is used instead. It also makes
the head projection slice contiguous, pads its projection to 24 rows as the
official head does, and uses `n_splits=1`, which is the actual split count
supported by this release's TileLang projection kernel. These are integration
choices; upstream kernel and autograd source files are unchanged.

An actual SM120 run exposed a launch-size limit: the official Sinkhorn backward
wrapper requests a 32-token tile, using 100 KiB of shared memory at 20 iterations;
SM120 allows only 99 KiB per block. On locally validated SM120 and other explicit portability-test architectures, a thin
autograd adapter calls the **same unmodified official forward/backward kernels**
with the backward `token_block_size=16` argument (50 KiB). SM90/SM100 retain the
official wrapper's 32-token tile. This changes scheduling only: stream count,
20 iterations, equations and differentiation are unchanged. It is an explicit
local launch adaptation, not a claim that every official wrapper works unmodified.

The official library supplies operations, not a complete nanochat training
recipe. Our existing parameter initialization is still a local choice:
Normal(0, 1/sqrt(nD)) projections, 0.01 dynamic scales, logit(1/n) pre bias,
zero post bias and identity residual-logit bias. Connection parameters retain
their separate AdamW group at native unembedding LR. Backbone MuonAdamW and
all data/token/LR settings are unchanged. There is no pipeline communication on
our single-GPU setup. We do not claim to reproduce the paper's 6.7% overhead.

## Environment and hardware

The official kernels target **SM90/SM100** and require the CUDA 13.1 toolchain.
Normal training also permits **locally validated SM120** with the tile-size
adaptation above. Other architectures require validation. The bounded gate
sets an explicit portability flag and may test SM80+ using the same unmodified
kernel code, because this version disables WGMMA and TMA in its mHC paths.
Passing such a gate establishes our tested configuration only, not upstream
support for all shapes or hardware. The existing baseline environment remains available; the official
mHC environment is independently locked under `runtime/` (Python 3.11,
PyTorch 2.10.0, TileLang 0.1.9, TVM-FFI 0.1.9, pinned TileKernels and the same
CCE source revision). The CUDA 13.1.115 compiler and matching 13.1 headers are
locked separately under `toolchain/`, avoiding a header collision with
PyTorch's pinned CUDA 13.0 runtime. Native CUDA 13.1+ drivers are used directly. On NVIDIA's supported 570/580
datacenter driver branches, the launcher extracts the official
[cuda-compat-13-1 package](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
into job scratch and selects its userspace libraries. It does not modify the
host driver or require root. The package version, SHA256 and actual selected
driver path are recorded. Architecture validation is separate from driver
compatibility.

On a compute node, run `uv sync --frozen --project research/connections/runtime`
and `uv sync --frozen --project research/connections/toolchain`.
The cached base environment is synchronized with this lock before every gate;
the loose upstream TVM-FFI bound is pinned to 0.1.9 after an import failure with
0.1.15.post2. The lock also includes nanochat's tokenizer/data dependencies.
From the repository root on a compute node with a compatible driver:

```bash
export PATH="$PWD/research/connections/runtime/.venv/bin:$PATH"
export CUDA_HOME="$PWD/research/connections/toolchain/.venv/lib/python3.11/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/research/connections/runtime/.venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
python -m scripts.connection_train --variant baseline --steps 50 --data-dir data --output-dir outputs/baseline
python -m scripts.connection_train --variant mhc --steps 50 --data-dir data --output-dir outputs/mhc
```

For older supported drivers, use the pinned compatibility-package setup in
`validate.sh` before importing CUDA libraries.

The data directory is the existing frozen FineWeb-Edu cache from job 7444406.
W&B is always `yibozhong657-none/nanochat`, group
`mhc-official-kernels-20261030`. This change authorizes implementation and bounded
validation only; no full 1B training is automatically submitted.

## Validation results — 2026-10-30

Job `7546888.0` exited successfully, one execution, source `6d4eae6`.
The official source hashes matched. **14 tests passed**, zero failures/errors;
10 native FA3 comparison tests were skipped because SM120 uses the SDPA path.
Checks cover D128/D512 official-reference output and all input/parameter gradients,
head gradients, Sinkhorn constraints/gradients, real 2K causality, KV cache,
CCE/dense CE, model state and exact legacy baseline equivalence. Maximum pipeline
relative-L2 output error was 0.000546, activation-gradient error 0.002667 and
parameter-gradient error 0.000022. See [kernel-errors.json](kernel-errors.json).
These checks do not claim full optimizer/RNG interrupted-training equivalence.

Both smokes finished on the same RTX PRO 6000 Blackwell Server Edition (SM120),
PyTorch 2.10.0, microbatch 16, accumulation 1, no activation checkpointing,
50 steps / 1,638,400 tokens per variant. Native driver 13.1 was used; the NVIDIA
forward-compatibility path was not exercised. All three W&B runs were verified
through the API. No 1B pretraining or independent test-set evaluation was launched.

| Variant | Initial dev CE | Step-50 dev CE | Steady seconds/step | Peak PyTorch reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| baseline | 10.398013 | 6.496188 | 0.080839 | 5.389 |
| mHC | 10.398013 | 6.498858 | 0.152461 | 8.475 |

Steady timings average steps 11–50, excluding initial compilation. mHC is
**88.6% slower** in this small-model setup. This is not the paper's large-model
optimized execution setup, and does not reproduce its 6.7% overhead. Reserved
memory excludes some non-PyTorch/device allocations and is not an actual 24GB-GPU
validation. Short-run loss reduction establishes trainability, not a quality win.

- [Correctness gate](https://wandb.ai/yibozhong657-none/nanochat/runs/19d45f79)
- [Baseline smoke](https://wandb.ai/yibozhong657-none/nanochat/runs/12c1457a)
- [mHC smoke](https://wandb.ai/yibozhong657-none/nanochat/runs/2a00a93e)
- [Audited validation](validation.json), [runtime packages](environment.txt),
  [toolchain packages](toolchain-environment.txt)

Earlier gates were retained: 7546808 failed during test collection due to missing
RustBPE; 7546843 exposed mixed CUDA compiler/runtime headers; 7546850 compiled
successfully but hit the SM120 shared-memory limit in the official Sinkhorn
backward wrapper. None of those runs started training. The final successful
kernel files are still byte-for-byte official; the launch adapter is local.

## Complete NVIDIA mHC backend

Full-operator replacement is a separate, explicit `--connection-backend nvidia`
option; `--projection-backend nvidia` is the older projection-only experiment.
The complete path calls the released TE 2.20.2 projection, fused scale, log-space
Sinkhorn, aggregate and expand/combine forward/backward APIs, including the
final head via projection/scale/aggregate. Input replication uses ordinary
PyTorch because TE has no standalone embedding expansion operation. The model
keeps `[B,T,D,4]` throughout, avoiding per-layer stream-layout copies. Its phi
columns are interleaved at initialization, preserving the effective seeded
weights; DeepSeek checkpoints require the same column permutation on import.
The full NVIDIA path retains upstream numerical conventions (FP32 machine
epsilon, sigmoid without added epsilon, log-space Sinkhorn), not exact old
DeepSeek arithmetic. The optional official FP32 accumulation buffer is enabled
with `--fused-connection-grad`; it now defaults off based on the completed
comparison below. Both settings were tested. [Full comparison protocol](nvidia-full/protocol.json).

Full replacement completed in job **7547667.0**, source `210b79b`, on one **H200**
(UUID `5cdb85f0-5737-043f-3398-ed859ad2fafe`). All **22 checks passed**, with no
failures or skips: six projection regressions, ten complete-pipeline tests and
six full-model/native-precision checks. They include independent FP64 math,
true 2K causality, KV cache, CCE, every model gradient, fused/unfused backward,
repeated backward, activation checkpointing and a fresh TMA allocation context.
The unchanged 26 legacy regressions also passed in the earlier 7547609 gate.
The six successful W&B runs are finished and API-verified.

Each configuration used the same 8-layer D512/4-head/2K/32K model, FineWeb data,
seed42, native MuonAdamW/LR, microbatch16/accum1 and no activation checkpointing
for timing. Each fresh model process ran 20 warmup +60 ordinary timed +26
diagnostic updates: **106 updates /3,473,408 tokens per variant**. Native NVIDIA
trials leave `NVIDIA_TF32_OVERRIDE` unset; baseline/DeepSeek use `0`. No 1B run or
checkpoint was produced. Raw samples are retained, including longer steps.

| H200 configuration | Mean ms/step | Median ms/step | Overhead vs vanilla | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 60.229 | 60.166 | — | 5.510 |
| DeepSeek mHC | 129.808 | 128.943 | 115.5% | 8.502 |
| NVIDIA projection only | 85.614 | 84.198 | 42.1% | 8.795 |
| Full NVIDIA, ordinary gradient accumulation | **83.348** | **82.806** | **38.4%** | **8.803** |
| Full NVIDIA, fused FP32 gradient accumulation | 84.834 | 84.527 | 40.9% | 12.791 |

Full NVIDIA with ordinary accumulation reduces mean step time **35.8%** versus
DeepSeek mHC and **2.65%** versus projection-only. The latter difference is small:
the corresponding median reduction is 1.65%, and SDs are 1.275 and 3.010 ms.
These are one short sequential trial per configuration, not a statistical claim
of a universal 2.65% improvement. The remaining overhead against vanilla is
substantial. This is a throughput result, not evidence of improved NTP quality.

The fused path takes 1.49 ms longer in this trial and allocates exactly **4 GiB**
more at peak, consistent with sixteen per-branch FP32 buffers. Consequently the
full NVIDIA option defaults to the measured ordinary-accumulation setting;
`--fused-connection-grad` remains available. This changes only the selection of
an already tested path. The explicit legacy backend remains available.

The CUDA traces contain **zero DeepSeek mHC kernels** in both full NVIDIA trials.
Across two instrumented steps, NVIDIA projection/scale/aggregate forward and
backward each appear 34 times, and Sinkhorn/expand-combine forward and backward
each appear 32 times, accounting for all sixteen branches plus the final head.
NVIDIA kernel files remain byte-for-byte SHA-verified against TE 2.20.2.

Two failed gates are retained in [failed attempts](nvidia-full/failed-attempts.json).
The first exposed an incorrect test assumption that 20 Sinkhorn iterations
must converge arbitrary inputs to a preset row residual. Actual/FP64 20-step
residuals are 0.000598311/0.000598355; same-iteration output/gradient checks and
the separate 200-iteration convergence check pass. The second exposed a Hopper
checkpoint integration issue: TE's global initialization guard outlived the
allocator's [Triton 3.6 ContextVar](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/_allocation.py).
The local adapter now registers the standard CUDA workspace allocator only in
an empty context and preserves valid allocators. No NVIDIA kernel was edited.

- [Full comparison results](nvidia-full/results.json), [trace/math audit](nvidia-full/audit.json).
- [W&B audit](https://wandb.ai/yibozhong657-none/nanochat/runs/3cd1824a).

## Earlier NVIDIA projection-only comparisons

The NVIDIA author's [full-model comparison in TE PR 2790](https://github.com/NVIDIA/TransformerEngine/pull/2790#issuecomment-4208434892)
reports a 2B Nemotron model on 16 nodes of 8 B200 GPUs, pure data parallelism,
global batch 768, microbatch 3 and accumulation 2. Approximate iteration times
were 960 ms without mHC, 1100 ms with the new Triton mHC implementation, and
1160 ms with the then-current Megatron cuTile implementation. Thus Triton mHC
added about 14.6% step time versus vanilla and reduced time about 5.2% versus
cuTile in that configuration. These are author-reported log estimates from
April 8, 2026, not measurements of our nanochat model or of TE 2.20.2. The historical
experiments in this section replace only projection; the other mHC operators
remain DeepSeek's. The completed full replacement is reported above.

Explicit `--projection-backend nvidia` and `nvidia_precise` use unmodified
NVIDIA Transformer Engine 2.20.2 projection forward/backward. The adapter only
flattens/reshapes tensors, slices the padded output and applies our RMS epsilon.
The final stream head retains its existing zero padding to 24 projection rows.
All other connections remain DeepSeek operators. Default selection is unchanged.
`nvidia_precise` requires `NVIDIA_TF32_OVERRIDE=0` at process startup: TE's
`use_tf32=False` argument controls Triton but does not disable TF32 in its
separate cuBLAS weight-gradient call. Two upstream source files are SHA-checked.

Controlled gate `7547306.0`, source `f531f54`, passed all 26 checks with no skips:
FP64 outputs/input/weight gradients, official comparisons, unchanged initial
weights, true 2K causality, cache, CCE, model gradients and baseline regression.
The six W&B runs were verified. It used one RTX PRO 6000 (SM120), the same
FineWeb data/seed/model/batch/optimizer as earlier profiling, and **disabled
cuBLAS TF32 for every trial**. It therefore does not measure fully default TE
arithmetic. The isolated newer environment requires TVM-FFI 0.1.11; the original
runtime remains pinned separately. No upstream kernel was patched.

| Controlled configuration | Mean ms/step | Median ms/step | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Vanilla | 80.569 | 80.498 | 5.389 |
| DeepSeek projection | 152.210 | 152.073 | 8.475 |
| Compiled PyTorch projection | 129.707 | 129.620 | 12.836 |
| NVIDIA Triton TF32, FP32 weight gradient | 133.004 | 120.104 | 8.795 |
| NVIDIA Triton TF32x3, FP32 weight gradient | 123.168 | 123.149 | 8.795 |

The NVIDIA TF32 trial includes one **376.977 ms** pause among twenty measured
steps (SD 57.426 ms). All samples are retained; its mean must not be presented as
stable throughput or silently replaced by a filtered mean. The same-process
TF32x3 microbenchmark also disagrees with the separately tuned full-model trace;
upstream's forward autotune cache key omits precision. A bounded follow-up uses
fresh processes per microbenchmark, 60 ordinary timing samples after warmup and
one GC collection (automatic GC remains enabled), and adds native TE/cuBLAS
precision. [Follow-up protocol](nvidia-native/protocol.json).

The follow-up **completed successfully** as `7547373.0`, source `558dbbc`, on one
H100 80GB (UUID `fff6db8e-df5b-6a7f-c506-eb78b2f0c043`). All **29 tests passed**,
none failed or skipped, and all seven W&B runs were verified. Each model used
106 temporary optimizer updates, including 20 warmup and 60 ordinary timed
steps, with the same D512/8-layer/2K, micro16/accum1, FineWeb data, seed and
optimizer. These H100 timings form their own comparison; do not mix them with
the earlier RTX PRO 6000 numbers.

| H100 configuration | Mean ms/step | Median ms/step | Overhead vs vanilla | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 64.933 | 64.677 | — | 5.471 |
| DeepSeek projection | 137.967 | 137.721 | 112.5% | 8.480 |
| Compiled PyTorch projection | 97.021 | 97.013 | 49.4% | 12.789 |
| NVIDIA native precision | 95.306 | 95.356 | 46.8% | 8.795 |
| NVIDIA TF32, FP32 weight gradient | 95.599 | 95.355 | 47.2% | 8.795 |
| NVIDIA TF32x3, FP32 weight gradient | 98.545 | 98.350 | 51.8% | 8.795 |

NVIDIA native precision reduces full-step time **30.9%** versus the original
DeepSeek projection, and **1.8%** versus compiled PyTorch in this short trial.
It still takes **46.8% longer than vanilla**. Its 60-step mean/median agree and
SD is 0.714 ms; there is no pause comparable to the first trial's 377 ms outlier.
Native versus FP32 weight-gradient timing differs by only 0.293 ms, so this
single short trial does not establish a meaningful advantage from relaxing
that GEMM's precision. NVIDIA's allocator footprint is lower than compiled
PyTorch's, but higher than vanilla's. All mHC variants still use DeepSeek's
remaining operators, so this is not a full NVIDIA mHC implementation benchmark.

Independent-process projection microbenchmarks (forward/backward ms) were:
DeepSeek 0.179/3.132; compiled PyTorch 0.387/0.594; NVIDIA native 0.272/0.597;
NVIDIA FP32 weight-gradient 0.227/0.629; NVIDIA precise 0.489/0.525. These include
launch gaps and a scalar probe; they are not isolated device-kernel times. The
earlier 5.206 ms precise-forward result must not be interpreted as intrinsic
TF32x3 cost. The H100 run uses independent tuning and different hardware, so it
does not alone prove the cause of the earlier anomaly.

Native NVIDIA projection's maximum relative FP64 errors over D128/D512 were
4.172e-4 (output), 1.997e-3 (input gradient), and 2.040e-4 (weight gradient).
The precise setting gave 5.607e-7, 2.235e-5 and 1.633e-7 respectively. Both passed
their declared precision-specific thresholds. Kernel traces verify the NVIDIA
forward and backward paths replaced all old projection calls. This is a
correctness/throughput result, with **no long-run quality comparison or 1B run**.

- [Completed follow-up results](nvidia-native/results.json),
  [numerical and trace audit](nvidia-native/audit.json),
  [W&B audit](https://wandb.ai/yibozhong657-none/nanochat/runs/5fb502fa).
- The initial H200 attempt was preempted during setup. The cached extension
  source was then fixed by declaring it as a direct dependency; the successful
  H100 attempt installed the SHA-verified wheel without rebuilding it. Package
  versions and numerical/measurement protocols were unchanged.

- [Controlled results](nvidia-projection/results.json),
  [numeric/trace audit](nvidia-projection/audit.json),
  [retained failed attempts](nvidia-projection/failed-attempts.json).
- [Controlled W&B audit](https://wandb.ai/yibozhong657-none/nanochat/runs/2a39d0e3).
- SM89 failed the unchanged DeepSeek residual kernel's PDL requirement; do not
  treat standalone NVIDIA projection success there as full-model support.
- The first dependency build lacked NCCL/cuSPARSE headers; explicit compiler
  include paths fixed it. The compiled official extension is retained in the
  successful archive for retries, with its Torch/CUDA/ABI and SHA-256 metadata.
