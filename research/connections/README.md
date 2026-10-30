# Official mHC kernels in nanochat

HC was removed at the user's request on 2026-10-30. Its code remains in Git tag
`hc-archive-2026-10-30`; the old measurements under `archive/` describe the
retired PyTorch reference implementation, not this kernel implementation.

The active variants are `baseline`, `mhc` and `mhc_no_manifold`. They retain the existing full
attention backbone: eight layers, D512, four heads, 2K context and 32K vocabulary.
The baseline has 58,720,256 parameters; mHC has 59,515,317. The default is
microbatch 16, accumulation 1, no whole-block activation checkpointing and
32,768 tokens/update. Native nanochat GPT/SSSL remains unchanged.

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
Results for this new ablation are pending; the completed mHC results below are
the earlier two-variant validation.

## Official code used directly

All mHC computation calls the unmodified functions in
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
