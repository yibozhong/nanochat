# Official mHC kernels in nanochat

HC was removed at the user's request on 2026-10-30. Its code remains in Git tag
`hc-archive-2026-10-30`; the old measurements under `archive/` describe the
retired PyTorch reference implementation, not this kernel implementation.

The active variants are `baseline` and `mhc`. Both retain the existing full
attention backbone: eight layers, D512, four heads, 2K context and 32K vocabulary.
The baseline has 58,720,256 parameters; mHC has 59,515,317. The default is
microbatch 16, accumulation 1, no whole-block activation checkpointing and
32,768 tokens/update. Native nanochat GPT/SSSL remains unchanged.

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

The official library supplies operations, not a complete nanochat training
recipe. Our existing parameter initialization is still a local choice:
Normal(0, 1/sqrt(nD)) projections, 0.01 dynamic scales, logit(1/n) pre bias,
zero post bias and identity residual-logit bias. Connection parameters retain
their separate AdamW group at native unembedding LR. Backbone MuonAdamW and
all data/token/LR settings are unchanged. There is no pipeline communication on
our single-GPU setup. We do not claim to reproduce the paper's 6.7% overhead.

## Environment and hardware

The official kernels target **SM90/SM100** and require the CUDA 13.1 toolchain.
A30 (SM80), L40 (SM89), and RTX PRO 6000 Blackwell (SM120) are not accepted by
this entry. The existing baseline environment remains available; the official
mHC environment is independently locked under `runtime/` (Python 3.11,
PyTorch 2.10.0, TileLang 0.1.9, pinned TileKernels, CUDA nvcc 13.1.115 and the same
CCE source revision). GPU drivers must support the requested toolchain.

On a compute node, install with `uv sync --frozen --project research/connections/runtime`.
Use that environment's Python to run:

```bash
python -m scripts.connection_train --variant baseline --steps 50 --data-dir data --output-dir outputs/baseline
python -m scripts.connection_train --variant mhc --steps 50 --data-dir data --output-dir outputs/mhc
```

The data directory is the existing frozen FineWeb-Edu cache from job 7444406.
W&B is always `yibozhong657-none/nanochat`, group
`mhc-official-kernels-20261030`. This change authorizes implementation and bounded
validation only; no full 1B training is automatically submitted.

## Validation status

GPU validation is pending. The new gate compares official kernel outputs and
all input/parameter gradients to the official PyTorch reference, checks both
D128 and D512, tests the final stream head and Sinkhorn, validates real 2K
causality/cache/CCE/model state, and verifies old baseline equivalence. It then
runs baseline and mHC for 50 updates each on the same GPU and environment.
The old reference implementation's passed tests do not certify this new path.
