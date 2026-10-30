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
Normal training rejects other architectures until validated. The bounded gate
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

## Validation status

GPU validation is pending. The new gate compares official kernel outputs and
all input/parameter gradients to the official PyTorch reference, checks both
D128 and D512, tests the final stream head and Sinkhorn, validates real 2K
causality/cache/CCE/model state, and verifies old baseline equivalence. It then
runs baseline and mHC for 50 updates each on the same GPU and environment.
The old reference implementation's passed tests do not certify this new path.
