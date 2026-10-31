# Native nanochat with NVIDIA connections

`GPTConfig.connection_variant` selects `baseline` (the original default), `mhc`,
or `mhc_no_sinkhorn`. Both added variants use the existing NVIDIA TE kernels.
The old full-attention vanilla experiment remains `architecture=vanilla`.

The native path retains SSSL windows (512/2048 at context 2048), alternating
value embeddings and their gates, smear, residual scaling, x0 injection,
mid-layer backout, QK norm, RoPE, ReLU², and the original output softcap.
Attention and MLP each get their own four-stream connection. Embeddings are
expanded after smear; x0, per-layer scaling and backout act streamwise. The
existing NVIDIA head collapses streams after backout, before final RMSNorm.
Consequently Sinkhorn constrains the connection mixing matrices, not the entire
residual graph including native scaling/x0/backout. This is a composition with
native nanochat, not a claim of reproducing the paper's whole architecture.

Native backbone initialization completes before connector initialization.
Equal seeds therefore match every shared native parameter across the three
variants, and match every parameter between mHC and no-Sinkhorn. Native optimizer
groups are retained, including value embeddings and scalar/gate treatment.
Connections receive a separate AdamW group with unembedding LR and zero decay.
They are counted separately from backbone scaling-law parameters. FLOP estimates
include their projection matrices but omit mixing/normalization overhead; use
measured time for efficiency claims.

The no-Sinkhorn variant skips the existing entire log-space Sinkhorn operation;
its residual matrix is raw, signed scale/projection output. Other NVIDIA kernels,
pre/post gates, head and initialization are unchanged. It is an ablation, not an
exact reproduction of original HC.

## Entrypoints

Native upstream loader and loss (example only; no run is launched by this file):

```bash
python -m scripts.base_train --connection-variant mhc --depth 8 --run dummy
```

The fixed FineWeb packed-data harness, full-vocabulary CCE, rolling latest,
separate best and exact resume support all three native variants:

```bash
python -m scripts.connection_train --architecture native --variant mhc \
  --steps 10 --data-dir data --output-dir results/native-mhc \
  --checkpoint-dir checkpoints --resume auto
```

Use `--variant baseline` or `--variant mhc_no_sinkhorn` for the other variants.
Native defaults to SSSL; `--window-pattern` changes it explicitly. No variant
silently replaces sliding windows with full attention. The packed harness follows
the native MuonAdamW batch/width/depth LR and decay rules. Only `--formal-training`
enables W&B. Do not apply the full-attention vanilla memory/throughput numbers to
this larger native model (value embeddings add parameters).

NVIDIA paths require BF16 and SM90+ with the existing TE runtime; no eager,
DeepSeek, or alternative connection kernel fallback is added. FP8 integration is
not supported. Only native attention/MLP branches are compiled, leaving TE custom
autograd outside whole-model compilation. Native default GPT still follows the
original upstream compilation path. Existing vanilla checkpoints default to
`architecture=vanilla`; native/vanilla and mHC/no-Sinkhorn identities are checked
before resuming in the packed harness.

## Validation

The gate compares the default native model against immutable pre-change `gpt.py`;
checks full-size matched initialization and optimizer grouping; reduces stream
connections to ordinary addition to verify native features and gradients;
exercises real NVIDIA kernels with 2K causality, SWA, GQA, smear-aware KV caching,
and all native feature gradients; tests CCE, compiled branches, activation
checkpointing and exact fixed-gradient optimizer/RNG restore. Three original
vanilla/mHC resume regressions also run.

Each full D512/8-layer native variant then runs a bounded 5→exit85→fresh-process10
step check on the existing FineWeb cache. This is a diagnostic, not a quality or
speed comparison, and does not create W&B runs or return diagnostic weights.
See `validation.json` once the GPU gate has completed; submission is not a pass.
