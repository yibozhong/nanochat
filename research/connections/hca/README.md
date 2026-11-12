# Fixed / Dynamic HCA reference

Status: implementation and validation harness; GPU results are recorded separately
in `validation.json` when available. No formal pretraining or quality claim.

This adds `attention_mode = full | fixed_hca | dynamic_hca` to the existing vanilla
`ConnectionModel`. `full` is the unchanged default. The first port requires
`variant=baseline, architecture=vanilla, streams=1, attention_kernel=original`.
Native GPT, existing residual variants and their checkpoints retain their paths.

## Research sources and deliberate adaptations

- [DeepSeek-V4 §2.3.2–3](https://arxiv.org/html/2606.19348v1#S2.SS3.SSS2),
  [official Compressor/Attention implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py):
  per-channel softmax pooling, recent uncompressed SWA, completed compressed KV.
- [DMC, ICML 2024 §3 / Appendix G](https://arxiv.org/html/2403.09636v2):
  learned discrete merge decisions, Gumbel-Sigmoid relaxation, unrolled partial
  accumulations and differentiable attention masking. No official standalone
  DMC implementation was identified during this review; its equations are the
  reference, not an unattributed third-party implementation.
- [H-Net §2.2–2.4](https://arxiv.org/html/2507.07955v1),
  [official routing implementation](https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py):
  content-dependent routing with separate forward/step state, ratio control.
  H-Net's hierarchical byte model and cosine router are not ported wholesale.

These are independently written nanochat adaptations. In particular we keep
nanochat's separate K/V and head count, full RoPE, QK RMSNorm and 1.2 scaling,
MLP, residuals, vocabulary and data. We do not adopt DeepSeek's shared K=V MQA,
partial RoPE, inverse output RoPE, grouped output projection or learned sink.
Both variants use identical pooling, local context and attention normalization;
only Dynamic adds one scalar content router per layer, shared across its heads.
Backbone weights and RNG match the same-seed baseline exactly at construction.

## Equal-budget causal chunking

Defaults: B=16, K=16 summaries per budget frame, frame=B*K=256 tokens,
local SWA W=256 tokens **including the current token**. Fixed closes every B
tokens. Dynamic decides to close **after** observing the current token:

    logit_t = Linear(normalized_hidden_t) + (current_chunk_length - B)/(B/4)
    soft_t = sigmoid((logit_t + logistic_noise_t)/temperature)
    end_t = hard(soft_t >= 0.5) + soft_t - stop_gradient(soft_t)

The age denominator is at least 1. Router weights/bias start at zero; without
noise this starts at fixed boundaries. Temperature defaults to 1. Training
uses logistic noise (difference of two Gumbels); evaluation/cache uses no noise.
`hca_noise=False` makes the hard training/evaluation computation identical.
The estimator is biased: hard decisions are not mathematically differentiable.
It supplies task gradients through both accumulator reset and summary visibility,
not just an auxiliary budget objective.

A causal quota controller reserves the final frame position for the last chunk,
allows at most K-1 earlier closures, and forces a closure when the remaining
token positions equal the remaining quota. It reads no future hidden states,
does no sequence-wide top-k, and never flushes at a forward-call boundary.
Every complete 256-token frame has exactly 16 nonempty chunks, regardless of
router parameters/noise. Thus both models have exactly 128 completed summaries
per layer at 2K. Short prefixes / partial frames can have different counts;
reports must use actual counts there, not label all prefixes as exactly 16x.

This budget-frame restriction is an intentional controlled comparison, **not**
original DMC or unconstrained variable-rate HCA. A ratio penalty alone would not
guarantee the user's equal-average-budget requirement. The tradeoff is forced
frame boundaries and zero router gradient at quota-forced decisions. Report
chunk-length distributions as well as quality in any later trained comparison.

## Pooling and unfinished chunks

A shared K/V weight is learned independently for each channel:

    score_t = 10 * tanh(Linear(hidden_t)/10)
    a_t = exp(score_t)
    mass_t = (1-end_(t-1))*mass_(t-1) + a_t
    sumK_t = (1-end_(t-1))*sumK_(t-1) + a_t*K_t
    sumV_t = (1-end_(t-1))*sumV_(t-1) + a_t*V_t
    pooledK_t = sumK_t/mass_t; pooledV_t = sumV_t/mass_t

This is exactly per-channel softmax pooling over the hard current chunk.
The smooth bounded scores prevent overflow in the STE reset derivative; there
is no block-position bias table. Accumulators and router/pooling projections
are FP32 (FP64 in the numerical oracle). Completed summaries are stored at the
activation dtype. Pool unrotated K/V, then RoPE the summary K at its **closing
token position** and apply nanochat's QK normalization/scaling. This position
choice differs from DeepSeek's fixed-block start position; both variants here
use the closing position. Values are not rotated.

A query at t attends to raw KV from max(0,t-W+1)..t and every summary closed
strictly before t, with **one joint softmax**. Local tokens may overlap summaries,
as in the HCA two-pool design. The new summary becomes visible at t+1.
The unclosed chunk remains only in the accumulation state plus the local raw
window; it is not a visible summary, and no future tokens enter a past query.
Requiring W >= frame ensures even the longest possible unclosed chunk remains
fully covered by local attention. A call ending mid-chunk preserves its state
for the next call; it does not publish a synthetic EOS summary.

## Training and cache implementation

Training unrolls one candidate prefix summary per token and applies hard gates
to its softmax numerator. Zero gates remove candidates exactly, while allowing
finite surrogate gradients at zero (no `log(0)` autograd). Future candidates
are also excluded from numerical softmax stabilization. This reference uses
dense attention and a Python recurrence: it is **not** a training memory/speed
optimization. Fullgraph block compilation is explicitly rejected for now.
Activation checkpointing preserves noise RNG and is tested.

Inference uses a distinct `HCAKVCache` storing only W raw KV, actual completed
summaries, three FP32 accumulator tensors, and small per-layer/per-example
counters. It supports arbitrary prefill splits, single-token decode,
state serialization, reset and batch=1 prefix broadcast. Input rows have equal
sequence lengths; padding masks, document-reset masks and speculative rollback
are not part of this initial API. Existing packed sequence semantics are kept.
Cache use requires `eval()` plus `no_grad()`; the mutable cache is not a training
TBPTT API. Baseline's `make_kv_cache()` returns the existing ordinary KVCache.

At T=2048, B=16, W=256, there are 384 stored KV pairs per head (256 raw + 128
summary) plus three FP32 accumulators. At BF16 this is about **5.29x actual tensor
cache reduction**, even though the history compression ratio is 16x. Dynamic
and Fixed have the same tensor cache budget at complete frame endpoints; Dynamic
is meant to improve *quality at that budget*, not to save additional bytes there.
Accounting includes live backing-storage sizes, not only logical tensor shapes.
Python object overhead, allocator fragmentation and model parameters are separate.

## Usage

```python
from nanochat.connection_model import ConnectionConfig, build_connection_model

cfg = ConnectionConfig(variant='baseline', streams=1,
                       attention_mode='dynamic_hca', hca_block_size=16)
model = build_connection_model(cfg).prepare(compile_blocks=False)
loss = model(tokens, targets)   # existing CCE; router/pool get task gradients
loss.backward()
model.eval()
cache = model.make_kv_cache(batch_size=1)
with torch.no_grad():
    model(prompt[:, :101], kv_cache=cache)
    logits = model(prompt[:, 101:], kv_cache=cache)
print(cache.statistics())
```

The existing `scripts.connection_train` accepts `--attention-mode`,
`--hca-block-size`, `--hca-budget-chunks`, `--hca-window`, `--hca-temperature`,
`--hca-noise` / `--no-hca-noise`. It chooses eager blocks for HCA. Pool/router
parameters use the existing connector AdamW group; backbone MuonAdamW remains
unchanged. Configs/checkpoints carry all HCA fields; legacy configs default to
full attention. This interface does not authorize a new training run.

## Validation scope

Run only on HTCondor workers:

```sh
python -m pytest -v tests/test_hca.py
python -m scripts.connection_hca_gate
```

Checks cover independent FP64 forward/all continuous-input VJPs, boundary quota
under extreme logits, finite pool gradients, nonzero LM gradients to routing,
future perturbation and exact zero future gradients, BF16/FP32 dense vs split
prefill/single decode, unfinished tails, cache serialization/broadcast/reset,
baseline weights/RNG, activation recomputation/Gumbel RNG, real MuonAdamW
next-step restoration and CCE gradients. Discrete STE derivatives are not
misrepresented as finite-difference derivatives of a hard threshold.

The bounded gate also compares full/fixed/dynamic on the existing FineWeb 2K
cache at D512/H4/two layers, measuring actual cache bytes and recording same-GPU
reference runtime. These timings describe the unoptimized reference only.
Diagnostic CE from untrained weights is not a trained quality comparison.
