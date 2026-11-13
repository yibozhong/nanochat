# Fixed HCA / free Dynamic HCA

Status: reference implementation; GPU validation results are recorded separately.
No formal pretraining or matched-budget quality result. The initial hard-quota
version was stopped; its checkpoints are explicitly rejected for HCA loading.

### Official-source audit, 2026-11-13

The current reference must not be treated as a faithful or efficient DeepSeek
HCA execution path. Direct review of DeepSeek's official `inference/model.py`
(`Compressor.forward`, `get_compress_topk_idxs`, `Attention.forward`) and
`inference/kernel.py` (`sparse_attn_kernel`) establishes these differences:

- Official prefill reshapes complete chunks and applies channel-wise softmax
  and weighted sum in parallel, producing `floor(T/R)` summaries. Our training
  code constructs T prefix snapshots in a Python token loop, even for Fixed.
- Official attention gathers only the causal local window and completed
  summaries and uses online softmax. Our training code materializes FP32 scores
  over T raw candidates plus T prefix candidates, including masked entries.
- Official incomplete chunks remain in compressor state; the completed summary
  is visible at its last included token. Our current prefix summary is visible
  before a chunk is complete. This is a declared local adaptation, not official
  HCA semantics, and is unnecessary for Fixed B16 with SWA256.
- Official compression includes learned within-chunk position bias and places
  compressed RoPE at chunk start. Our port omits that bias and uses prefix end.
  Separate K/V and nanochat QK normalization are additional declared adaptations.

For the requested T2048/R16/W256, the official selection pattern needs at most
384 selected entries per query; our dense training scores have 4096 columns.
At microbatch8/four heads one such FP32 score tensor alone is 1 GiB per layer,
before softmax intermediates and backward. These are implementation differences,
not evidence that HCA intrinsically needs a 54-day 10B run. The actual official
release reviewed is inference code; its custom kernel is not an available
training/backward implementation. Training remains blocked pending diagnosis.

`ConnectionConfig.attention_mode = full | fixed_hca | dynamic_hca` selects the
attention. `full` retains the original default. HCA currently requires vanilla,
baseline residuals, one stream and the original Q/K/V projection layout.

## Sources and scope

- [DeepSeek-V4 §2.3.2–3](https://arxiv.org/html/2606.19348v1#S2.SS3.SSS2),
  [official Compressor/Attention implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py):
  per-channel softmax pooling, local raw KV plus compressed history.
- [DMC, ICML 2024 §3.2, equations 9–12 and Appendix G](https://arxiv.org/html/2403.09636v2#S3.SS2):
  Gumbel-Sigmoid, partial accumulations, masking intermediate states, global
  one-sided compression loss. No official standalone DMC implementation was
  identified in this review; its equations are the reference.
- [H-Net §2.2–2.4](https://arxiv.org/html/2507.07955v1),
  [official routing implementation](https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py):
  content-adaptive routing and separate prefill/step state handling.

This is an independent nanochat adaptation. We retain separate K/V, the original
head count, full RoPE/QK RMSNorm/1.2 scaling, MLP, residuals and data. We do not
port DeepSeek's shared K=V MQA, partial/inverse RoPE, grouped output projection
or learned sink, or H-Net's hierarchical byte model. Same-seed backbone weights
and RNG are preserved when adding pooling/router modules.

## Free boundaries and the DMC-style objective

Fixed uses B=16 by default, starting a new chunk at t % B == 0. Dynamic predicts
one scalar per token per layer from the current causal normalized hidden state;
the layer shares its boundaries across heads:

    logit_t = Linear(hidden_t)
    soft_start_t = sigmoid((logit_t + logistic_noise_t)/temperature)
    hard_start_t = (soft_start_t >= 0.5)
    start_t = hard_start_t + soft_start_t - stop_gradient(soft_start_t)

Only the first token is forced to start a chunk. There is no age prior, maximum
chunk length, per-frame quota, fixed number of summaries, or end-of-call flush.
Positive or negative router logits can generate one entry per token or a single
chunk spanning the whole sequence. Changing target R does not directly change
inference boundaries. Routing weights start at zero and bias at +2, starting
mostly uncompressed. No initial or untrained 16x compression is claimed.

During training logistic noise is the difference of two Gumbels. Evaluation and
cached decoding use noiseless decisions. Temperature defaults to 1 and is
configurable; `hca_noise=False` gives identical hard forward semantics in train
and eval. Straight-through gradients are a biased estimator, not exact
derivatives of the hard threshold. They reach the LM objective through pooling
reset and summary visibility. The compression objective uses the **continuous**
relaxation, separately from the hard-forward estimator:

    global_soft_keep_rate = sum(soft_start) / number_of_token_layer_decisions
    L_CR = max(0, global_soft_keep_rate - 1/R)
    L_total = L_CE + hca_ratio_loss_weight * L_CR

Here soft_start = 1-alpha in DMC notation, R=`hca_block_size` (default 16), and
the loss coefficient defaults to 1. All samples, tokens and layers in a forward
microbatch are aggregated **before** the ReLU. Because every layer has the same
number of heads and shares its routing across those heads, head weighting
cancels. Layers can compensate each other. With gradient accumulation this is
a per-forward-microbatch global loss, not one ReLU over all accumulated batches.
The compulsory first KV entry is counted, including in short sequences.

Too many retained KV entries incur a penalty; greater compression incurs no
penalty from L_CR, with the LM objective providing the information-preservation
tradeoff. This is not an equality constraint and does not guarantee actual R.
Log continuous keep rate, hard keep rate, and actual inference cache entries
separately. A later quality comparison must match measured average cache budgets
on held-out data, freeze its settings, and report any mismatch. A target R=16
label alone is not evidence of a fair comparison. No long training is authorized
or launched by this implementation task.

## Pooling, causality and arbitrarily long unfinished chunks

The following is per-channel softmax pooling with common weights for separate
K and V, accumulating in FP32 (FP64 for the oracle):

    score_t = 10 * tanh(Linear(hidden_t)/10)
    a_t = exp(score_t)
    mass_t = (1-start_t)*mass_(t-1) + a_t
    sumK_t = (1-start_t)*sumK_(t-1) + a_t*K_t
    sumV_t = (1-start_t)*sumV_(t-1) + a_t*V_t
    prefixK_t = sumK_t/mass_t; prefixV_t = sumV_t/mass_t

Smoothly bounded pooling logits prevent overflow in the STE reset derivative.
We pool unrotated K/V; the prefix K then receives RoPE at the last included token
and nanochat QK normalization/scaling. Values are not rotated. There is no
learned block-position bias table in this port.

A query attends with **one softmax** to:

1. Recent W=256 raw KV, including the current token.
2. Completed chunks from its observed prefix.
3. The current unfinished chunk's prefix summary, including only observed tokens.

The third entry is a DMC-style adaptation applied equally to Fixed and Dynamic.
It differs from DeepSeek HCA's completed-only history. Without it, an arbitrarily
long dynamic chunk would hide its older tokens after they leave SWA. This design
needs only a constant-size accumulation state, not an unbounded raw tail.
Local raw KV may overlap summarized information in both modes.

When token t starts a new chunk, the previous prefix ending at t-1 becomes an
immutable summary and a fresh prefix begins with t. Otherwise it updates the
current prefix. A training candidate j is visible to query t>j only if j+1
starts a chunk; the query's own candidate is always visible. Thus no earlier
query sees a future version of a prefix or a future routing decision. Future
candidates are excluded from numerical softmax stabilization as well.
Ending a prefill call or sequence does not add a synthetic closure.

## API, training and actual cache storage

Training stores all intermediate prefix candidates and uses dense masked
attention plus a Python recurrence. This is a correctness reference, not a
training memory or speed optimization. Fullgraph compilation is rejected for
HCA. Checkpoint recomputation returns auxiliary statistics explicitly, preserving
Gumbel RNG; it does not rely on mutable module attributes for loss gradients.

Inference uses `HCAKVCache`: W raw KV, only the actual completed summaries, one
current prefix KV pair, three accumulation tensors and small counters. It
supports split prefill, token decode, serialization, reset and prefix broadcast.
Equal-length batch rows are supported; padding masks, speculative rollback and
document-reset masks are outside this initial API. Existing packed-data sequence
semantics remain. Use `eval()` and `no_grad()`; cached calls produce logits only.

`cache.statistics()` reports completed chunks, all compressed entries (including
the open prefix), tail lengths, logical bytes and actual backing-storage bytes.
At 2K, Fixed B=16 gives 128 entries, including the unfinished final chunk; with
W=256 and FP32 accumulators, BF16 tensor cache reduction is about 5.29x rather
than 16x. Dynamic entries and byte savings are **measured**, never assumed equal
to Fixed. An untrained near-uncompressed router may even use more cache than
the baseline. Python metadata/model/allocator overhead are separate metrics.

```python
import torch
from nanochat.connection_model import ConnectionConfig, build_connection_model

cfg = ConnectionConfig(variant='baseline', streams=1,
                       attention_mode='dynamic_hca', hca_block_size=16,
                       hca_ratio_loss_weight=1.0)
model = build_connection_model(cfg).prepare(compile_blocks=False)
parts = model(tokens, targets, return_loss_components=True)
parts['total'].backward()  # CE + global compression loss in train mode
print(model.last_hca_metrics)  # detached CE, objective, soft/hard rates, L_CR
model.eval()
with torch.no_grad():
    ce = model(tokens, targets)  # evaluation returns pure CE
    cache = model.make_kv_cache(batch_size=1)
    model(prompt[:, :101], kv_cache=cache)
    logits = model(prompt[:, 101:], kv_cache=cache)
print(cache.statistics())
```

`scripts.connection_train` accepts `--attention-mode`, `--hca-block-size`,
`--hca-ratio-loss-weight`, `--hca-window`, `--hca-temperature`, `--hca-noise` /
`--no-hca-noise`. Pool/router use the existing connector AdamW group; backbone
MuonAdamW is unchanged. Training logs separate CE/objective/compression/rates;
dev/test and checkpoint selection use CE only. Config saves target and weight.

## Bounded validation

Only run on HTCondor workers:

```sh
python -m pytest -v tests/test_hca.py
python -m scripts.connection_hca_gate
```

Checks cover FP64 independent prefix-pooling oracle/all continuous VJPs,
single-sided global penalty/value/gradient direction, free boundary counts and
target-independent inference decisions, nonzero **CE-only** router/pool gradients,
prefix causality/zero future gradients, FP32/BF16 prefill/decode equivalence,
unclosed chunks longer than SWA, serialization/broadcast/reset, backbone RNG,
Gumbel activation checkpoint recomputation, true MuonAdamW next-step restore
and CCE gradients. The hard STE is not finite-difference-tested as an exact
derivative. A same-GPU FineWeb 2K/D512/H4/two-layer comparison reports actual
bytes, actual ratios and unoptimized runtime, and labels untrained diagnostic CE
as unsuitable for quality conclusions. It does not assert target R was achieved.
