# Fixed HCA / free Dynamic HCA

Status: reference plus opt-in Triton implementation. Source `3ed3470` passed the
full H100 startup gate. Fixed B16 and separately authorized B4 formal pretraining
are running. Dynamic stopped on a nonfinite update and remains held; see the
[training record](../hca-train/README.md). No completed 10B or matched-budget
quality result. The initial hard-quota
version was stopped; its checkpoints are explicitly rejected for HCA loading.

### Dynamic stability diagnosis and alternatives, 2026-11-13

The original Dynamic run completed update 148 and rejected update 149 before
the optimizer ran. Its durable checkpoint is update 50. Two bounded diagnostic
trajectories from that checkpoint reached 301 and 1501 without reproducing the
nonfinite event. Update 51's forward loss matched the original; later updates
diverged under the previously isolated CCE backward nondeterminism. This is not
a repaired or stable 10B result. Optimizer tensors at diagnostic step 1501 were
finite, while loss had deteriorated to roughly 7–8.

One conditioning problem is independently confirmed. For one pooling channel,
at hard start=1, the derivative of the pooled value with respect to start is:

    prior_mass / current_weight * (current_value - prior_pooled_value)

Exponential channel weights and a hard reset can make this multiplier very large.
On a diagnostic update, layer 0's largest mass ratio was 47,568,932; its boundary
gradient reached 7,404,190.5 in Triton and 7,404,186.15 in the independent FP64
reference. Layer 7 had mass ratio 223,359,568, upstream magnitude 8.25e-5, and
boundary gradient 296.56. Thus the measured amplification is in this estimator's
mathematics, not merely FP32 roundoff. It is a plausible instability mechanism,
not proof of the unrecorded original failing backward. No clipping, threshold
relaxation, or silent formal restart was used.

The debug evidence, isolated operator inputs, and profiles are under
`/home/yzhong239/hca-10b-20261112/debug-dynamic/results/dynamic-diagnosis/`.
`NonfiniteTrainingError` now identifies the rejected update/data row, separates
nonfinite loss from gradients, lists affected parameter names, and writes a
small `failure.json`. Three GPU tests verify the failure and successful paths.

Official dynamic-chunking designs reviewed:

- **H-Net:** the [routing/chunking source](https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py)
  uses adjacent projected cosine similarity, hard boundary selection, and direct
  selection of encoder vectors. Dechunking uses a probability-weighted EMA,
  clamped probabilities, and the Mamba2 scan kernel. The
  [model source](https://github.com/goombalab/hnet/blob/main/hnet/models/hnet.py)
  places a confidence STE on the reconstructed output. Its
  [ratio loss](https://arxiv.org/html/2507.07955v1#S2.SS3.SSS2)
  couples actual and soft keep rates around a target, without a fixed quota.
  This is a hierarchy with chunk selection/dechunking, not our weighted KV reset.
- **BLT:** the [official patcher](https://github.com/facebookresearch/blt/blob/main/bytelatent/data/patcher.py)
  thresholds entropy from a separate prediction model; entropy computation
  defaults to no-grad. Boundaries adapt to content without differentiating the
  main LM through this hard threshold. It offers a simpler stability reference,
  at the cost of a separate patcher and a different training setup.
- **Dynamic Token Pooling:** the [boundary predictor](https://github.com/PiotrNawrot/dynamic-pooling/blob/main/hourglass.py)
  provides a relaxed-Bernoulli hard STE with a binomial prior, or supervised
  entropy/unigram boundaries. Its [pooling code](https://github.com/PiotrNawrot/dynamic-pooling/blob/main/shortening.py)
  averages groups and delays their use to the following group to preserve
  causality. This is a useful reference for supervised causal chunking.

Our inference from this comparison: H-Net's smoothing and confidence-gradient
path is the most relevant end-to-end alternative to investigate. Merely swapping
our scalar router for cosine similarity would leave the problematic pooling
reset derivative intact. Adapting its complete gradient path to per-chunk KV
pooling would change the estimator and requires separate validation; it is not
an already implemented or proven fix. DMC itself notes difficulties with
[training from scratch](https://arxiv.org/html/2403.09636v2#A8), unlike its main
retrofitting setting. That observation does not establish our NaN's cause.

### Measured execution overhead

The current kernels are an independent correctness-first implementation informed
by NVIDIA/DeepSeek source, not their complete optimized kernels. Same-H200
profiling over eight layers measured baseline FlashAttention forward/backward
at 8.033 ms and Fixed B16 attention at 20.815 ms, including 13.198 ms for dKV.
Fixed adds 5.462 ms of FP32 score-projection GEMMs and 2.110 ms of pooling.
Dynamic dKV takes 48.209 ms and pooling 5.033 ms. These are individual GPU kernel
durations, not a cross-card attribution of end-to-end wall time.

Fixed pooling parallelizes independent chunks. Dynamic parallelizes within
64-token scans and across batch/head/channel, but carries state serially between
time tiles. Attention is tiled and parallel. Remaining cost comes from extra
operations, FP32/TF32x3 products, intermediate traffic, and implementation
efficiency—not from Triton intrinsically. Fixed still builds all token prefixes
and shares precision choices motivated by Dynamic. Those are concrete future
optimization targets, not measured speedups.

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
training/backward implementation. Training was blocked at that point pending
the execution repair and validation, now completed as recorded above.

### NVIDIA source review and execution repair, 2026-11-13

Directly reviewed NVIDIA's [Megatron CSA/HCA model code](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/experimental_attention_variant/csa.py),
[sparse-attention adapter](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/experimental_attention_variant/csa_utils/fused_sparse_attention.py),
and cuDNN Frontend's [Compressor kernels](https://github.com/NVIDIA/cudnn-frontend/blob/develop/python/cudnn/csa/compressor/compressor_sm100.py)
and [API support checks](https://github.com/NVIDIA/cudnn-frontend/blob/develop/python/cudnn/csa/compressor/api.py).
The fused Compressor is CuTe DSL, with both forward and backward; it is not
itself a complete HCA attention implementation. Its public validated ratios
are 2, 4 and 128, not 16. Ratios 2/4 support SM90 and capability major >=10;
ratio 128 requires major >=10 and head dimension 128 or 512. The generic
ratio-2/4 API does not share that two-value head-dimension restriction.
The filename `compressor_sm100.py` therefore does not imply Blackwell-only.
These are the reviewed development branch's checks, not a promise about an
installed cuDNN package.

The useful implementation choices are:

- Each output chunk/channel owns its pooling work. Compute the channel softmax
  in FP32 and round the output once, avoiding token-by-token Python dispatch.
- Recompute probabilities in backward rather than saving expanded pooling
  intermediates. The nonoverlapping input-gradient stores have unique owners,
  avoiding atomics for dKV/dScore. NVIDIA also handles unused tail gradients
  explicitly, including segments shorter than one chunk.
- Shared position-bias gradients need a reduction. NVIDIA exposes both an
  atomic path and a deterministic partial-buffer/fixed-order reduction path;
  deterministic training is not intrinsically excluded by its Compressor.
- The model's attention uses local raw positions plus causally valid compressed
  positions in a common normalization. The fused adapter uses FlashMLA forward
  and cuDNN DSA backward with additional layout/head/alignment requirements.
  This is not a drop-in kernel for nanochat's separate K/V, four heads, D128.

The same-H200 diagnosis (source `01ddae1`, microbatch 8, global 32768 tokens,
8 layers, D512/H4/T2048, two warmed updates) measured baseline 0.08165 s/update,
Fixed 15.0530 s/update, Dynamic 21.2035 s/update. Each HCA update dispatched
32768 Python pooling calls. Forward routing/pooling/mask preparation alone
took about 5.07 s for Fixed and 6.72 s for Dynamic on the CUDA timeline, which
includes launch gaps. This is execution overhead, not a time-to-quality result.
For completed fixed chunks alone, the sequential pooling forward+backward
took 682.4 ms versus 0.661 ms for the vectorized official pattern; FP64
outputs and VJPs matched. That component speedup is not an end-to-end claim.

A Triton implementation can target our R16/H4/D128 geometry on Hopper without
porting MLA, FP8/FP4, or the CSA indexer. The implementation sequence is:

1. Use batched PyTorch pooling as an oracle, then implement nonoverlapping
   per-channel softmax pooling forward/backward for separate K and V with
   shared weights. For upstream gradients gK/gV, dP = gK*K + gV*V and
   dScore = P*(dP - sum(P*dP)); dK=P*gK and dV=P*gV. Do not use BF16-rounded
   pooled outputs as the FP32 softmax-backward reduction. Validate tails,
   extreme scores, FP32/BF16 VJPs and repeatability before selecting it.
2. Implement tiled attention over the local window and compact completed
   summaries with ONE online softmax/LSE and a corresponding backward. Do not
   add two separately normalized attention outputs. Resolve the declared
   completed-only versus visible-open-prefix semantics in the oracle, cache
   and training together; an attention kernel alone does not fix that choice.
3. Dynamic needs a segmented recurrence and an explicit boundary backward.
   Hard compaction alone drops the current STE visibility gradient at zero
   gates: for a weighted softmax, dL/dgate_j can be nonzero even when gate_j=0.
   Preserving that estimator can require backward work for unselected prefix
   candidates. Test CE-only router gradients against the reference, including
   arbitrarily long chunks, before claiming equivalent sparse training.

At the time of the source review no new Triton kernel was implemented or GPU
validated. Both 10B runs were at zero formal steps. Controlled-CCE-gradient
checkpoint replay passed for both variants; the original uncontrolled replay
still differs because actual CCE backward is numerically nondeterministic.
The diagnostic result has not silently relaxed the formal startup gate.

### Opt-in Triton implementation

`ConnectionConfig.hca_backend = reference | triton` and the training argument
`--hca-backend triton` select `nanochat/hca_triton.py`. Full-attention defaults
are unchanged. This implementation preserves the existing bounded pooling
logits, prefix-end RoPE and visible unfinished summary in both variants; it is
an execution optimization of this adaptation, not a switch to completed-only
DeepSeek semantics or to MLA. Cached inference retains the reference state
machine, with split-prefill parity tested against the accelerated training path.

Fixed computes all within-chunk prefixes in parallel and attention reads only
the completed R16 positions plus its current diagonal prefix. Dynamic computes
affine segmented scans in time tiles and compacts closed indices for forward
and query backward. Its prefix backward also evaluates unselected candidates
to retain the zero-gate visibility derivative. Prefix states cost O(T*D);
Dynamic's counterfactual backward can still cost O(T*T*D), but never stores a
T-by-T score/probability matrix. No maximum dynamic chunk length is introduced.
All input-gradient stores have unique owners; head/channel router contributions
are reduced explicitly rather than atomically. FP64 calls use the reference
oracle. Initial explicit kernel envelope: CUDA FP32/BF16, head dimensions
16..128, sequence up to 8192, fixed block sizes 1..128. Formal scheduling remains
Hopper-only until GPU checks establish broader support.

Validation is `tests/test_hca_triton.py` plus the reference tests, 2K cached
comparison, full-shape finite gradients/replay and same-GPU throughput. Kernel
correctness and repeatability have separate checks; BF16 parity uses measured
relative-L2 and normalized-peak error, not bitwise equality to different eager
reduction orders. The startup contract now requires exact state/RNG/data and
controlled full-update replay, supported by the prior CCE-isolation evidence.
No CCE tape is used in formal training. See the paired-run README for gates.

Interactive H100 debug fixed a Triton loop-carried shape/type reuse in query
backward and the forced-first-boundary graph for T=1. Probability/value and
score-adjoint/input products now retain FP32 operands using TF32x3, avoiding
extra BF16 rounding that distorted a cancelling router-bias gradient. All 24
kernel/integration tests passed on that allocation; full production-shape
validation is a separate gate. For singleton pooling, the score derivative is
analytically zero: its test independently establishes this in FP64 and bounds
both FP32 cancellation residuals by 16 eps times the uncancelled local VJP
scale. Relative error between two near-zero residuals is not a valid metric;
the nonzero-gradient tolerances were not widened.

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
label alone is not evidence of a fair comparison. The subsequent two 10B runs
are authorized in RESEARCH_HISTORY.md but remain gated; neither has started.

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
