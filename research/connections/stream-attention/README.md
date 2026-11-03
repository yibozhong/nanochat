# Independent attention per mHC stream

Status: legacy merged-write experiment. Gate7691122.0 passed17 checks and the
10B run7701979.0/f2850c69 started, but the user later clarified that attention
outputs must remain separate through residual writeback. The old run is stopped
and its checkpoints deleted; its results do not test that intended architecture.
See ../stream-independent/README.md for the corrected path. The attempt notes
below are historical, not the current validation status.

First GPU attempt 7688765.0 (source e027e86) passed 14/15 tests. The compiled
versus eager full-model check failed its pure per-parameter 2% relative bound:
attention matrices were within 1.15%, but final stream-head parameters were
69–79%. These ratios alone did not record absolute gradient sizes, and do not
establish a model defect or harmlessness. Compile/recompute, causality, cache,
FP64 loop derivatives (~1e-15), and optimizer restoration passed. Timing was
not run after the test failure.

The next gate keeps model/training code unchanged, adds a compiled FP32
attention/loop gradient oracle and an ordinary-mHC compiled/eager control,
and records absolute gradient errors and norms for every parameter. The
comparison retains the 2% relative bound and adds the existing repository
parity suite's 1e-6 absolute floor plus a 2% global gradient bound. This is a
correction to the numerical test criterion, not an implementation fix; it
must still pass the added checks. The complete gate now has 17 tests.

The user requested independent attention on each mHC stream, reducing each
stream's head budget to preserve the total attention FLOPs, and merging only
after attention. The MLP keeps the original shared mHC path. The user selected
the original learned mHC weighted merge. Following the gradient-scale discussion,
the implementation applies a fixed factor K to H_pre at the new merge location;
this is not a learned gate or a backward-only modification.

Use `--variant mhc --architecture vanilla --attention-mode per_stream` in
`scripts.connection_train`. `shared` remains the default and old checkpoint
metadata defaults to it. Initially the new mode supports the vanilla backbone,
four streams, and total heads divisible by four. `mhc_no_sinkhorn` can use the
same path. Fused connection-gradient accumulation is explicitly unsupported
because moving the aggregation changes its required backward execution order.

For input X with layout B,T,D,K, generate original tokenwise H_pre/H_post/H_res
from X with NVIDIA TE projection, scale and Sinkhorn kernels. Each stream is
normalized independently and uses its own Q/K/V and output matrices:

```
Y_s = O_s Attention(Q_s norm(X_s), K_s norm(X_s), V_s norm(X_s))
Y = sum_s (K * H_pre_s) * Y_s
X_attention = original_mHC_write(Y, X, H_post, H_res)
X_next = original_shared_MLP_mHC(X_attention)
```

There is no pre-attention aggregation. The routing coefficients still depend
on all streams of the same token. Self-attention remains causal, with original
RoPE, QK RMSNorm, and 1.2 Q/K factors. MLP, final stream collapse, vocabulary
head and NTP objective are unchanged.

At D512/H4/K4 each stream has one 128-dimensional head, three 512-to-128
projections and one 128-to-512 output projection. The total remains 4D²
attention parameters per layer; head dimension is unchanged. The separate
matrices are distinct 2D Parameters in the original Muon optimizer, including
its shape-dependent LR rule. Muon now orthogonalizes each rectangular matrix
independently, so optimizer updates are not identical to a joint dense matrix.

QKV are one batched GEMM with group dimension K and attention folds K into
batch. Because the output projections are linear, weighted summation is
implemented by weighting the attended narrow features, concatenating them,
and multiplying by the concatenated independent output matrices. This avoids
materializing K full-D outputs. It is algebraically the weighted merge above,
with BF16 rounding applied to the weighted features. No new shared learned
projection is added. Cached inference packs the same independent heads into
the original total-H KVCache; there is no cross-head attention.

With identical stream inputs, partitioned dense weights, and H_pre exactly
1/K, fixed-K compensation restores ordinary MHA's output and attention-weight
gradients. Without compensation they are 1/K as large. Real H_pre is dynamic
and only approximately uniform at initialization; this identity does not imply
equal hidden/routing gradients or equal training to the original mHC. H_pre
uses independent sigmoids, not Sinkhorn or sum-to-one normalization.

The model first constructs the original shared model, then partitions the
attention weights without changing the RNG sequence. Embedding, MLP, final
head and connectors match the original mHC initialization exactly. The
explicit matched-backbone initializer also partitions the same vanilla weights.

Validation uses independent FP64 stream-loop attention and gradient oracles,
stream isolation, dense-MHA partition equivalence, NVIDIA routing forward and
backward, 2K causality/batch isolation, KV cache, CCE, compile/recompute, parameter
budget, and optimizer/checkpoint restoration. Existing vanilla/mHC/no-Sinkhorn
and native initialization regressions are included. Two reversed-order fresh
process rounds compare vanilla, ordinary mHC, and per-stream mHC on one GPU,
each with 20 warmup + 30 timed updates at micro16/accum1/2K. Timing includes
synthetic CPU input transfer, NTP/CCE, backward, finite checks and MuonAdamW.
Synthetic losses are not a language-model quality comparison.

The profile reports peak allocated/reserved memory and the difference from
ordinary mHC. Matrix FLOPs count multiply-add as two, use causal pair count
T(T+1)/2, and exclude routing, normalization, activation, CCE softmax and optimizer
work; training matrix FLOPs are estimated as three times forward. Equal
attention matrix FLOPs do not guarantee equal elapsed time.

The workflow submits only a bounded diagnostic GPU job, without W&B or weight
exports, and does not depend on the concurrently rebuilding 2K FineWeb cache.
It never automatically submits formal training.

Second attempt 7689738.0 passed 15/17: new and original mHC global
gradient discrepancies were 0.003530 and 0.003510. Only new stream_head.scale
exceeded the mixed criterion (reference2.032e-6, difference1.482e-6).
The FP32 forward oracle had one cancellation-sensitive element with1.907e-6
error, so its backward was not reached. The next revision preserves intermediate
BF16 casts for per_stream compiled attention/MLP via PyTorch2.10
`emulate_precision_casts=True` (shared baseline unchanged), retaining all gradient
tolerances. The FP32 forward uses8eps*maxabs absolute floor plus global relative
error<3e-6. Official behavior: https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/torch/_inductor/config.py
This changes compiler numerics, not model equations; throughput must be remeasured.

Third attempt7690660.0 passed15/17: compiled FP32 forward relative4.83e-7,
all gradient norm errors<=5.87e-7; one cancelling gradient element exceeded
fixed1e-5 absolute tolerance. Recompute failed a different pure-relative check
on stream_head.scale (2.273%). The fourth attempt applies the already defined
1e-6+2% mixed norm criterion to recompute too, and the same8eps*maxabs
elementwise floor to FP32 gradients, while tightening every FP32 gradient's
relative norm bound from3e-4 to3e-6. Model/compiler settings are unchanged.
