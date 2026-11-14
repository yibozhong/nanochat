# Projected previous-token layer feedback

Two opt-in variants extend the existing `--latent-feedback --feedback-source layers` path:

| `--feedback-source` | Query and feedback location | Independent parameters |
| --- | --- | --- |
| `embedding_qkv` | Normalized token embedding, before block 0 | One full-width Q/K/V/O set |
| `per_layer_qkv` | Current hidden state, before every block; RMSNorm before Q | One full-width Q/K/V/O set per block |

Each reader attends over **all block outputs of the previous token**, from the previous parallel pass. It does not attend over the current token's bank or over the entire token history. The ordinary causal self-attention remains unchanged.

For destination block j, with previous-pass bank H:

```
bank = RMSNorm(stop_gradient(H))
q = Wq[j] * RMSNorm(current_hidden)
k = Wk[j] * bank[previous_token, all_source_layers]
v = Wv[j] * bank[previous_token, all_source_layers]
current_hidden += Wo[j] * (softmax(q * k / sqrt(width)) * v)
```

The input embedding is already normalized, so the first reader does not normalize it a second time. Each set shares its K/V weights over source layers; different destination blocks have separate weights. All projections have full model width, no biases, and normal std .02 initialization. The detached bank's normalization uses FP32 RMSNorm, eps 1e-5, then returns to the activation dtype. K/V projection happens **after** detach, so both matrices receive gradients. First-token feedback is exactly zero.

Training retains the existing 15% one-pass / 85% two-pass sampler. Earlier passes use no_grad; only the final pass and final LM head backpropagate. Eval defaults to four parallel passes. One-pass updates exactly follow the vanilla backbone and skip feedback momentum/weight decay. Baseline, top-only feedback and unprojected layer feedback remain available.

Cached decoding uses exact token recurrence through `FeedbackKVCache`, including sequential prompt prefill. This converges with parallel inference when passes cover the sequence length; finite P2/P4 are the existing training/evaluation approximation. Cache reset and batched branch prefill preserve the previous token's complete layer bank. No KV-cache saving is claimed or implemented.

The default `--feedback-projection-order reassociated` avoids materializing the projected bank. With row-vector convention:

```
q * (bank * Wk.T).T = (q * Wk) * bank.T
attention_weights * (bank * Wv.T) = (attention_weights * bank) * Wv.T
```

Q/K/V/O remain four independent, full-width trainable matrices, including their original optimizer state. Only multiplication order changes. The depth read reuses the existing Triton kernel, skipping normalization because the bank was already normalized once for all readers. It fuses shifted gather, scores, softmax and mixing; backward writes only the effective-query gradient. Autograd through the tokenwise K and V GEMMs supplies their parameter gradients. No projected K/V bank or dK/dV bank is written. This is exact over real numbers; FP64 output/all-gradient tests and BF16 full-model comparisons cover numerical effects of reassociation.

`--feedback-projection-order materialized` retains the initial implementation for comparison: K/V projections use one packed GEMM with separate parameters. Its Triton backward writes dQ/dK/dV without atomics, including zero gradients for unused final bank rows. `--feedback-attention-backend torch` selects independent PyTorch attention references for either order. GEMMs use the native backend.

At depth 8, width 512, additional parameters are 1,048,576 (`embedding_qkv`) and 8,388,608 (`per_layer_qkv`). Materialized order uses 18 tokenwise-GEMM equivalents per reader forward; reassociated order uses 4, retaining full parameters. At B16/T2048 the latter also removes the 512 MiB projected bank per reader (4 GiB for eight readers), plus its backward buffer. These are analytic counts; `connection_cost.py` distinguishes both implementations, and debug measures actual step time and peak memory.

Validation: `research/connections/projected-feedback/submit_debug.py` submits one bounded diagnostic job for both variants, with no W&B or formal training. It checks FP64 operator/reassociation oracles, all gradients, baseline identity, detach semantics, independent layer parameters, causality, cached recurrence, conditional optimizer state, production-shape finite training and real CLI exit85/resume. Four main configurations train 96 diagnostic steps each; the two materialized comparisons use 48. Component forward/backward timings separate bank normalization, projections and attention. One warmed ordinary update per configuration records a compressed CUDA trace, kernels, launch gaps, copy/allocation operators and nested backbone/reader/head/optimizer times; instrumented updates are excluded from throughput and memory peaks. P2 timing compares projection orders without differing P1 fractions. Failed checks stop the chain and retain the allocation for 900 seconds. Existing formal jobs are unaffected; no increase to the 2400-second debug time limit.
