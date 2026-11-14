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

K/V projections use one packed GEMM with separate trainable parameters. A Triton forward kernel fuses shifted gather, attention scores, softmax and value mixing. Its backward computes dQ/dK/dV without atomics; each memory row has a single consuming query. The first query zeros the unused final bank row's gradients, without a separate zero-fill launch. Projection GEMMs use PyTorch's native GEMM backend. `--feedback-attention-backend torch` selects the independent reference implementation.

At depth 8, width 512, additional parameters are 1,048,576 (`embedding_qkv`) and 8,388,608 (`per_layer_qkv`). Per-block K/V projections run over the complete source-layer bank, so the latter has additional quadratic-in-depth projection work. `connection_cost.py` records this explicitly, and the bounded debug job measures actual time and peak memory.

Validation: `research/connections/projected-feedback/submit_debug.py` submits one bounded diagnostic job for both variants, with no W&B or formal training. It checks FP64 operator oracles, all gradients, baseline identity, detach semantics, independent layer parameters, causality, cached recurrence, conditional optimizer state, production-shape finite training, same-GPU timing/memory and real CLI exit85/resume. Failed checks stop the chain and retain the allocation for 900 seconds for diagnosis. Existing formal jobs are unaffected.
