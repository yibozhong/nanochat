# Content-selected layer feedback

Enable `--latent-feedback --feedback-source layers --feedback-attention-backend triton`.
The existing `--feedback-source top` and original baseline remain available.

At position t, the current normalized token embedding produces a full-width
query. It attends over the previous position's outputs from all complete
Transformer blocks, using their RMS-normalized states as both keys and values:

```
q[t] = Wq normalized_embedding[t]
v[t-1,l] = RMSNorm(h[t-1,l], epsilon=1e-5)
a[t,:] = softmax_over_layers(dot(q[t], v[t-1,l]) / sqrt(D))
input[t] = normalized_embedding[t] + Wo sum_l(a[t,l] * v[t-1,l])
```

The first position has no feedback. Current token t is already an input; its
output predicts token t+1. Both projections are D by D, with no low-rank
bottleneck, no K/V projections, and no change to the LM head. D512 adds 524,288
parameters. Initialization matches the top-feedback backbone and output
projection; the query projection uses normal standard deviation .02.

Training keeps the existing seed/step 15% P1 / 85% P2 sampler. Only preliminary
passes collect the layer bank, without autograd; the final pass computes CE and
backward. Feedback gradients reach Wq, Wo and the final-pass embedding, while
the previous pass's layer bank is detached. Default evaluation uses P4 and
separately records P1. Cached decoding retains all previous-token layer states,
with exact token recurrence and reset/prefill support.

The Triton operator performs RMS normalization, dot products, layer softmax
and weighted reduction in FP32 in one program per token, returning the model
dtype. Backward computes only dQ, recomputing normalized states and reusing
FP32 probabilities. It never materializes a normalized full-sequence layer bank
or a gradient for that bank. Query/output GEMMs use the existing Linear and
torch.compile path. The detached raw BF16 bank at B16/T2048/L8/D512 occupies
256 MiB; temporary collection and other activations are additional.

The same-allocation preflight runs 12 existing top-feedback regression checks,
27 new layer-feedback checks, a production-shape operator benchmark and 96
steps each of baseline/top-feedback/layer-reference/layer-Triton. It verifies
finite gradients, query learning, strict causality, sequential-cache agreement,
single-pass conditional optimizer behavior, and actual CLI checkpoint transport.
Checkpoint state and a next fixed-gradient update must match bitwise; independent
FA2/CCE gradient reduction nondeterminism is recorded separately.

Formal training starts only after all checks pass, in a new process with fresh
seed42 weights, optimizer, RNG and data position. No diagnostic W&B run is
created. It uses the existing FineWeb-Edu packed2K cache, 305,176 updates at
32,768 tokens/update, P4 dev/test, online W&B and 1800-second exit85 checkpoint
transport. Failures stop the automatic chain and retain the allocation for
900 seconds of debugging. No automatic resubmission or threshold relaxation.
