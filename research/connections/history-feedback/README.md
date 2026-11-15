# Strict-past top-history feedback

`--latent-feedback --feedback-source history_qkv --feedback-attention-backend flash --feedback-detach`

The normalized current token embedding queries **all earlier tokens' top
hidden states from the previous pass**. Full-width Q/K/V/O, four heads of 128
channels at D512, backbone RoPE positions and FP32 RMS-normalized source states
(eps1e-5). The mixed result is added to the input embedding. No current-token
or future-token top state is visible as KV. The first token receives no feedback.

`--feedback-query previous_top` changes only the query source to the previous
pass's **current-position top hidden**, normalized with FP32 RMSNorm eps1e-5.
K/V still contain strictly earlier positions. Current-position information
is available through Q only, and is causal because that top hidden has seen
only the current input and its prefix. The readout still adds to the original
embedding. Default `embedding` preserves the original model/checkpoint behavior.
Q and KV share the source normalization; no parameters or GEMMs are added.
With detach enabled, the source of Q is also detached, while Q/K/V/O weights
all receive gradients. In full-gradient tests, the last top state now receives
Q gradients even though it remains excluded from KV.

Training pairs Q at positions 1..T-1 with KV at 0..T-2, so ordinary equal-length
causal FlashAttention implements strict j<i. There is no dense attention mask
or score matrix, Python token loop, or per-layer history bank in training.
K/V use one packed GEMM while keeping separate square Muon parameters.
`torch.compile(fullgraph=True)` fuses norm/RoPE/packing/addition around the native
FlashAttention operator. The `torch` backend is an explicit numerical oracle.
There are 4D² additional parameters (1,048,576 at D512).

The existing pass loop is reused: sample 15%P1 / 85%P2 per update, final-pass
CE only, P4 evaluation plus P1. Detach is the first authorized 10B setting;
the implementation and gradient tests also support full cross-pass gradients.
Backbone initialization, optimizer, data, token budget and seed are unchanged.

`model.make_kv_cache(batch_size, passes=P)` creates **one backbone KV cache per
pass** and projected top KV between adjacent passes. Whole-prompt, chunked and
single-token inference use exactly the same finite-P computation as parallel
evaluation. Prefill is parallel within each pass. Cached reads exclude the
current position and use lower-right causal alignment for rectangular chunks.
Memory scales with P; this is not the older feedback cache's exact recursive
fixed-point decoding. Reset and batch-one-to-many prefill are supported.

The native lower-right causal bias dispatch in the locked PyTorch2.10 runtime
uses FlashAttention without materializing masks:
[PyTorch2.10 implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/attention/bias.py).
The common full-attention SDPA cache helper uses this for rectangular prefill;
the sliding-window branch retains its existing mask.

One authorized HTCondor job validates then trains from scratch in the same GPU
allocation. Gate: 64 GPU checks (12 original latent +52 history/query checks), production16x2048x512
compiled forward/backward kernel and memory profile, two matched arms each512
finite updates with the same recipe, and CLI exit85/exact recovery checks.
Compiled mix must stay below2GiB peak and within1.3x eager mix; the full P2 step
must stay within1.35x original top feedback for the embedding-query variant.
For `previous_top`, the comparator is embedding-query history attention with
bitwise-identical initialized parameters; its P2 overhead limit is1.15x.
Tests cover both query modes, current-position query versus strict-past KV,
detach/full-gradient, compiled/checkpointed references and identical finite-P
parallel/chunked/tokenwise inference. These thresholds are set before
GPU execution. Gate2400s, failure retains900s debug and stops; no automatic
retry or diagnostic W&B. Success starts a fresh formal process/seed42, never
using diagnostic weights. The 10B run uses d8/D512/T2048, global32768/micro16,
FA2/CCE, compiled blocks and the frozen FineWeb-Edu sample-10BT packed2K cache.

The embedding-query history run was stopped at the user's request. Its results
are retained as a comparator. The replacement authorization creates one fresh
10B with `previous_top`, seed42 and the same detach/4-head/RoPE/training recipe;
it does not load the stopped run's weights. Earlier stopped runs remain stopped.
