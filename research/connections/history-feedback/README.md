# Strict-past top-history feedback

`--latent-feedback --feedback-source history_qkv --feedback-attention-backend flash --feedback-detach`

The normalized current token embedding queries **all earlier tokens' top
hidden states from the previous pass**. Full-width Q/K/V/O, four heads of 128
channels at D512, backbone RoPE positions and FP32 RMS-normalized source states
(eps1e-5). The mixed result is added to the input embedding. No current-token
or future-token top state is visible. The first token receives no feedback.

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
allocation. Gate: 41 GPU checks (12 existing +29 new), production16x2048x512
compiled forward/backward kernel and memory profile, top/history each512
finite updates with the same recipe, and CLI exit85/exact recovery checks.
Compiled mix must stay below2GiB peak and within1.3x eager mix; the full P2 step
must stay within1.35x original top feedback. These thresholds are set before
GPU execution. Gate2400s, failure retains900s debug and stops; no automatic
retry or diagnostic W&B. Success starts a fresh formal process/seed42, never
using diagnostic weights. The 10B run uses d8/D512/T2048, global32768/micro16,
FA2/CCE, compiled blocks and the frozen FineWeb-Edu sample-10BT packed2K cache.

Earlier stopped feedback runs remain stopped. Their results are comparators;
this authorization creates only this new history-feedback10B run.
