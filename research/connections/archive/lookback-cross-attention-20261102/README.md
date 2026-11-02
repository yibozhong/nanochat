# Shared Lookback KV experiment

Two authorized runs train from seed42 for 305176 updates (10,000,007,168
tokens), batch32768, on freshly packed FineWeb-Edu16K examples. The baseline
is the existing eight-layer D512 full-attention vanilla model. Lookback uses
eight sequential2048-token chunks; each chunk has ordinary full causal
attention internally and unrestricted access to the available historical bank.

The shared bank contains one independently projected K/V pair per historical
token. Every backbone layer reads the same historical tensors while retaining
its own Q/K/V for current tokens, in one joint attention softmax. Historical
per-layer KV is not retained. Absolute RoPE positions remain original token
positions; QK RMS normalization and1.2 scaling match the backbone.

After C2 finishes producing its outputs, one shared cross-attention module
updates raw final hidden H1 using Q=norm(H1), K/V=norm(H2), and replaces C1's
bank entry with shared projections of H1+CrossAttention(H1,H2). C2's raw final
hidden is appended. C3 then reads revised C1 and original C2 at every layer.
Each preceding chunk is revised exactly once by its immediate successor.
Updates are functional: earlier predictions and autograd inputs are immutable.
There is no detach, auxiliary objective, or cross-sequence persistent state.
The last boundary also executes once; its write has no later NTP consumer.

There are six additional bias-free D×D matrices: two shared memory projections
and four lookback cross-attention projections. Lookback output projection
starts at zero, like the existing backbone residual branches. Backbone,
embedding and LM-head parameters exactly match the baseline initialization.
All added matrices use the existing Muon parameter group and schedule.
Parameters: vanilla58,720,256; Lookback60,293,120. Disabling
`--lookback-enabled` provides the no-lookback shared-memory baseline (not an
additional formal run in this experiment).

`scripts.connection_train` accepts `--variant lookback --sequence-length 16384
--chunk-size 2048`. Standard token-incremental KVCache is deliberately rejected;
this implementation constructs its separate sequence-local bank per forward.
Chunks may end partially and their number is configurable through the context
and chunk lengths. There is no sliding window inside chunks.

The data job repacks original documents with the native BOS best-fit loader.
It preserves tokenizer, train shards000–012 and heldout shard013, but produces
new16K dev/test examples. Both new runs share these samples and data order.
Old2K results are not matched evaluation controls. Token count is an actual
training budget, not a claim of10B unique source tokens. Old2K caches were
deleted at the user's request; old diagnostic launchers referencing them must
not be rerun without rebuilding their inputs.

`workflow.py WORKFLOW_DIRECTORY` collects CPU data jobs, checks12 GPU tests,
profiles both models twice in opposite order on one GPU, and performs each
variant's25→exit85→fresh-process50 resume (305176-step LR horizon throughout).
The independent small validation cache must match the full cache's first128
steps and dev/test hashes. Only after all checks pass does it publish the
report/source and submit exactly two formal jobs. Tests do not receive W&B
credentials or return weights. Formal runs use the nanochat project, group
lookback-16k-10b-20261102, retain latest500/best separately, and persist via
checkpoint exit85 every1800seconds. Public GPUs have soft preference;
backfill remains allowed with resume enabled.

Profiles choose common microbatch2 or1 to fit90% of22000MiB; accumulation
preserves batch32768. No automatic architecture or activation-checkpoint
fallback is allowed. Logged FLOPs are analytical matrix estimates, not
hardware counters; report exact exclusions. Logical bank storage is64MiB at
microbatch2, but end-to-end autograd retains earlier versions. Actual peak
allocated/reserved and their same-GPU differences are measured separately.
No inference-memory reduction or quality advantage is assumed.

## KV-residual update (2026-11-02)

The new explicit configuration is `--lookback-update kv_residual`. Legacy
`hidden` remains available for existing runs and checkpoints. The new update is:

```text
M2 = SharedKV(H2)
U1 = W_o Attention(Q(H1), M2.K, M2.V)
(deltaK1, deltaV1) = SharedKV(U1, positions=C1)
M1_updated = (M1.K + deltaK1, M1.V + deltaV1)
C3 reads [M1_updated, M2]
```

Here U1 is the cross-attention branch output (the user's `H1_updated`), with
the residual addition placed at the memory output. Query uses the previous
hidden's RMSNorm, learned Q projection and original-position RoPE/QK norm.
Current memory K/V are computed first and reused directly: no new K/V
projection or repeated rotary transformation inside Lookback. Delta projection
reuses the shared weights and existing input RMSNorm/key RMSNorm/RoPE policy;
there is no extra normalization after the K/V residual addition. All previous
causal availability and once-only revision rules remain unchanged. In
particular H1 is not added a second time inside the delta. Zero initialization
of W_o produces exact zero delta and preserves the unmodified memory bank.

This removes the two private Lookback K/V matrices, reducing parameters from
60,293,120 to59,768,832. The query/output and shared memory matrices still use
native Muon. Checkpoint metadata records the update type; absent metadata means
legacy `hidden`. Old weights cannot silently load as the new model. Existing
formal jobs use their immutable old source bundles and remain the old variant.

`kv_workflow.py WORKFLOW_DIRECTORY submit` submits only a bounded validation
job; `... watch` collects and publishes after success. It checks25 tests,
profiles hidden/KV-residual in both orders, and verifies the new variant's
25→exit85→fresh-process50 recovery with the original305176-step LR horizon.
It never submits or cancels formal training, receives no W&B credentials,
and returns no test weights.
