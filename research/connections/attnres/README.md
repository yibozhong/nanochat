# NVIDIA Block Attention Residuals

User authorized port, validation, then one vanilla+AttnRes10B run on2026-11-06.
No new formal baseline or parameter sweep. Retired experiments stay retired.

The NVIDIA NeMo AutoModel commit in source.json supplies the exact forward and
backward Triton kernels. Only its optional-import decorator import is adapted;
all kernel bodies and launch tuning are unchanged. Autograd and FP32 reference
come from the same commit's kimi_k3/situ.py. Apache2.0 license retained.

Architecture follows kimi_k3/model.py's block recurrence. Our public block size
now uses Moonshot's paper/official README convention: Attention and MLP each
count as one sublayer. Default S=2:8 residual blocks,16 sublayer reads
plus final read, embedding preserved as the first history entry. Completed
blocks are sums of their raw branch outputs. Each read uses per-token softmax
over RMS-normalized history/prefix scores and combines unnormalized values.
History is reset each forward; token positions never share depth state.
No full QKV projections, Sinkhorn, explicit gates, auxiliary losses, or extra
identity bypass. Normal branch pre-norm and vanilla attention/MLP stay intact.

`--attnres-block-size` accepts even S>=2; it maps internally to S//2 complete
Transformer layers. Full AttnRes (S=1) and odd boundaries are not implemented
and are rejected explicitly. This unit conversion preserves the existing
grouping; it does not implement a new architecture. New model/checkpoint
metadata records `attnres_block_size_unit="sublayers"`. `from_checkpoint`
converts legacy unmarked sizes from Transformer layers by multiplying by2,
without modifying the supplied metadata or tensors. Marked new metadata is
not converted again.

Historical `validation.json` and the running10B's immutable source/config stay
as originally recorded: their unmarked block_size1 means S=2, not Full AttnRes.
The running job is not restarted for a units-only change. NVIDIA's vendored
kernel comments retain their upstream NeMo units; kernel code is unchanged.

NVIDIA initialization: score vectors normal(std=.02), score RMSNorm weights1,
epsilon1e-5. Vanilla backbone initialized first and explicitly matched to seed42.
Keep nanochat's zero-initialized attention/MLP output projections; short real
training must pass before launch. Averaging is not exact vanilla residual sum.
The first read has one entry and consequently zero score/norm gradients.

Default model:58,737,664 parameters (+17,408 vs vanilla),8layers,D512,H4,
original ReLU-squared MLP, full attention,2K,32K vocabulary. New vectors use the
existing connector AdamW convention (unembedding LR, no weight decay); backbone
MuonAdamW groups and schedules are unchanged. This is an adaptation to nanochat,
not a complete replication of Kimi's training recipe.

Gate:21 checks including independent FP64 all-VJP reference, zero/tiny/ragged
inputs, locality, block sizes S=2/4/6, matched parameters and optimizer ownership,
2K causality, cache, CCE, branch compile and recomputation, real kernel dispatch,
model gradients against eager reference, optimizer/RNG serialization, and
retained mHC/backend checks. Same-device baseline/new60-step profiles;128 real
NTP steps split into independent processes at25. No W&B or returned checkpoints.
Use largest common microbatch16/8/4 fitting90% of22,000MiB; global batch32768.

After passing, collector validates source/results, commits the report, pushes
and verifies main, then submits one formal10B from seed42, using the tested
source archive and existing FineWeb-Edu2K cache.305176 updates=10,000,007,168
tokens. W&B nanochat/group attnres-10b-20261106; latest500 overwrite plus separate
best; exit85/1800s for scheduler persistence. Formal training is restricted to
H100/H200 with FA3. A100 with FA2 remains allowed for validation only, following
the user's2026-11-06 hardware preference. No test weights enter training. Shared
runtime, data and existing checkpoints untouched.

Automatic watcher polls300s and stops at the first finite formal training step.
Failures stop for attention; no automatic retry or weakened test tolerance.
Initial implementation retains NVIDIA's boundary torch.cat bank creation; peak
training memory and wall time must be measured, not inferred from FLOPs alone.
