# NVIDIA Block Attention Residuals

User authorized port, validation, then one vanilla+AttnRes10B run on2026-11-06.
No new formal baseline or parameter sweep. Retired experiments stay retired.

The NVIDIA NeMo AutoModel commit in source.json supplies the exact forward and
backward Triton kernels. Only its optional-import decorator import is adapted;
all kernel bodies and launch tuning are unchanged. Autograd and FP32 reference
come from the same commit's kimi_k3/situ.py. Apache2.0 license retained.

Architecture follows kimi_k3/model.py's block recurrence, with block size in
complete Transformer layers. Default1:8 Transformer blocks,16 sublayer reads
plus final read, embedding preserved as the first history entry. Completed
blocks are sums of their raw branch outputs. Each read uses per-token softmax
over RMS-normalized history/prefix scores and combines unnormalized values.
History is reset each forward; token positions never share depth state.
No full QKV projections, Sinkhorn, explicit gates, auxiliary losses, or extra
identity bypass. Normal branch pre-norm and vanilla attention/MLP stay intact.

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
inputs, locality, block sizes1/2/3, matched parameters and optimizer ownership,
2K causality, cache, CCE, branch compile and recomputation, real kernel dispatch,
model gradients against eager reference, optimizer/RNG serialization, and
retained mHC/backend checks. Same-device baseline/new60-step profiles;128 real
NTP steps split into independent processes at25. No W&B or returned checkpoints.
Use largest common microbatch16/8/4 fitting90% of22,000MiB; global batch32768.

After passing, collector validates source/results, commits the report, pushes
and verifies main, then submits one formal10B from seed42, using the tested
source archive and existing FineWeb-Edu2K cache.305176 updates=10,000,007,168
tokens. W&B nanochat/group attnres-10b-20261106; latest500 overwrite plus separate
best; exit85/1800s for scheduler persistence. HopperFA3 or A100FA2; unseen family
passes the21 checks and25→60 local NTP recovery before formal work. No test
weights enter training. Shared runtime, data and existing checkpoints untouched.

Automatic watcher polls300s and stops at the first finite formal training step.
Failures stop for attention; no automatic retry or weakened test tolerance.
Initial implementation retains NVIDIA's boundary torch.cat bank creation; peak
training memory and wall time must be measured, not inferred from FLOPs alone.
