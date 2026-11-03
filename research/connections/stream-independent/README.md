# Independent attention writeback, MLP-only aggregation

This corrects the earlier `per_stream` experiment, which merged attention
outputs before broadcasting a shared update. That old 10B run was stopped by
the user; its checkpoints were deleted, and its W&B/logs remain historical.
The old explicit mode stays readable for reproducibility. It is not the new run.

New mode: `--attention-mode per_stream_independent`, vanilla+mHC, full attention,
D512, 8 layers, total4 heads, 4 streams, 2K, micro16/accum1, seed42.
Each stream has independent QKV512-to-128 and O128-to-512 matrices.

```
A_s = O_s Attention_s(norm(X_s))
X_attention[..., r] = sum_s R[s,r] X[..., s] + post[r] A_r
h_mlp, skip, mlp_mixes = original_mhc_mlp_read(X_attention)
y_mlp = shared_MLP(norm(h_mlp))
X_next = original_mhc_mlp_write(y_mlp, skip, mlp_mixes)
```

No attention H_pre aggregation, no sum of A_s before writeback, and no old
four-times merge compensation. H_res and H_post are still generated from the
incoming streams. MLP H_pre is generated from the four updated streams, not
from their pre-attention values. The original MLP read/write and final head
are unchanged. This is not four fully independent Transformer blocks: the
MLP still shares one branch and H_res still exchanges old stream states.

QKV and per-stream output projections use grouped GEMMs; attention folds the
stream axis into batch. The output now materializes four D-vectors as required.
NVIDIA TE projection/scale/Sinkhorn/residual-combine and their backward kernels
are retained. Its shared-output expand kernel computes H_res X with a zero
branch; a small compiled FP32 multiply/add applies post[r]*A_r, returning BF16.
The intermediate residual mix is BF16, so this is not one-rounding arithmetic.

The unused attention pre projection rows/bias/scale are removed from parameters;
the original 24-output TE API is padded with constants. Total59,449,741 params,
65,576 fewer than shared mHC. All retained initial weights match the previous
seed42 construction; QKV/O are stored as the original four D-by-D Parameters, with disjoint
row/column slices used in each stream forward. Autograd assembles full matrix
gradients; the unmodified Muon performs joint orthogonalization, row
equilibration, momentum, factored second moments, renormalization and cautious
weight decay with the original square-matrix LR factor1. MLP/head/embedding
optimizer rules remain unchanged. This controls the optimizer transformation,
not the different gradients caused by the architecture. New metadata explicitly distinguishes this
mode, and old merged-write checkpoints cannot silently resume into it.

Validation: 36 GPU checks including FP64 attention/routing gradient oracles,
stream output/gradient isolation, a writeback test that detects broadcasting,
and capture of the four independently updated streams at the MLP entrance.
An identical-gradient control checks exact shared/new Muon weights and states
across warmup, momentum-ramp and warmdown steps. Includes 2K causality/cache/CCE/eager-compile/recompute/optimizer-RNG restore
and retained vanilla/native/shared/no-Sinkhorn regressions. Two reverse-order
same-GPU profile rounds compare vanilla/shared mHC/new independent write.
A real 25 -> exit85 -> fresh process -> 50 restart uses the existing 2K prefix
and the full305176-step LR horizon. No validation W&B or exported weights.

Authorized formal experiment: ONE new run from zero for305176 updates,
10,000,007,168 tokens, original FineWeb-Edu token rows/dev/test and default LR.
Only after GPU checks, profile, restart and publication pass. W&B project
nanochat/group mhc-attn-independent-10b-20261103. Latest every500 steps,
best separately, durable exit85 every1800s; public soft preference with
backfill allowed. Do not restart the old run or reuse its optimizer state.

Status: implementation prepared; GPU validation and formal submission pending.
