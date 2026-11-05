# Compact read/write routing

Optional `--compact-routing` for `mhc_stream_mlp` and `mhc_channel_no_sinkhorn`.
The default is false. Existing checkpoints and currently running immutable jobs
retain the original NVIDIA implementation. No new formal experiment is submitted.

Only the eight read/write coefficients are computed. The fused forward combines
projection, mean-square normalization, learned scale/bias and sigmoid, returning
four H_pre and four H_post coefficients. The mathematical formula, FP32 epsilon,
parameter shapes/init, TF32/TF32x3 policy and BF16 activations remain unchanged.
NVIDIA's aggregation and the existing variant-specific write/mixing stay intact.
The final stream collapse continues to use the original NVIDIA implementation.

The projection uses 16-column tensor-core tiles instead of the official32.
Removing the Python padding alone would not avoid that official kernel work.
Eight alignment lanes remain; this is not a claim of perfect hardware utilization.
A100 uses ordinary loads; Hopper uses tensor descriptors/TMA. Gradients include
the norm derivative and all routing parameters. Parameter reductions use explicit
partial buffers rather than atomic adds. No graph-breaking optimizer is involved.

The GPU gate checks FP64 forward/all VJPs in TF32 and precise modes, zero/tiny/
ragged inputs, deterministic repeated gradients, NVIDIA read-path parity for
both variants, matched initialization, full2K causality and recompute gradients,
then measures old/new routing and whole-model throughput on the same GPU.
A25->exit85->fresh-process50 checkpoint test retains the305176-step LR schedule.
Only local numeric reports return; no W&B credentials, weights or formal runs.
No speedup is claimed before the gate completes; this stays opt-in.
