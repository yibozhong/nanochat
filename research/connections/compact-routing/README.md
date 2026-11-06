# Compact read/write routing

Optional `--compact-routing` for `mhc_channel_no_sinkhorn`.
The default is false. Existing channel-wise checkpoints retain the original
NVIDIA implementation unless this option is explicitly enabled.

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

`tests/test_compact_routing.py` retains the existing FP64 forward/VJP,
determinism, channel-wise NVIDIA read-path parity, initialization, 2K causality,
recompute-gradient and checkpoint-schema checks. Run GPU checks through HTCondor.

`validation.json` records the historical 13-check gate before Stream MLP was
retired. Its whole-model timing and 25-to-50-step restart measurements used that
retired model, so they are not channel-wise speed or restart measurements.
The Stream MLP test cases and its submission/profiling workflows have been
removed; the shared kernel and already-validated channel-wise checks remain.
