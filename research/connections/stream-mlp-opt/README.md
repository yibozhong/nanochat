# Shared stream MLP kernel optimization

This changes the implementation of the shared 4→16→ReLU→4 mixer, not its
architecture, initialization, optimizer, or training protocol. Existing
running jobs retain their immutable sources. No new formal run is submitted.

`stream_mlp_kernel=legacy` remains the default, including for old checkpoints.
The optional candidates are `rowwise` (four split streams, local FMAs),
`tensorcore` (fused TF32x3 dots), and `hybrid` (rowwise forward, dot backward).
They all fuse the branch write in forward and recompute hidden activations
in backward. Small matrices and hidden arithmetic remain FP32; activations
and output remain BF16. No global 16-channel hidden tensor is stored. Dot
tiles pad the four-stream dimension to16; this cost is measured, not ignored.
Weight gradients use deterministic tile partials followed by a reduction.

The rowwise design is informed by the split-stream FMA implementation in
[NVIDIA TransformerEngine v2.20.2](https://github.com/NVIDIA/TransformerEngine/blob/v2.20.2/transformer_engine/common/triton/mhc.py).
Our shared MLP differs from NVIDIA's per-token dynamic4×4 matrix: its weight
gradients reduce across both token and channel rows. The dot backward uses
this structure. Generated PTX is inspected for actual MMA instructions.

One bounded GPU gate runs22 correctness checks, a production-size
B16/T2048/D512 microbenchmark including compiled FP32 PyTorch, reverse-order
timing repeats, and kernel-level CUDA timing. It then measures the best new
candidate against legacy in ABBA whole-model order, with original routing
fixed. A separate combined profile enables the already-validated compact8
routing. Finally,25→exit85→fresh-process50 validates checkpoint recovery.
All measurements use the same GPU and data. No W&B or diagnostic weights
are returned. A100/FA2 and Hopper/FA3 are eligible; one report validates only
the GPU family it actually ran on. A negative performance result is retained.

`submit.py ROOT` submits one immutable gate; `collect.py ROOT` polls every
300seconds, publishes the audited result, and stops. It cannot submit training.
