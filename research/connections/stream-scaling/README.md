# Frozen-weight stream-count scaling benchmark

One single-GPU job compares mHC and shared stream-axis MLP mixing at
n=4,8,16,32. Per-stream width stays512; MLP inner width is4n (16,32,64,128).
The backbone stays8layers,4attention heads,2048context,32768vocabulary.
mHC keeps20 log-space Sinkhorn iterations; MLP has no Sinkhorn. No optimizer
is constructed, no parameters are updated, no weights are saved, and no
W&B run is created. Model and connector forward/backward timing are not
reported as trained tokens or as convergence evidence.

Pinned NVIDIA TransformerEngine2.20.2 public APIs and its aggregate/write
kernels support only4streams. A benchmark-only generalized implementation
is therefore used for **both** methods at **all** four stream counts.
Both use the same compiled cuBLAS FP32/TF32 projection and RMS scaling,
with FP32 parameters and BF16 residuals. A custom projection backward
recomputes the FP32 input instead of retaining it across depth. Both use
fused Triton read/write, with no full MLP hidden bank saved. MLP uses fused
TF32x3 matrix products and deterministic parameter partial reductions.

The Sinkhorn forward/backward is adapted from the pinned NVIDIA source
`transformer_engine/common/triton/mhc.py`, SHA256
`0045990824e46034a2d3162a4690fec1fa87f880dbbab0f43c546ba29d62ff7f`.
Only the n==4 compiler assumption/autotuner are removed and the token tile
is bounded as n grows. Arithmetic and20iterations remain unchanged.
Apache2.0 copyright/license are retained in NVIDIA-LICENSE. See the
[upstream source](https://github.com/NVIDIA/TransformerEngine/blob/v2.20.2/transformer_engine/common/triton/mhc.py).

Existing NVIDIA mHC4 and optimized MLP4 (rowwise+compact routing) are timed
separately as anchors, never spliced into the generalized scaling curves.
The experiment measures these implementations, not an asymptotically optimal
kernel or an official NVIDIA16/32-stream implementation. Report the anchor
gap and distinguish logical matrix FLOPs from padding, recomputation,
Sinkhorn and actual elapsed time.

30GPU checks cover all four stream counts: FP64 Sinkhorn and all primitive
VJPs, complete connector gradients, comparison with existing4stream paths,
and full-model2K causality/finite gradients/no updates. Timing follows only
after all checks pass. Every configuration runs in its own subprocess.
Each timing series uses8warmups+20CUDA-event samples and a repeated backward
series. Inputs and weights remain fixed. Timing includes NTP CCE/backward
but excludes optimizer steps, data transfer, initialization and compilation.

Primary micro/full-model batch is16. If any full-model configuration OOMs,
rerun **all eight** generalized model cases at batch8, then4 if needed.
Never silently reduce batch for only one method. Record every OOM and
selected common batch. Generic4 and production4 anchors share that batch.
Single full Hopper/A100 GPU>=40000MiB,16GBhost RAM,40GBscratch, public soft
priority/backfill permitted. Return only small logs/JSON/CSV. Existing
formal training jobs and immutable inputs stay unchanged.

`submit.py ROOT` submits once. `collect.py ROOT` checks every300seconds,
audits/publishes the results and exits; it has no training-submission path.
