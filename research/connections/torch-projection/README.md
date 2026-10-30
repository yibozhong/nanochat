# Ordinary PyTorch connection projection

The user asked whether the slow mHC projection can use ordinary PyTorch.
`projection_backend=official` remains the default. Explicit `torch` and
`torch_compile` alternatives replace only the connection projection and RMS:

```python
flat = x.flatten(-2).float()
out = F.linear(flat, phi) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
```

Both alternatives use PyTorch autograd; neither defines a custom backward.
The compiled option applies `torch.compile(fullgraph=True, dynamic=False)` to
these same operations. Input activations remain BF16 and projection weights
and intermediates FP32. The diagnostic disables TF32 for PyTorch matmuls;
the official kernel retains its own explicit TF32 rounding. Consequently this
compares the same mathematical map at different arithmetic precision, not
bitwise equality or a lower-precision speedup. It does not change backbone
precision, the model's parameters, initialization or optimizer.

The choice also applies to the final stream head. Evaluation bypasses the
official fused preprocessing operation when a PyTorch backend is selected,
so no_grad does not silently revert to the original projection. Sinkhorn,
coefficient generation, stream read/write and expansion remain official ops.

The bounded same-GPU experiment first runs the existing mHC tests and new
FP64 output/input-gradient/weight-gradient comparisons, official-kernel
comparisons, seed equality, real 2K causality, cache, CCE and model-gradient
checks. It then measures the real 32768-token projection shape with 10 warmup
and 30 event-timed iterations per backend. Finally it profiles vanilla,
official mHC, eager PyTorch mHC and compiled PyTorch mHC in separate processes
on the same GPU, using the existing 66-update profiling protocol. Every run
records to `nanochat`, group `mhc-torch-projection-20261030`. No checkpoint or
1B pretraining is created. Results are pending.

After validation, an explicit training selection is available as
`--projection-backend torch` or `--projection-backend torch_compile` in
`scripts.connection_train`; existing commands retain the official path.
