"""Shape/RMS adapter for NVIDIA Transformer Engine's unmodified projection.

TE 2.20.2 supplies the Triton forward/dx and cuBLAS weight-gradient operations.
Only the small output normalization is native PyTorch here. All other mHC
operations remain the existing DeepSeek implementation. No silent fallback.
"""
from functools import lru_cache
import hashlib
import os
from importlib.metadata import version
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def nvidia_ops():
    if version('transformer-engine') != '2.20.2':
        raise RuntimeError('NVIDIA projection requires the pinned TE 2.20.2 environment')
    import transformer_engine.pytorch.triton.mhc as ops
    import transformer_engine.common.triton.mhc as kernels
    for module, digest in (
        (ops, '4e71eff5d727625ef188e5ec86c2f91b25baf3742b7798313372b2a022e886da'),
        (kernels, '0045990824e46034a2d3162a4690fec1fa87f880dbbab0f43c546ba29d62ff7f'),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'NVIDIA mHC source mismatch: {module.__file__}')
    return ops


def nvidia_projection(x, phi, *, use_tf32=True):
    # TE's use_tf32 argument controls Triton, not its separate cuBLAS dphi.
    # This must be configured before CUDA/library initialization by the caller.
    if not use_tf32 and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('nvidia_precise requires NVIDIA_TF32_OVERRIDE=0 at process startup for the cuBLAS weight gradient')
    if x.shape[-2] != 4 or phi.shape != (24, x.shape[-2]*x.shape[-1]):
        raise ValueError('NVIDIA projection requires four streams and 24 output rows')
    shape = x.shape[:-2]
    flat = x.reshape(-1, phi.shape[1])
    raw, mean_square = nvidia_ops().mhc_fused_projection(
        flat, phi, use_tf32=use_tf32, norm_weight=None,
        fused_grad_x_acc_buffer=None, use_split_k=False)
    # TE pads its raw output to 32; retain our RMS epsilon and coefficient map.
    normalized = raw[:, :24] * torch.rsqrt(mean_square[:, None] + 1e-6)
    return normalized.reshape(*shape, 24)
