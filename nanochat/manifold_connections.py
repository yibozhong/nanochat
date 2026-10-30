"""nanochat containers around official ops, with an explicit PyTorch projection option.

Pinned official release: 36d9e45d38e204ebb87e6f6e833821eee0482fe5.
No locally implemented mHC forward/backward kernel, and no automatic fallback.
Parameter initialization is a documented local choice, not an official recipe.
The default remains official. The optional projection uses ordinary FP32
PyTorch operations/autograd; it does not reproduce official TF32 rounding.
"""
from functools import lru_cache
import math
import os

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=1)
def official_ops():
    # Lazy import keeps the baseline usable in its original environment.
    try:
        from tile_kernels.modeling.mhc import ops
    except ImportError as exc:
        raise RuntimeError('mHC requires the pinned research/connections/runtime environment') from exc
    return ops


def check_official_device(device):
    if torch.device(device).type != 'cuda':
        raise RuntimeError('Official TileKernels mHC requires CUDA')
    major, minor = torch.cuda.get_device_capability(device)
    if major not in (9,10) and (major, minor) != (12,0) and os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY') != '1':
        raise RuntimeError('Supported: upstream SM90/SM100 and locally validated SM120. Other GPUs require an explicit portability gate; no reference fallback is enabled')
    if major < 8:
        raise RuntimeError('This CUDA 13.1 validation environment requires SM80 or later')


def expand_streams(x, streams):
    return official_ops().expand_to_mhc(x.contiguous(), streams)


def torch_projection(x, phi):
    """Linear projection and per-token RMS; automatic PyTorch backward.

    FP32 intermediates preserve small connection coefficients. With TF32
    disabled this is an accuracy control, not bitwise official-kernel parity.
    """
    flat = x.flatten(-2).float()
    return F.linear(flat, phi) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)


@lru_cache(maxsize=1)
def compiled_torch_projection():
    # Token count varies between training, prefix prefill and cached decoding.
    # A shared static graph reaches Dynamo's per-code recompilation limit.
    return torch.compile(torch_projection, dynamic=True, fullgraph=True)


def projection(x, phi, backend='official'):
    if backend == 'torch':
        return torch_projection(x, phi)
    if backend == 'torch_compile':
        return compiled_torch_projection()(x, phi)
    if backend != 'official':
        raise ValueError(f'Unknown projection backend: {backend}')
    # Official default fused accumulation assumes an externally attached storage
    # gradient buffer. nanochat uses ordinary autograd accumulation instead.
    return official_ops().mhc_pre_norm_fn(x, phi, None, 1e-6, fuse_grad_acc=False, n_splits=1)


class _PortableSinkhorn(torch.autograd.Function):
    """Only a launch adapter; both computations remain official kernels.

    Upstream's 32-token backward tile uses 100 KiB at repeat=20. SM120
    allows 99 KiB per block. A 16-token tile uses 50 KiB, with identical math.
    """
    @staticmethod
    def forward(ctx, x, repeat, eps):
        from tile_kernels.mhc.sinkhorn_kernel import _mhc_sinkhorn_fwd, _mhc_sinkhorn_bwd
        output = torch.empty_like(x)
        ctx.save_for_backward(x)
        ctx.bwd_kernel = _mhc_sinkhorn_bwd(x.shape[1], 16, repeat, eps)
        _mhc_sinkhorn_fwd(x.shape[1], 1, repeat, eps)(x, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        ctx.bwd_kernel(grad_output.contiguous(), x, grad_input)
        return grad_input, None, None


def sinkhorn_normalize(x, repeat=20, eps=1e-6):
    if torch.cuda.get_device_capability(x.device)[0] in (9, 10):
        return official_ops().sinkhorn_normalize(x, repeat=repeat, eps=eps)
    return _PortableSinkhorn.apply(x.contiguous().view(-1, *x.shape[-2:]), repeat, eps).view_as(x)


class ManifoldHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20, *, manifold=True, projection_backend='official'):
        super().__init__()
        self.width, self.streams, self.iterations = width, streams, iterations
        self.manifold = manifold
        self.projection_backend = projection_backend
        self.phi = nn.Parameter(torch.empty(2*streams+streams*streams, streams*width))
        self.bias = nn.Parameter(torch.empty(2*streams+streams*streams))
        self.scales = nn.Parameter(torch.empty(3))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1/math.sqrt(self.streams*self.width))
        n = self.streams
        self.bias[:n].fill_(-math.log(n-1))
        self.bias[n:2*n].zero_()
        self.bias[2*n:].view(n,n).copy_(torch.eye(n, device=self.phi.device))
        self.scales.fill_(.01)

    def read(self, x):
        ops = official_ops()
        x = x.contiguous()
        # The fused inference operation includes Sinkhorn internally. The
        # unconstrained ablation must use the unfused path even in no_grad.
        if self.manifold and self.projection_backend == 'official' and not torch.is_grad_enabled():
            post, comb, branch = ops.mhc_pre_big_fuse(
                x, self.phi, self.scales, self.bias, rms_eps=1e-6,
                mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
                mhc_post_mult_value=2., sinkhorn_repeat=self.iterations, n_splits=1)
        else:
            mixes = projection(x, self.phi, self.projection_backend)
            pre, post, comb = ops.mhc_pre_split_mixes(mixes, self.scales, self.bias, self.streams, 2., 1e-6)
            if self.manifold:
                comb = sinkhorn_normalize(comb, repeat=self.iterations, eps=1e-6)
            branch = ops.mhc_pre_apply_mix(x, pre)
        return branch, x, (post, comb)

    @staticmethod
    def write(output, residual, mixes):
        return official_ops().mhc_post(output.contiguous(), residual, *mixes)


class ManifoldHyperHead(nn.Module):
    def __init__(self, width, streams, *, projection_backend='official'):
        super().__init__()
        self.width, self.streams = width, streams
        self.projection_backend = projection_backend
        self.phi = nn.Parameter(torch.empty(streams, streams*width))
        self.scale = nn.Parameter(torch.empty(1))
        self.bias = nn.Parameter(torch.empty(streams))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1/math.sqrt(self.streams*self.width))
        self.scale.fill_(.01)
        self.bias.fill_(-math.log(self.streams-1))

    def forward(self, x):
        ops = official_ops()
        n = self.streams
        # Same 24-row padding as the official mhc_head; materialize the 4-row
        # slice because the kernel interface requires contiguous tensors.
        phi = F.pad(self.phi, (0,0,0,n*(n+2)-n))
        raw = projection(x.contiguous(), phi, self.projection_backend)[..., :n].contiguous()
        pre = ops.mhc_head_compute_mix(raw, self.scale, self.bias, 1e-6)
        return ops.mhc_pre_apply_mix(x.contiguous(), pre.unsqueeze(-1))
