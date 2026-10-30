"""nanochat parameter containers around unmodified DeepSeek TileKernels ops.

Pinned official release: 36d9e45d38e204ebb87e6f6e833821eee0482fe5.
No locally implemented mHC forward/backward kernel, and no automatic fallback.
Parameter initialization is a documented local choice, not an official recipe.
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
    if major not in (9,10) and os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY') != '1':
        raise RuntimeError('Upstream documents SM90/SM100. Other GPUs require an explicit portability gate; no reference fallback is enabled')
    if major < 8:
        raise RuntimeError('This CUDA 13.1 validation environment requires SM80 or later')


def expand_streams(x, streams):
    return official_ops().expand_to_mhc(x.contiguous(), streams)


def projection(x, phi):
    # Official default fused accumulation assumes an externally attached storage
    # gradient buffer. nanochat uses ordinary autograd accumulation instead.
    return official_ops().mhc_pre_norm_fn(x, phi, None, 1e-6, fuse_grad_acc=False, n_splits=1)


class ManifoldHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20):
        super().__init__()
        self.width, self.streams, self.iterations = width, streams, iterations
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
        if not torch.is_grad_enabled():
            post, comb, branch = ops.mhc_pre_big_fuse(
                x, self.phi, self.scales, self.bias, rms_eps=1e-6,
                mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
                mhc_post_mult_value=2., sinkhorn_repeat=self.iterations, n_splits=1)
        else:
            mixes = projection(x, self.phi)
            pre, post, comb = ops.mhc_pre_split_mixes(mixes, self.scales, self.bias, self.streams, 2., 1e-6)
            comb = ops.sinkhorn_normalize(comb, repeat=self.iterations, eps=1e-6)
            branch = ops.mhc_pre_apply_mix(x, pre)
        return branch, x, (post, comb)

    @staticmethod
    def write(output, residual, mixes):
        return official_ops().mhc_post(output.contiguous(), residual, *mixes)


class ManifoldHyperHead(nn.Module):
    def __init__(self, width, streams):
        super().__init__()
        self.width, self.streams = width, streams
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
        raw = projection(x.contiguous(), phi)[..., :n].contiguous()
        pre = ops.mhc_head_compute_mix(raw, self.scale, self.bias, 1e-6)
        return ops.mhc_pre_apply_mix(x.contiguous(), pre.unsqueeze(-1))
