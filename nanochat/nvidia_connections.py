"""Complete mHC pipeline using unmodified NVIDIA TE 2.20.2 operations.

Persistent activation layout is [batch, time, channel, stream]. TE treats the
two leading axes as independent tokens, so no batch/sequence transpose is
needed. Projection parameters use the same interleaved channel/stream order.
TE's FP32 machine epsilon and log-space Sinkhorn are retained as published;
this is not bitwise identical to DeepSeek's epsilon-regularized arithmetic.
"""
import os

import torch
from torch.nn import functional as F

from nanochat.manifold_connections import ManifoldHyperConnection, ManifoldHyperHead
from nanochat.nvidia_projection import nvidia_ops


def interleave_phi(phi, streams=4):
    """Convert DeepSeek stream-major projection columns to NVIDIA's layout."""
    return phi.reshape(phi.shape[0], streams, -1).transpose(1, 2).reshape_as(phi).contiguous()


def expand_streams(x, streams=4):
    # TE has no standalone embedding replication API. Ordinary autograd sums
    # these four input paths; all five mHC operations below are NVIDIA kernels.
    return x.unsqueeze(-1).expand(*x.shape, streams).contiguous()


def check_precision(use_tf32):
    if not use_tf32 and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('NVIDIA precise mode requires NVIDIA_TF32_OVERRIDE=0 at startup')


class NvidiaHyperConnection(ManifoldHyperConnection):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False):
        super().__init__(width, streams, branch_index, iterations)
        self.use_tf32, self.fused_grad_acc = use_tf32, fused_grad_acc
        if streams != 4 or iterations != 20:
            raise ValueError('The NVIDIA composite API requires four streams and 20 iterations')

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.phi.copy_(interleave_phi(self.phi, self.streams))

    def read(self, x):
        check_precision(self.use_tf32)
        x = x.contiguous()
        # The documented backward order is write -> aggregate -> projection.
        # write initializes this per-invocation buffer; the latter two add to it.
        # No module-global buffer is shared across invocations/checkpoints.
        buffer = (torch.empty_like(x, dtype=torch.float32)
                  if self.fused_grad_acc and torch.is_grad_enabled() and x.requires_grad else None)
        branch, post, comb = nvidia_ops().mhc_generate_mix_and_aggregate(
            x, self.phi, self.scales, self.bias.unsqueeze(0),
            norm_weight=None, use_tf32=self.use_tf32,
            fused_grad_x_acc_buffer=buffer, use_split_k=False)
        return branch, x, (post, comb, buffer)

    def write(self, output, residual, mixes):
        post, comb, buffer = mixes
        return nvidia_ops().mhc_fused_expand_combine(
            output.contiguous(), None, post, residual, comb, self.streams,
            use_tf32=self.use_tf32, fused_grad_x_acc_buffer=buffer)


class NvidiaHyperHead(ManifoldHyperHead):
    def __init__(self, width, streams, *, use_tf32=True):
        super().__init__(width, streams)
        self.use_tf32 = use_tf32

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        self.phi.copy_(interleave_phi(self.phi, self.streams))

    def forward(self, x):
        check_precision(self.use_tf32)
        ops = nvidia_ops()
        b, t, d, n = x.shape
        # TE has no dedicated final head API. Its projection/scale/aggregate
        # implement it by padding unused post/residual coefficients to zero.
        phi = F.pad(self.phi, (0, 0, 0, 24-n))
        alpha = F.pad(self.scale, (0, 2))
        beta = F.pad(self.bias, (0, 24-n)).unsqueeze(0)
        raw, ms = ops.mhc_fused_projection(x.reshape(b*t, d*n), phi,
            use_tf32=self.use_tf32, norm_weight=None,
            fused_grad_x_acc_buffer=None, use_split_k=False)
        pre, _, _ = ops.mhc_fused_scale(raw, alpha, beta, ms, n)
        return ops.mhc_fused_aggregate(x, pre.reshape(b, t, n), n,
            use_tf32=self.use_tf32, fused_grad_x_acc_buffer=None)
