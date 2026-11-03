"""Complete mHC pipeline using unmodified NVIDIA TE 2.20.2 operations.

Persistent activation layout is [batch, time, channel, stream]. TE treats the
two leading axes as independent tokens, so no batch/sequence transpose is
needed. Projection parameters use the same interleaved channel/stream order.
TE's FP32 machine epsilon and log-space Sinkhorn are retained as published;
this is not bitwise identical to DeepSeek's epsilon-regularized arithmetic.
"""
import os
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.nvidia_ops import nvidia_ops


def interleave_phi(phi, streams=4):
    """Interleave stream-major random draws, preserving our audited initialization."""
    return phi.reshape(phi.shape[0], streams, -1).transpose(1, 2).reshape_as(phi).contiguous()


def expand_streams(x, streams=4):
    # TE has no standalone embedding replication API. Ordinary autograd sums
    # these four input paths; all five mHC operations below are NVIDIA kernels.
    return x.unsqueeze(-1).expand(*x.shape, streams).contiguous()


def check_nvidia_device(device):
    if torch.device(device).type != 'cuda':
        raise RuntimeError('NVIDIA mHC requires CUDA')
    major, _ = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError('NVIDIA TMA kernels require SM90 or later; no fallback is provided')
    if major not in (9, 10) and os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY') != '1':
        raise RuntimeError('Other GPU architectures require an explicit portability gate')


def check_precision(use_tf32):
    if not use_tf32 and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('NVIDIA precise mode requires NVIDIA_TF32_OVERRIDE=0 at startup')


class NvidiaHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, sinkhorn=True):
        super().__init__()
        self.width, self.streams, self.iterations = width, streams, iterations
        self.phi = nn.Parameter(torch.empty(2*streams+streams*streams, streams*width))
        self.bias = nn.Parameter(torch.empty(2*streams+streams*streams))
        self.scales = nn.Parameter(torch.empty(3))
        self.use_tf32, self.fused_grad_acc = use_tf32, fused_grad_acc
        self.sinkhorn = sinkhorn
        if streams != 4 or iterations != 20:
            raise ValueError('The NVIDIA composite API requires four streams and 20 iterations')

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1/math.sqrt(self.streams*self.width))
        n = self.streams
        self.bias[:n].fill_(-math.log(n-1))
        self.bias[n:2*n].zero_()
        self.bias[2*n:].view(n,n).copy_(torch.eye(n, device=self.phi.device))
        self.scales.fill_(.01)
        self.phi.copy_(interleave_phi(self.phi, self.streams))

    def read(self, x):
        check_precision(self.use_tf32)
        x = x.contiguous()
        # The documented backward order is write -> aggregate -> projection.
        # write initializes this per-invocation buffer; the latter two add to it.
        # No module-global buffer is shared across invocations/checkpoints.
        buffer = (torch.empty_like(x, dtype=torch.float32)
                  if self.fused_grad_acc and torch.is_grad_enabled() and x.requires_grad else None)
        ops = nvidia_ops()
        if self.sinkhorn:
            branch, post, comb = ops.mhc_generate_mix_and_aggregate(
                x, self.phi, self.scales, self.bias.unsqueeze(0),
                norm_weight=None, use_tf32=self.use_tf32,
                fused_grad_x_acc_buffer=buffer, use_split_k=False)
        else:
            # Exactly TE's composite above, omitting its log-space Sinkhorn.
            # Raw signed coefficients are used directly: no exp, softmax,
            # clipping, row normalization, or replacement PyTorch kernels.
            b, t, d, n = x.shape
            raw, ms = ops.mhc_fused_projection(x.view(b*t, d*n), self.phi,
                norm_weight=None, use_tf32=self.use_tf32,
                fused_grad_x_acc_buffer=buffer, use_split_k=False)
            pre, post, comb = ops.mhc_fused_scale(raw, self.scales, self.bias.unsqueeze(0), ms, n)
            branch = ops.mhc_fused_aggregate(x, pre.view(b, t, n), n,
                use_tf32=self.use_tf32, fused_grad_x_acc_buffer=buffer)
            post, comb = post.view(b, t, n), comb.view(b, t, n, n)
        return branch, x, (post, comb, buffer)

    def write(self, output, residual, mixes):
        post, comb, buffer = mixes
        return nvidia_ops().mhc_fused_expand_combine(
            output.contiguous(), None, post, residual, comb, self.streams,
            use_tf32=self.use_tf32, fused_grad_x_acc_buffer=buffer)

    def routing(self, x):
        """Generate original coefficients without merging the input streams.

        For post-attention aggregation the activation paths no longer follow
        TE's fused gradient-buffer ordering. Use its ordinary custom backwards.
        Existing read()/write() calls are unchanged.
        """
        if self.fused_grad_acc:
            raise ValueError('Post-attention merge does not support fused gradient accumulation')
        check_precision(self.use_tf32)
        x = x.contiguous()
        b, t, d, n = x.shape
        ops = nvidia_ops()
        raw, ms = ops.mhc_fused_projection(x.view(b*t, d*n), self.phi,
            norm_weight=None, use_tf32=self.use_tf32,
            fused_grad_x_acc_buffer=None, use_split_k=False)
        pre, post, comb = ops.mhc_fused_scale(raw, self.scales, self.bias.unsqueeze(0), ms, n)
        comb = comb.view(b, t, n, n)
        if self.sinkhorn:
            comb = ops.mhc_fused_sinkhorn(comb, n, iters=self.iterations)
        return pre.view(b, t, n), x, (post.view(b, t, n), comb, None)


def diagonal_post_write(mixed, output, post):
    """One attention output per stream; no sum over the stream axis."""
    return (mixed.float() + output.float()*post.unsqueeze(-2)).to(mixed.dtype)


class NvidiaStreamConnection(NvidiaHyperConnection):
    """NVIDIA residual mixing with diagonal, independent branch writeback.

    Construct from an initialized ordinary connector to preserve all shared
    RNG draws. No attention H_pre parameters survive: aggregation happens at
    the following, unchanged MLP connector. TE still supplies projection,
    scale, Sinkhorn and residual-combine kernels and their custom backwards.
    """
    def __init__(self, shared):
        nn.Module.__init__(self)
        self.width, self.streams, self.iterations = shared.width, shared.streams, shared.iterations
        self.use_tf32, self.fused_grad_acc = shared.use_tf32, False
        self.sinkhorn = shared.sinkhorn
        n = self.streams
        self.phi = nn.Parameter(shared.phi[n:].detach().clone())
        self.bias = nn.Parameter(shared.bias[n:].detach().clone())
        self.scales = nn.Parameter(shared.scales[1:].detach().clone())
        self.compiled_diagonal = None

    def routing(self, x):
        check_precision(self.use_tf32)
        x = x.contiguous()
        b, t, d, n = x.shape
        ops = nvidia_ops()
        # Pad the unused pre slots for the unmodified NVIDIA 24-output API.
        phi = F.pad(self.phi, (0, 0, n, 0))
        bias = F.pad(self.bias, (n, 0)).unsqueeze(0)
        scales = F.pad(self.scales, (1, 0))
        raw, ms = ops.mhc_fused_projection(x.view(b*t, d*n), phi,
            norm_weight=None, use_tf32=self.use_tf32,
            fused_grad_x_acc_buffer=None, use_split_k=False)
        _, post, comb = ops.mhc_fused_scale(raw, scales, bias, ms, n)
        comb = comb.view(b, t, n, n)
        if self.sinkhorn:
            comb = ops.mhc_fused_sinkhorn(comb, n, iters=self.iterations)
        return x, (post.view(b, t, n), comb, None)

    def write(self, output, residual, mixes):
        if output.shape != residual.shape:
            raise ValueError('Independent attention writeback requires one D-vector per stream')
        post, _, _ = mixes
        # TE's branch input is one shared vector. Use it only for H_res X,
        # then add post_s * attention_s with a compiled pointwise operation.
        mixed = super().write(torch.zeros_like(output[..., 0]), residual, mixes)
        return (self.compiled_diagonal or diagonal_post_write)(mixed, output, post)


class NvidiaHyperHead(nn.Module):
    def __init__(self, width, streams, *, use_tf32=True):
        super().__init__()
        self.width, self.streams = width, streams
        self.phi = nn.Parameter(torch.empty(streams, streams*width))
        self.scale = nn.Parameter(torch.empty(1))
        self.bias = nn.Parameter(torch.empty(streams))
        self.use_tf32 = use_tf32

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1/math.sqrt(self.streams*self.width))
        self.scale.fill_(.01)
        self.bias.fill_(-math.log(self.streams-1))
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
