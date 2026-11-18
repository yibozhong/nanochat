"""Diagnostic fused BF16 read for the additive lookahead residual.

Use the FP32 per-CTA partials and final torch sum of nanochat/attnres_triton.py,
and the explicit contiguous upstream gradient used by the FLA attention wrapper
in nanochat/fla_attention.py. Each CTA walks a contiguous tile of hidden values.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _read(H, V, M, Q, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    h = tl.load(H + i, i < N, 0).to(tl.float32)
    v = tl.load(V + i, i < N, 0)
    q = h + tl.load(M) * v
    tl.store(Q + i, q.to(tl.bfloat16, fp_downcast_rounding="rtne"), i < N)


@triton.jit
def _backward(DQ, V, M, DV, PARTIAL, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    dq = tl.load(DQ + i, i < N, 0).to(tl.float32)
    v = tl.load(V + i, i < N, 0)
    tl.store(DV + i, dq * tl.load(M), i < N)
    tl.store(PARTIAL + pid, tl.sum(dq * v, axis=0))


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, v, mu):
        n = h.numel()
        q = torch.empty_like(h)
        grid = (triton.cdiv(n, 512),)
        _read[grid](h, v, mu, q, n, 512, enable_fp_fusion=False)
        ctx.save_for_backward(v, mu)
        return q

    @staticmethod
    def backward(ctx, grad):
        v, mu = ctx.saved_tensors
        grad = grad.contiguous()
        dv = torch.empty_like(v)
        blocks = triton.cdiv(v.numel(), 512)
        sums = v.new_empty(blocks)
        _backward[(blocks,)](grad, v, mu, dv, sums, v.numel(), 512, enable_fp_fusion=False)
        return grad, dv, sums.sum().reshape_as(mu)


def fused_read(h, v, mu):
    if (h.is_cuda and h.is_contiguous() and v.is_contiguous() and h.shape == v.shape
            and h.numel() and h.dtype == torch.bfloat16 and v.dtype == mu.dtype == torch.float32
            and mu.numel() == 1):
        return _Read.apply(h, v, mu)
    return (h.float() + mu * v).to(h.dtype)
