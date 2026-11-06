"""Fused1x4/4x5 state attention, with recomputed analytic backward.

References: FLA parallel attention's stable softmax/softmax VJP and NVIDIA
TE2.20.2 mHC's small-axis expansion/FP32 accumulation (see kernel-sources.json).
Only the 4/5-key bank is kept on chip. All queries of a token/head belong to
one program, so dK/dV require neither atomics nor padded reduction workspaces.
QKV/O projections remain ordinary GEMMs outside this operator.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _probability(q, k, key, NK: tl.constexpr, SCALE: tl.constexpr):
    logits = tl.sum(q[:, None, :] * k, 2) * SCALE * 1.4426950408889634
    logits = tl.where(key[None, :] < NK, logits, float('-inf'))
    p = tl.exp2(logits - tl.max(logits, 1)[:, None])
    return p / tl.sum(p, 1)[:, None]


@tr.jit
def _forward(Q, K, V, O, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             NQ: tl.constexpr, NK: tl.constexpr, BG: tl.constexpr,
             BD: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    group = tl.program_id(0)*BG + tl.arange(0, BG)
    token, head = group // H, group % H
    key, channel = tl.arange(0, BK), tl.arange(0, BD)
    koff = ((token[:, None, None]*NK+key[None, :, None])*H
            +head[:, None, None])*D+channel[None, None, :]
    mask = (token[:, None, None]<N)&(key[None, :, None]<NK)&(channel[None, None, :]<D)
    k = tl.load(K+koff, mask, 0).to(tl.float32)
    v = tl.load(V+koff, mask, 0).to(tl.float32)
    for i in tl.static_range(NQ):
        qoff = ((token[:, None]*NQ+i)*H+head[:, None])*D+channel[None, :]
        qmask = (token[:, None]<N)&(channel[None, :]<D)
        q = tl.load(Q+qoff, qmask, 0).to(tl.float32)
        p = _probability(q, k, key, NK, SCALE)
        out = tl.sum(p[:, :, None]*v, 1)
        tl.store(O+qoff, out, qmask)


@tr.jit
def _backward(Q, K, V, GO, GQ, GK, GV, N: tl.constexpr, H: tl.constexpr,
              D: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr,
              BG: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr,
              SCALE: tl.constexpr):
    group = tl.program_id(0)*BG + tl.arange(0, BG)
    token, head = group // H, group % H
    key, channel = tl.arange(0, BK), tl.arange(0, BD)
    koff = ((token[:, None, None]*NK+key[None, :, None])*H
            +head[:, None, None])*D+channel[None, None, :]
    mask = (token[:, None, None]<N)&(key[None, :, None]<NK)&(channel[None, None, :]<D)
    k = tl.load(K+koff, mask, 0).to(tl.float32)
    v = tl.load(V+koff, mask, 0).to(tl.float32)
    dk = tl.zeros((BG, BK, BD), tl.float32)
    dv = tl.zeros((BG, BK, BD), tl.float32)
    for i in tl.static_range(NQ):
        qoff = ((token[:, None]*NQ+i)*H+head[:, None])*D+channel[None, :]
        qmask = (token[:, None]<N)&(channel[None, :]<D)
        q = tl.load(Q+qoff, qmask, 0).to(tl.float32)
        go = tl.load(GO+qoff, qmask, 0).to(tl.float32)
        p = _probability(q, k, key, NK, SCALE)
        dp = tl.sum(go[:, None, :]*v, 2)
        ds = p*(dp-tl.sum(p*dp, 1)[:, None])*SCALE
        dq = tl.sum(ds[:, :, None]*k, 1)
        dk = tl.fma(ds[:, :, None], q[:, None, :], dk)
        dv = tl.fma(p[:, :, None], go[:, None, :], dv)
        tl.store(GQ+qoff, dq, qmask)
    tl.store(GK+koff, dk, mask)
    tl.store(GV+koff, dv, mask)


class _StateAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        n, nq, heads, dim = q.shape
        nk = k.shape[1]
        out = torch.empty_like(q)
        args = (n, heads, dim, nq, nk, 4, tr.next_power_of_2(dim),
                tr.next_power_of_2(nk), dim**-.5)
        _forward[(tr.cdiv(n*heads, 4),)](q,k,v,out,*args,num_warps=4)
        ctx.save_for_backward(q,k,v)
        ctx.args = args
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        q,k,v = ctx.saved_tensors
        go = go.contiguous()
        gq,gk,gv = torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
        n,heads = ctx.args[:2]
        _backward[(tr.cdiv(n*heads, 4),)](q,k,v,go,gq,gk,gv,*ctx.args,num_warps=4)
        return gq,gk,gv


def state_attention(q, k, v):
    """[tokens,slots,heads,head_dim]; no cross-token mixing or slot mask."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError('State attention expects compatible four-dimensional Q/K/V')
    if (q.shape[1],k.shape[1]) not in ((1,4),(4,5)):
        raise ValueError('State attention supports read1x4 and write4x5 only')
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:] or min(q.shape) < 1:
        raise ValueError('State attention batch/head dimensions do not match')
    if q.shape[-1] > 256:
        raise ValueError('State attention supports head dimensions up to256')
    if q.dtype not in (torch.bfloat16,torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError('State attention expects BF16 or FP32 inputs of the same dtype')
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError('State attention expects one CUDA device')
    return _StateAttention.apply(q.contiguous(), k.contiguous(), v.contiguous())
