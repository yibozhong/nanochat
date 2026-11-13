"""Fused residual + main-RMS + narrow readout, with fused exact VJP.

Multiple token rows per CTA; independent power-of-two main/branch tiles avoid
padding D+R=576 to 1024. No atomics, no FP32 activation-sized intermediates,
no separate concatenate or gradient-split launches. GEMMs use packed weights.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _forward(X, P, Y, Inv, M: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
             BM: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.arange(0, BD)
    extra = tl.arange(0, BR)
    x = tl.load(X+rows[:, None]*D+cols[None, :], (rows[:, None]<M)&(cols[None, :]<D), 0).to(tl.float32)
    p = tl.load(P+rows[:, None]*(D+R)+cols[None, :], (rows[:, None]<M)&(cols[None, :]<D), 0).to(tl.float32)
    main = (x+p).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.rsqrt(tl.sum(main*main, 1)/D + 1.1920928955078125e-7)
    b = tl.load(P+rows[:, None]*(D+R)+D+extra[None, :], (rows[:, None]<M)&(extra[None, :]<R), 0).to(tl.float32)
    tl.store(Y+rows[:, None]*(D+R)+cols[None, :], main*inv[:, None], (rows[:, None]<M)&(cols[None, :]<D))
    tl.store(Y+rows[:, None]*(D+R)+D+extra[None, :], b*inv[:, None], (rows[:, None]<M)&(extra[None, :]<R))
    tl.store(Inv+rows, inv, rows<M)


@tr.jit
def _backward(X, P, G, Inv, DX, DP, M: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
              BM: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0)*BM + tl.arange(0, BM)
    cols = tl.arange(0, BD)
    extra = tl.arange(0, BR)
    mask = (rows[:, None]<M)&(cols[None, :]<D)
    bmask = (rows[:, None]<M)&(extra[None, :]<R)
    x = tl.load(X+rows[:, None]*D+cols[None, :], mask, 0).to(tl.float32)
    p = tl.load(P+rows[:, None]*(D+R)+cols[None, :], mask, 0).to(tl.float32)
    main = (x+p).to(X.dtype.element_ty).to(tl.float32)
    b = tl.load(P+rows[:, None]*(D+R)+D+extra[None, :], bmask, 0).to(tl.float32)
    gh = tl.load(G+rows[:, None]*(D+R)+cols[None, :], mask, 0).to(tl.float32)
    gb = tl.load(G+rows[:, None]*(D+R)+D+extra[None, :], bmask, 0).to(tl.float32)
    inv = tl.load(Inv+rows, rows<M, 0)
    dot = tl.sum(gh*main, 1)+tl.sum(gb*b, 1)
    correction = dot*(inv*inv*inv/D)
    dx = gh*inv[:, None]-main*correction[:, None]
    tl.store(DX+rows[:, None]*D+cols[None, :], dx, mask)
    tl.store(DP+rows[:, None]*(D+R)+cols[None, :], dx, mask)
    tl.store(DP+rows[:, None]*(D+R)+D+extra[None, :], gb*inv[:, None], bmask)


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        d, r = x.shape[-1], p.shape[-1]-x.shape[-1]
        m = x.numel()//d
        y = torch.empty_like(p)
        inv = torch.empty(m, dtype=torch.float32, device=x.device)
        _forward[(tr.cdiv(m, 4),)](x,p,y,inv,m,d,r,4,tr.next_power_of_2(d),tr.next_power_of_2(r),num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,p,inv)
        return y

    @staticmethod
    def backward(ctx, g):
        x,p,inv = ctx.saved_tensors
        d, r = x.shape[-1], p.shape[-1]-x.shape[-1]
        m = x.numel()//d
        dx, dp = torch.empty_like(x), torch.empty_like(p)
        _backward[(tr.cdiv(m, 4),)](x,p,g.contiguous(),inv,dx,dp,m,d,r,4,tr.next_power_of_2(d),tr.next_power_of_2(r),num_warps=4,enable_fp_fusion=False)
        return dx, dp


def fused_readout(residual, projected):
    if not residual.is_cuda or residual.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('Triton readout requires CUDA BF16/FP32')
    if projected.dtype != residual.dtype or projected.device != residual.device:
        raise ValueError('Readout operands must share dtype and device')
    if projected.shape[:-1] != residual.shape[:-1] or projected.shape[-1] <= residual.shape[-1]:
        raise ValueError('Expected packed main and branch projection')
    return _Readout.apply(residual.contiguous(), projected.contiguous())
