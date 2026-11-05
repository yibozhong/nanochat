"""Shared 4->16->4 ReLU stream mixer, fused with mHC branch distribution.

Each [token, channel] row is independent. Activations are BF16; the tiny
matrix arithmetic and parameter reductions use FP32 like existing routing.
Backward recomputes the 16 activations instead of storing their full bank.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

from nanochat.channel_mixing import _backward_branch


@tr.jit
def _hidden(X, Win, row, N: tl.constexpr):
    s = tl.arange(0, 4)
    h = tl.arange(0, 16)
    x = tl.load(X + row[:, None]*4+s[None, :], row[:, None]<N, 0).to(tl.float32)
    z = tl.full((row.shape[0], 16), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(Win+h*4+k)
        xk = tl.sum(tl.where(s[None, :]==k, x, 0.), 1)
        z += xk[:, None]*w[None, :]
    return x, z, tl.maximum(z, 0.)


@tr.jit
def _forward(X, Y, Post, Win, Wout, Out, N: tl.constexpr, C: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    h = tl.arange(0, 16)
    x, z, a = _hidden(X, Win, row, N)
    mixed = tl.full((BLOCK, 4), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(Wout+k*16+h)
        value = tl.sum(a*w[None, :], 1)
        mixed += tl.where(s[None, :]==k, value[:, None], 0.)
    y = tl.load(Y+row, row<N, 0).to(tl.float32)
    post = tl.load(Post+(row[:, None]//C)*4+s[None, :], row[:, None]<N, 0)
    out = mixed+y[:, None]*post
    tl.store(Out+row[:, None]*4+s[None, :], out, row[:, None]<N)


@tr.jit
def _backward(X, Win, Wout, GO, GX, Parts, N: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    row = tile*BLOCK+tl.arange(0, BLOCK)
    s = tl.arange(0, 4)
    h = tl.arange(0, 16)
    x, z, a = _hidden(X, Win, row, N)
    go = tl.load(GO+row[:, None]*4+s[None, :], row[:, None]<N, 0).to(tl.float32)
    da = tl.full((BLOCK, 16), 0., tl.float32)
    for k in tl.static_range(4):
        gok = tl.sum(tl.where(s[None, :]==k, go, 0.), 1)
        w = tl.load(Wout+k*16+h)
        da += gok[:, None]*w[None, :]
        dw = tl.sum(gok[:, None]*a, 0)
        tl.store(Parts+tile*128+64+k*16+h, dw)
    dz = tl.where(z>0., da, 0.)
    dx = tl.full((BLOCK, 4), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(Win+h*4+k)
        dxk = tl.sum(dz*w[None, :], 1)
        dx += tl.where(s[None, :]==k, dxk[:, None], 0.)
        xk = tl.sum(tl.where(s[None, :]==k, x, 0.), 1)
        dw = tl.sum(dz*xk[:, None], 0)
        tl.store(Parts+tile*128+h*4+k, dw)
    tl.store(GX+row[:, None]*4+s[None, :], dx, row[:, None]<N)


class _StreamMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, post, win, wout):
        if x.dtype != torch.bfloat16 or y.dtype != x.dtype:
            raise ValueError('Stream MLP requires BF16 activations')
        if not all(t.is_contiguous() for t in (x, y, post, win, wout)):
            raise ValueError('Stream MLP requires contiguous tensors')
        if any(t.dtype != torch.float32 for t in (post, win, wout)):
            raise ValueError('Routing and tiny mixer matrices require FP32')
        if x.shape[-1] != 4 or y.shape != x.shape[:-1] or post.shape != (*x.shape[:-2], 4):
            raise ValueError('Invalid stream dimensions')
        if win.shape != (16, 4) or wout.shape != (4, 16):
            raise ValueError('Initial implementation uses 4->16->4')
        n = x.numel()//4
        out = torch.empty_like(x)
        _forward[(tr.cdiv(n, 128),)](x, y, post, win, wout, out, n, x.shape[-2], 128, num_warps=4)
        ctx.save_for_backward(x, y, post, win, wout)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        x, y, post, win, wout = ctx.saved_tensors
        go = go.contiguous()
        n = x.numel()//4
        c = x.shape[-2]
        gx = torch.empty_like(x)
        gy = torch.empty_like(y)
        gp = torch.empty_like(post)
        parts = torch.empty((tr.cdiv(n, 256), 128), device=x.device, dtype=torch.float32)
        _backward[(tr.cdiv(n, 256),)](x, win, wout, go, gx, parts, n, 256, num_warps=8)
        _backward_branch[(n//c,)](y, post, go, gy, gp, n//c, c, tr.next_power_of_2(c), num_warps=4)
        reduced = parts.sum(0)
        return gx, gy, gp, reduced[:64].reshape_as(win).contiguous(), reduced[64:].reshape_as(wout).contiguous()


def stream_mlp(x, y, post, win, wout):
    return _StreamMLP.apply(x, y, post, win, wout)
