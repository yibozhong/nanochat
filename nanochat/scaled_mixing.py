"""Static per-channel/input-stream scaling fused with shared HC write-back.

out[t,c,j] = sum_i x[t,c,i] * a[c,i] * H[t,i,j] + y[t,c] * post[t,j].
BF16 states, FP32 coefficients/arithmetic and deterministic parameter reductions.
No scaled activation or per-channel matrix bank is saved or materialized.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _scaled_forward(X, Y, Post, H, A, Out, N: tl.constexpr, C: tl.constexpr,
                    BT: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)*BT + tl.arange(0, BT)
    c = tl.program_id(1)*BC + tl.arange(0, BC)
    r = tl.arange(0, 4)
    mask = (t[:, None, None] < N) & (c[None, :, None] < C)
    off = (t[:, None, None]*C+c[None, :, None])*4+r[None, None, :]
    x = tl.load(X+off, mask, 0).to(tl.float32)
    a = tl.load(A+c[:, None]*4+r[None, :], c[:, None] < C, 0)
    scaled = x*a[None, :, :]
    out = tl.full((BT, BC, 4), 0., tl.float32)
    for i in tl.static_range(4):
        xi = tl.sum(tl.where(r[None, None, :] == i, scaled, 0.), 2)
        h = tl.load(H+t[:, None]*16+i*4+r[None, :], t[:, None] < N, 0)
        out += xi[:, :, None]*h[:, None, :]
    y = tl.load(Y+t[:, None]*C+c[None, :], (t[:, None] < N)&(c[None, :] < C), 0).to(tl.float32)
    post = tl.load(Post+t[:, None]*4+r[None, :], t[:, None] < N, 0)
    out += y[:, :, None]*post[:, None, :]
    tl.store(Out+off, out, mask)


@tr.jit
def _scaled_dx_da(X, H, A, GO, GX, Parts, N: tl.constexpr, C: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr):
    split = tl.program_id(0)
    t = split*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    r = tl.arange(0, 4)
    mask = (t[:, None, None] < N)&(c[None, :, None] < C)
    off = (t[:, None, None]*C+c[None, :, None])*4+r[None, None, :]
    x = tl.load(X+off, mask, 0).to(tl.float32)
    go = tl.load(GO+off, mask, 0).to(tl.float32)
    du = tl.full((BT, BC, 4), 0., tl.float32)
    for i in tl.static_range(4):
        h = tl.load(H+t[:, None]*16+i*4+r[None, :], t[:, None] < N, 0)
        value = tl.sum(go*h[:, None, :], 2)
        du += tl.where(r[None, None, :] == i, value[:, :, None], 0.)
    a = tl.load(A+c[:, None]*4+r[None, :], c[:, None] < C, 0)
    tl.store(GX+off, du*a[None, :, :], mask)
    tl.store(Parts+(split*C+c[:, None])*4+r[None, :], tl.sum(du*x, 0), c[:, None] < C)


@tr.jit
def _scaled_routes(X, Y, Post, A, GO, GY, GPost, GH, C: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, BC)
    r = tl.arange(0, 4)
    off = (t*C+c[:, None])*4+r[None, :]
    x = tl.load(X+off, c[:, None] < C, 0).to(tl.float32)
    a = tl.load(A+c[:, None]*4+r[None, :], c[:, None] < C, 0)
    scaled = x*a
    go = tl.load(GO+off, c[:, None] < C, 0).to(tl.float32)
    for i in tl.static_range(4):
        xi = tl.sum(tl.where(r[None, :] == i, scaled, 0.), 1)
        tl.store(GH+t*16+i*4+r, tl.sum(xi[:, None]*go, 0))
    y = tl.load(Y+t*C+c, c < C, 0).to(tl.float32)
    post = tl.load(Post+t*4+r)
    tl.store(GY+t*C+c, tl.sum(go*post[None, :], 1), c < C)
    tl.store(GPost+t*4+r, tl.sum(go*y[:, None], 0))


@tr.jit
def _scaled_da_reduce(Parts, GA, K: tl.constexpr, S: tl.constexpr, BS: tl.constexpr):
    k = tl.program_id(0)*16+tl.arange(0, 16)
    s = tl.arange(0, BS)
    p = tl.load(Parts+s[:, None]*K+k[None, :], (s[:, None] < S)&(k[None, :] < K), 0)
    tl.store(GA+k, tl.sum(p, 0), k < K)


class _ScaledMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, post, h, scale):
        if x.ndim != 4 or x.shape[-1] != 4 or not x.is_cuda:
            raise ValueError('Scaled HC expects CUDA [batch,time,channel,4]')
        c = x.shape[-2]; n = x.numel()//(4*c)
        if n < 1 or x.dtype != torch.bfloat16 or y.dtype != x.dtype:
            raise ValueError('Scaled HC uses nonempty BF16 states')
        if y.shape != x.shape[:-1] or post.shape != (*x.shape[:2],4) or h.shape != (*x.shape[:2],4,4) or scale.shape != (c,4):
            raise ValueError('Invalid branch/routing/scale shapes')
        if any(p.dtype != torch.float32 for p in (post,h,scale)):
            raise ValueError('Routing and scale must be FP32')
        if not all(p.is_contiguous() and p.device == x.device for p in (x,y,post,h,scale)):
            raise ValueError('Scaled HC requires contiguous colocated inputs')
        out = torch.empty_like(x)
        _scaled_forward[(tr.cdiv(n,8),tr.cdiv(c,128))](x,y,post,h,scale,out,n,c,8,128,num_warps=4)
        ctx.save_for_backward(x,y,post,h,scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        x,y,post,h,scale = ctx.saved_tensors
        go=go.contiguous(); c=x.shape[-2]; n=x.numel()//(4*c)
        gx=torch.empty_like(x);gy=torch.empty_like(y)
        gp=torch.empty_like(post);gh=torch.empty_like(h);ga=torch.empty_like(scale)
        splits=tr.cdiv(n,64)
        parts=torch.empty((splits,c,4),device=x.device,dtype=torch.float32)
        _scaled_dx_da[(splits,tr.cdiv(c,32))](x,h,scale,go,gx,parts,n,c,64,32,num_warps=8)
        _scaled_routes[(n,)](x,y,post,scale,go,gy,gp,gh,c,tr.next_power_of_2(c),num_warps=4)
        _scaled_da_reduce[(tr.cdiv(c*4,16),)](parts,ga,c*4,splits,tr.next_power_of_2(splits),num_warps=4)
        return gx,gy,gp,gh,ga


def scaled_mix(x, y, post, h, scale):
    return _ScaledMix.apply(x,y,post,h,scale)
