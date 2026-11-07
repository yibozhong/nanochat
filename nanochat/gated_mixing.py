"""Input-side SiLU gates fused with shared HC mixing and branch write-back.

No gated-state or gate bank is materialized. BF16 states, FP32 arithmetic,
analytic first-order backward, deterministic parameter reductions. The local
generator sees exactly the four streams of its own token/channel.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable

# Unique positive solution of b * sigmoid(b) = 1. An initialization only:
# the forward always applies ordinary, unshifted, unscaled SiLU.
SILU_IDENTITY_BIAS = 1.2784645427610737


@tr.jit
def _gate(X, W, Bias, t, c, N: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, 4)
    mask = (t[:, None, None] < N) & (c[None, :, None] < C)
    x = tl.load(X+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :], mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x*x, 2)*0.25+1.1920928955078125e-7)
    v = x*inv[:, :, None]
    z = tl.full((t.shape[0], c.shape[0], 4), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(W+c[:, None]*16+r[None, :]*4+k, c[:, None] < C, 0)
        vk = tl.sum(tl.where(r[None, None, :] == k, v, 0.), 2)
        z += vk[:, :, None]*w[None, :, :]
    z += tl.load(Bias+c[:, None]*4+r[None, :], c[:, None] < C, 0)[None, :, :]
    sigmoid = tl.sigmoid(z)
    return x, v, inv, z, sigmoid, z*sigmoid


@tr.jit
def _forward(X, Y, Post, H, W, Bias, Out, N: tl.constexpr, C: tl.constexpr,
             BT: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    r = tl.arange(0, 4)
    x, v, inv, z, sigmoid, gate = _gate(X, W, Bias, t, c, N, C)
    gated = x*gate
    out = tl.full((BT, BC, 4), 0., tl.float32)
    for i in tl.static_range(4):
        xi = tl.sum(tl.where(r[None, None, :] == i, gated, 0.), 2)
        h = tl.load(H+t[:, None]*16+i*4+r[None, :], t[:, None] < N, 0)
        out += xi[:, :, None]*h[:, None, :]
    y = tl.load(Y+t[:, None]*C+c[None, :], (t[:, None] < N)&(c[None, :] < C), 0).to(tl.float32)
    post = tl.load(Post+t[:, None]*4+r[None, :], t[:, None] < N, 0)
    out += y[:, :, None]*post[:, None, :]
    tl.store(Out+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :], out,
             (t[:, None, None] < N)&(c[None, :, None] < C))


@tr.jit
def _local_backward(X, H, W, Bias, GO, GX, Parts, N: tl.constexpr, C: tl.constexpr,
                    BT: tl.constexpr, BC: tl.constexpr):
    split = tl.program_id(0)
    t = split*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    r = tl.arange(0, 4)
    x, v, inv, z, sigmoid, gate = _gate(X, W, Bias, t, c, N, C)
    go = tl.load(GO+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :],
                 (t[:, None, None] < N)&(c[None, :, None] < C), 0).to(tl.float32)
    du = tl.full((BT, BC, 4), 0., tl.float32)
    for i in tl.static_range(4):
        h = tl.load(H+t[:, None]*16+i*4+r[None, :], t[:, None] < N, 0)
        dui = tl.sum(go*h[:, None, :], 2)
        du += tl.where(r[None, None, :] == i, dui[:, :, None], 0.)
    dz = du*x*(sigmoid+z*sigmoid*(1.-sigmoid))
    dv = tl.full((BT, BC, 4), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(W+c[:, None]*16+r[None, :]*4+k, c[:, None] < C, 0)
        dvk = tl.sum(dz*w[None, :, :], 2)
        dv += tl.where(r[None, None, :] == k, dvk[:, :, None], 0.)
        vk = tl.sum(tl.where(r[None, None, :] == k, v, 0.), 2)
        dw = tl.sum(dz*vk[:, :, None], 0)
        tl.store(Parts+(split*C+c[:, None])*20+r[None, :]*4+k, dw, c[:, None] < C)
    dx = du*gate+inv[:, :, None]*(dv-v*(tl.sum(dv*v, 2)*.25)[:, :, None])
    tl.store(GX+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :], dx,
             (t[:, None, None] < N)&(c[None, :, None] < C))
    tl.store(Parts+(split*C+c[:, None])*20+16+r[None, :], tl.sum(dz, 0), c[:, None] < C)


@tr.jit
def _routes_backward(X, Y, Post, W, Bias, GO, GY, GPost, GH,
                     N: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    # One token owns its entire routing gradient: no atomics or N*C*16 bank.
    t = tl.program_id(0)+tl.arange(0, 1)
    c = tl.arange(0, BC)
    r = tl.arange(0, 4)
    x, v, inv, z, sigmoid, gate = _gate(X, W, Bias, t, c, N, C)
    go = tl.load(GO+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :],
                 c[None, :, None] < C, 0).to(tl.float32)
    gated = x*gate
    for i in tl.static_range(4):
        xi = tl.sum(tl.where(r[None, None, :] == i, gated, 0.), 2)
        tl.store(GH+t[:, None]*16+i*4+r[None, :], tl.sum(xi[:, :, None]*go, 1))
    y = tl.load(Y+t[:, None]*C+c[None, :], c[None, :] < C, 0).to(tl.float32)
    post = tl.load(Post+t[:, None]*4+r[None, :])
    tl.store(GY+t[:, None]*C+c[None, :], tl.sum(go*post[:, None, :], 2), c[None, :] < C)
    tl.store(GPost+t[:, None]*4+r[None, :], tl.sum(go*y[:, :, None], 1))


class _GatedMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, post, h, weight, bias):
        if x.ndim != 4 or x.shape[-1] != 4 or not x.is_cuda:
            raise ValueError('Gated HC expects CUDA [batch,time,channel,4]')
        c = x.shape[-2]; n = x.numel()//(4*c)
        if x.dtype != torch.bfloat16 or y.dtype != x.dtype:
            raise ValueError('Gated HC uses BF16 states')
        if y.shape != x.shape[:-1] or post.shape != (*x.shape[:2], 4) or h.shape != (*x.shape[:2], 4, 4):
            raise ValueError('Invalid branch/routing shapes')
        if weight.shape != (c, 4, 4) or bias.shape != (c, 4):
            raise ValueError('Invalid local gate parameter shapes')
        if any(p.dtype != torch.float32 for p in (post, h, weight, bias)):
            raise ValueError('Gate and routing parameters must be FP32')
        if not all(p.is_contiguous() and p.device == x.device for p in (x,y,post,h,weight,bias)):
            raise ValueError('Gated HC requires contiguous colocated inputs')
        out = torch.empty_like(x)
        _forward[(tr.cdiv(n,32),tr.cdiv(c,8))](x,y,post,h,weight,bias,out,n,c,32,8,num_warps=8)
        ctx.save_for_backward(x,y,post,h,weight,bias)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        x,y,post,h,weight,bias = ctx.saved_tensors
        go = go.contiguous(); c=x.shape[-2]; n=x.numel()//(4*c)
        gx=torch.empty_like(x); gy=torch.empty_like(y)
        gp=torch.empty_like(post); gh=torch.empty_like(h)
        # 10 MiB at B16/T2048/D512, versus channelmix's40.5 MiB.
        parts=torch.empty((tr.cdiv(n,128),c,20),device=x.device,dtype=torch.float32)
        _local_backward[(tr.cdiv(n,128),tr.cdiv(c,4))](x,h,weight,bias,go,gx,parts,n,c,128,4,num_warps=8)
        _routes_backward[(n,)](x,y,post,weight,bias,go,gy,gp,gh,n,c,tr.next_power_of_2(c),num_warps=8)
        reduced=parts.sum(0)
        return gx,gy,gp,gh,reduced[:,:16].reshape_as(weight).contiguous(),reduced[:,16:].contiguous()


def gated_mix(x, y, post, h, weight, bias):
    return _GatedMix.apply(x,y,post,h,weight,bias)
