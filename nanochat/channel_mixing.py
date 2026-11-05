"""Local channel-wise signed mixing, fused with mHC branch distribution.

Layout follows NVIDIA TE: x[..., channel, input_stream], M[..., input, output].
Each channel has its own 4 -> 16 generator. Only its four stream values enter
the generator, with parameter-free RMS normalization (FP32 epsilon). No
Sinkhorn/softmax/clipping is applied. Coefficients never leave GPU registers.
Backward recomputes them and uses deterministic partial parameter reductions.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _coeff(X, W, Bias, A, t, c, N: tl.constexpr, C: tl.constexpr):
    r = tl.arange(0, 4)
    x = tl.load(X + (t[:, None, None]*C+c[None, :, None])*4+r[None, None, :],
                (t[:, None, None]<N)&(c[None, :, None]<C), 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x*x, 2)*0.25 + 1.1920928955078125e-7)
    v = x*inv[:, :, None]
    i = r[:, None]
    j = r[None, :]
    raw = tl.full((t.shape[0], c.shape[0], 4, 4), 0., tl.float32)
    for k in tl.static_range(4):
        w = tl.load(W + c[:, None, None]*64 + i[None, :, :]*16 + j[None, :, :]*4+k,
                    c[:, None, None]<C, 0)
        vk = tl.sum(tl.where(r[None, None, :]==k, v, 0.), 2)
        raw += vk[:, :, None, None]*w[None, :, :, :]
    b = tl.load(Bias+c[:, None, None]*16+i[None, :, :]*4+j[None, :, :], c[:, None, None]<C, 0)
    a = tl.load(A)
    return x, v, inv, raw, b[None, :, :, :]+a*raw


@tr.jit
def _forward(X, Y, Post, W, Bias, A, Out, N: tl.constexpr, C: tl.constexpr,
             BT: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    r = tl.arange(0, 4)
    x, v, inv, raw, m = _coeff(X, W, Bias, A, t, c, N, C)
    y = tl.load(Y+t[:, None]*C+c[None, :], (t[:, None]<N)&(c[None, :]<C), 0).to(tl.float32)
    post = tl.load(Post+t[:, None]*4+r[None, :], t[:, None]<N, 0)
    out = tl.sum(x[:, :, :, None]*m, 2)+y[:, :, None]*post[:, None, :]
    tl.store(Out+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :], out,
             (t[:, None, None]<N)&(c[None, :, None]<C))


@tr.jit
def _backward_mix(X, W, Bias, A, GO, GX, Parts, N: tl.constexpr, C: tl.constexpr,
                  BT: tl.constexpr, BC: tl.constexpr):
    split = tl.program_id(0)
    t = split*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    r = tl.arange(0, 4)
    i, j = r[:, None], r[None, :]
    x, v, inv, raw, m = _coeff(X, W, Bias, A, t, c, N, C)
    go = tl.load(GO+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :],
                 (t[:, None, None]<N)&(c[None, :, None]<C), 0).to(tl.float32)
    dm = x[:, :, :, None]*go[:, :, None, :]
    dx = tl.sum(m*go[:, :, None, :], 3)
    dv = tl.full((BT, BC, 4), 0., tl.float32)
    a = tl.load(A)
    for k in tl.static_range(4):
        w = tl.load(W+c[:, None, None]*64+i[None, :, :]*16+j[None, :, :]*4+k,
                    c[:, None, None]<C, 0)
        dvk = a*tl.sum(tl.sum(dm*w[None, :, :, :], 3), 2)
        dv += tl.where(r[None, None, :]==k, dvk[:, :, None], 0.)
        vk = tl.sum(tl.where(r[None, None, :]==k, v, 0.), 2)
        dw = tl.sum(dm*(a*vk[:, :, None, None]), 0)
        off = (split*C+c[:, None, None])*81+i[None, :, :]*16+j[None, :, :]*4+k
        tl.store(Parts+off, dw, c[:, None, None]<C)
    dx += inv[:, :, None]*(dv-v*(tl.sum(dv*v, 2)*0.25)[:, :, None])
    tl.store(GX+(t[:, None, None]*C+c[None, :, None])*4+r[None, None, :], dx,
             (t[:, None, None]<N)&(c[None, :, None]<C))
    db = tl.sum(dm, 0)
    tl.store(Parts+(split*C+c[:, None, None])*81+64+i[None, :, :]*4+j[None, :, :], db, c[:, None, None]<C)
    da = tl.sum(tl.sum(tl.sum(dm*raw, 3), 2), 0)
    tl.store(Parts+(split*C+c)*81+80, da, c<C)


@tr.jit
def _backward_branch(Y, Post, GO, GY, GPost, N: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, BC)
    r = tl.arange(0, 4)
    go = tl.load(GO+(t*C+c[:, None])*4+r[None, :], c[:, None]<C, 0).to(tl.float32)
    post = tl.load(Post+t*4+r)
    y = tl.load(Y+t*C+c, c<C, 0).to(tl.float32)
    tl.store(GY+t*C+c, tl.sum(go*post[None, :], 1), c<C)
    tl.store(GPost+t*4+r, tl.sum(go*y[:, None], 0))


class _ChannelMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, post, weight, bias, scale):
        if x.dtype != torch.bfloat16 or y.dtype != x.dtype:
            raise ValueError('Channel mixing requires BF16 activations')
        if not all(t.is_contiguous() for t in (x, y, post, weight, bias, scale)):
            raise ValueError('Channel mixing requires contiguous inputs')
        if any(t.dtype != torch.float32 for t in (post, weight, bias, scale)):
            raise ValueError('Routing parameters and coefficients must be FP32')
        c = x.shape[-2]; n = x.numel()//(4*c)
        if x.shape[-1] != 4 or y.shape != x.shape[:-1] or post.shape != (*x.shape[:-2], 4):
            raise ValueError('Invalid channel mixing activation shapes')
        if weight.shape != (c,4,4,4) or bias.shape != (c,4,4) or scale.shape != (1,):
            raise ValueError('Invalid channel mixing parameter shapes')
        out = torch.empty_like(x)
        _forward[(tr.cdiv(n,32),tr.cdiv(c,8))](x,y,post,weight,bias,scale,out,n,c,32,8)
        ctx.save_for_backward(x,y,post,weight,bias,scale)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, go):
        x,y,post,weight,bias,scale = ctx.saved_tensors
        go = go.contiguous()
        c=x.shape[-2]; n=x.numel()//(4*c)
        gx=torch.empty_like(x); gy=torch.empty_like(y); gp=torch.empty_like(post)
        # Temporary deterministic reduction workspace, not a saved activation.
        parts=torch.empty((tr.cdiv(n,128),c,81),device=x.device,dtype=torch.float32)
        _backward_mix[(tr.cdiv(n,128),tr.cdiv(c,4))](x,weight,bias,scale,go,gx,parts,n,c,128,4)
        _backward_branch[(n,)](y,post,go,gy,gp,n,c,tr.next_power_of_2(c))
        reduced=parts.sum(0)
        gw=reduced[:,:64].reshape_as(weight).contiguous()
        gb=reduced[:,64:80].reshape_as(bias).contiguous()
        ga=reduced[:,80].sum().reshape_as(scale)
        return gx,gy,gp,gw,gb,ga


def channel_mix(x, y, post, weight, bias, scale):
    return _ChannelMix.apply(x, y, post, weight, bias, scale)
