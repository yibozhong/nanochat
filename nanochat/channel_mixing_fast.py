"""Optional channel mixing tiling and fused branch backward.

Uses the legacy coefficient generator and tensor reduction expressions.
Fused variants reuse GO for branch gradients and deterministic dpost partials.
No saved coefficient bank, atomics or precision/parameter changes.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.channel_mixing import _backward_branch, _coeff, _forward as _channel_fast_forward

# Keep the legacy forward launch geometry: changing its layout perturbs BF16
# rounding enough to fail the full-model routing-gradient tolerance.
# forward BT/BC/warps, backward BT/BC/warps, fuse branch backward.
GEOMETRIES = {
    'split': (32,8,8,128,8,8,False),
    'fused': (32,8,8,128,8,8,True),
    'fused_wide': (32,8,8,64,16,8,True),
}


@tr.jit
def _channel_fast_backward(X,Y,Post,W,Bias,A,GO,GX,GY,Parts,PostParts,
                           N:tl.constexpr,C:tl.constexpr,BT:tl.constexpr,BC:tl.constexpr,FUSE:tl.constexpr):
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
    if FUSE:
        post=tl.load(Post+t[:,None]*4+r[None,:],t[:,None]<N,0)
        y=tl.load(Y+t[:,None]*C+c[None,:],(t[:,None]<N)&(c[None,:]<C),0).to(tl.float32)
        tl.store(GY+t[:,None]*C+c[None,:],tl.sum(go*post[:,None,:],2),(t[:,None]<N)&(c[None,:]<C))
        gp=tl.sum(go*y[:,:,None],1)
        tl.store(PostParts+(t[:,None]*tl.cdiv(C,BC)+tl.program_id(1))*4+r[None,:],gp,t[:,None]<N)


class _ChannelMixFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,post,weight,bias,scale,kernel):
        c=x.shape[-2];n=x.numel()//(4*c)
        if not x.is_cuda or x.ndim!=4 or x.shape[-1]!=4 or x.dtype!=torch.bfloat16 or y.dtype!=x.dtype:
            raise ValueError('Expected CUDA BF16 four-stream activations')
        if y.shape!=x.shape[:-1] or post.shape!=(*x.shape[:2],4) or weight.shape!=(c,4,4,4) or bias.shape!=(c,4,4) or scale.shape!=(1,):
            raise ValueError('Invalid channel mixing shapes')
        if not all(p.is_contiguous() and p.device==x.device for p in (x,y,post,weight,bias,scale)):
            raise ValueError('Expected contiguous colocated tensors')
        if any(p.dtype!=torch.float32 for p in (post,weight,bias,scale)):
            raise ValueError('Channel coefficients and parameters remain FP32')
        ft,fc,fw,bt,bc,bw,fuse=GEOMETRIES[kernel]
        out=torch.empty_like(x)
        _channel_fast_forward[(tr.cdiv(n,ft),tr.cdiv(c,fc))](x,y,post,weight,bias,scale,out,n,c,ft,fc,num_warps=fw)
        ctx.save_for_backward(x,y,post,weight,bias,scale);ctx.geometry=(bt,bc,bw,fuse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,y,post,weight,bias,scale=ctx.saved_tensors;go=go.contiguous()
        c=x.shape[-2];n=x.numel()//(4*c);bt,bc,bw,fuse=ctx.geometry
        gx=torch.empty_like(x);gy=torch.empty_like(y)
        parts=torch.empty((tr.cdiv(n,bt),c,81),device=x.device,dtype=torch.float32)
        pp=torch.empty((n,tr.cdiv(c,bc),4),device=x.device,dtype=torch.float32) if fuse else torch.empty_like(post)
        _channel_fast_backward[(tr.cdiv(n,bt),tr.cdiv(c,bc))](x,y,post,weight,bias,scale,go,gx,gy,parts,pp,n,c,bt,bc,fuse,num_warps=bw)
        if fuse:gp=pp.sum(1).reshape_as(post)
        else:
            gp=pp
            _backward_branch[(n,)](y,post,go,gy,gp,n,c,tr.next_power_of_2(c))
        reduced=parts.sum(0)
        return gx,gy,gp,reduced[:,:64].reshape_as(weight).contiguous(),reduced[:,64:80].reshape_as(bias).contiguous(),reduced[:,80].sum().reshape_as(scale),None


def channel_mix_fast(x,y,post,weight,bias,scale,*,kernel='fused'):
    return _ChannelMixFast.apply(x,y,post,weight,bias,scale,kernel)
