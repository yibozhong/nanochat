"""Fused nanochat RoPE, RMSNorm and 1.2 scale, preserving BF16 round points.

Cosine/sine are constant model buffers. Backward recomputes the rotated input
and retains the eager BF16 rounding at scale, RMSNorm, and RoPE boundaries.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _rotate(X, C, S, row, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            BD: tl.constexpr, N: tl.constexpr):
    j=tl.arange(0,BD)
    valid=(row[:,None]<N)&(j[None,:]<D//2)
    pos=row//H%T
    x1=tl.load(X+row[:,None]*D+j[None,:],valid,0).to(tl.float32)
    x2=tl.load(X+row[:,None]*D+D//2+j[None,:],valid,0).to(tl.float32)
    c=tl.load(C+pos[:,None]*(D//2)+j[None,:],valid,0).to(tl.float32)
    s=tl.load(S+pos[:,None]*(D//2)+j[None,:],valid,0).to(tl.float32)
    dtype:tl.constexpr=X.dtype.element_ty
    y1=((x1*c).to(dtype).to(tl.float32)+(x2*s).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    y2=((x1*(-s)).to(dtype).to(tl.float32)+(x2*c).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    return y1,y2,c,s


@tr.jit
def _fwd(X,C,S,Y,INV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
         EPS:tl.constexpr,BD:tl.constexpr,N:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR)
    j=tl.arange(0,BD)
    valid=(row[:,None]<N)&(j[None,:]<D//2)
    y1,y2,c,s=_rotate(X,C,S,row,T,H,D,BD,N)
    inv=tl.rsqrt(tl.sum(y1*y1+y2*y2,1)/D+EPS)
    dtype:tl.constexpr=X.dtype.element_ty
    o1=(y1*inv[:,None]).to(dtype).to(tl.float32)*1.2
    o2=(y2*inv[:,None]).to(dtype).to(tl.float32)*1.2
    tl.store(Y+row[:,None]*D+j[None,:],o1,valid)
    tl.store(Y+row[:,None]*D+D//2+j[None,:],o2,valid)
    tl.store(INV+row,inv,row<N)


@tr.jit
def _bwd(X,C,S,INV,DO,DX,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,N:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR)
    j=tl.arange(0,BD)
    dtype:tl.constexpr=X.dtype.element_ty
    valid=(row[:,None]<N)&(j[None,:]<D//2)
    y1,y2,c,s=_rotate(X,C,S,row,T,H,D,BD,N)
    g1=(tl.load(DO+row[:,None]*D+j[None,:],valid,0).to(tl.float32)*1.2).to(dtype).to(tl.float32)
    g2=(tl.load(DO+row[:,None]*D+D//2+j[None,:],valid,0).to(tl.float32)*1.2).to(dtype).to(tl.float32)
    inv=tl.load(INV+row,row<N,0)
    correction=tl.sum(g1*y1+g2*y2,1)/D*(inv*inv)
    dy1=((g1-y1*correction[:,None])*inv[:,None]).to(dtype).to(tl.float32)
    dy2=((g2-y2*correction[:,None])*inv[:,None]).to(dtype).to(tl.float32)
    dx1=(dy1*c).to(dtype).to(tl.float32)+(dy2*(-s)).to(dtype).to(tl.float32)
    dx2=(dy1*s).to(dtype).to(tl.float32)+(dy2*c).to(dtype).to(tl.float32)
    tl.store(DX+row[:,None]*D+j[None,:],dx1,valid)
    tl.store(DX+row[:,None]*D+D//2+j[None,:],dx2,valid)


class _RotaryNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,cos,sin):
        x=x.contiguous();cos=cos.contiguous();sin=sin.contiguous()
        b,t,h,d=x.shape
        out=torch.empty_like(x)
        inv=torch.empty(b*t*h,device=x.device,dtype=torch.float32)
        _fwd[(tr.cdiv(b*t*h,8),)](x,cos,sin,out,inv,t,h,d,torch.finfo(x.dtype).eps,
                       tr.next_power_of_2(d//2),b*t*h,8,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,cos,sin,inv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,cos,sin,inv=ctx.saved_tensors
        b,t,h,d=x.shape
        dx=torch.empty_like(x)
        _bwd[(tr.cdiv(b*t*h,8),)](x,cos,sin,inv,grad.contiguous(),dx,t,h,d,
                       tr.next_power_of_2(d//2),b*t*h,8,num_warps=4,enable_fp_fusion=False)
        return dx,None,None


def rotary_norm(x,cos,sin):
    if cos.requires_grad or sin.requires_grad:
        raise ValueError('Fused HCA RoPE expects constant cosine/sine buffers')
    return _RotaryNorm.apply(x,cos,sin)
