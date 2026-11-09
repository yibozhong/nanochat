"""Two-channel, content-dependent Cayley rotation of ReLU-squared features.

For each adjacent pair u,v and learned w:
  a=w/sqrt(1+w*w), t=a*(u-v)/(1+abs(u)+abs(v))
  c=(1-t*t)/(1+t*t), s=2*t/(1+t*t)
  output=(c*relu(u)^2-s*relu(v)^2, s*relu(u)^2+c*relu(v)^2).
The rotation preserves each activated pair's norm in exact arithmetic. This
is not a bound on the full input-dependent Jacobian. w=0 recovers ReLU^2.
All pairs/tokens are independent; there is no additional GEMM or state.
"""
import torch
from torch import nn
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from nanochat.scaled_reduction import reduce_scale_partials


def pairwise_torch(x, weight):
    dtype=torch.float64 if x.dtype==torch.float64 else torch.float32
    z=x.to(dtype).reshape(*x.shape[:-1],-1,2);u,v=z.unbind(-1)
    w=weight.to(dtype);a=w*torch.rsqrt(1+w.square())
    t=a*(u-v)/(1+u.abs()+v.abs());q=(1+t.square()).reciprocal()
    c=(1-t.square())*q;s=2*t*q
    p=u.relu().square();r=v.relu().square()
    return torch.stack((c*p-s*r,s*p+c*r),-1).flatten(-2).to(x.dtype)


@tr.jit
def _pair_fwd(X,W,Y,M:tl.constexpr,P:tl.constexpr,BM:tl.constexpr,BP:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BP+tl.arange(0,BP)
    mask=(rows[:,None]<M)&(cols[None,:]<P)
    offsets=rows[:,None]*(2*P)+2*cols[None,:]
    u=tl.load(X+offsets,mask,0).to(tl.float32)
    v=tl.load(X+offsets+1,mask,0).to(tl.float32)
    w=tl.load(W+cols,cols<P,0)
    strength=w*tl.rsqrt(1+w*w)
    t=strength[None,:]*(u-v)/(1+tl.abs(u)+tl.abs(v))
    tt=t*t;q=1/(1+tt);c=(1-tt)*q;s=2*t*q
    pu=tl.maximum(u,0);pv=tl.maximum(v,0);a=pu*pu;b=pv*pv
    tl.store(Y+offsets,c*a-s*b,mask)
    tl.store(Y+offsets+1,s*a+c*b,mask)


@tr.jit
def _pair_bwd(X,W,G,DX,DW,M:tl.constexpr,P:tl.constexpr,BM:tl.constexpr,BP:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BP+tl.arange(0,BP)
    mask=(rows[:,None]<M)&(cols[None,:]<P)
    offsets=rows[:,None]*(2*P)+2*cols[None,:]
    u=tl.load(X+offsets,mask,0).to(tl.float32)
    v=tl.load(X+offsets+1,mask,0).to(tl.float32)
    g1=tl.load(G+offsets,mask,0).to(tl.float32)
    g2=tl.load(G+offsets+1,mask,0).to(tl.float32)
    w=tl.load(W+cols,cols<P,0)
    inv_w=tl.rsqrt(1+w*w);strength=w*inv_w
    absu=tl.abs(u);absv=tl.abs(v);inv=1/(1+absu+absv)
    ratio=(u-v)*inv;t=strength[None,:]*ratio
    tt=t*t;q=1/(1+tt);c=(1-tt)*q;s=2*t*q
    pu=tl.maximum(u,0);pv=tl.maximum(v,0);a=pu*pu;b=pv*pv
    # Analytic derivatives avoid cancellation near saturated rotation strength.
    dc=-4*t*q*q;ds=2*(1-tt)*q*q
    dt=(g1*a+g2*b)*dc+(g2*a-g1*b)*ds
    signu=tl.where(u>0,1.,tl.where(u<0,-1.,0.))
    signv=tl.where(v>0,1.,tl.where(v<0,-1.,0.))
    shared=dt*strength[None,:]*inv*inv
    du=2*pu*(c*g1+s*g2)+shared*(1+absv+v*signu)
    dv=2*pv*(-s*g1+c*g2)-shared*(1+absu+u*signv)
    tl.store(DX+offsets,du,mask)
    tl.store(DX+offsets+1,dv,mask)
    dw=tl.sum(dt*ratio,axis=0)*(inv_w*inv_w*inv_w)
    tl.store(DW+tl.program_id(0)*P+cols,dw,cols<P)


class _Pairwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        if x.ndim<2 or x.shape[-1]%2 or x.numel()==0 or not x.is_cuda:
            raise ValueError('Pairwise activation requires nonempty CUDA [...,even width]')
        if x.dtype not in (torch.bfloat16,torch.float32) or weight.dtype!=torch.float32:
            raise ValueError('Pairwise activation uses BF16/FP32 input and FP32 parameters')
        if not x.is_contiguous() or not weight.is_contiguous() or weight.device!=x.device or weight.shape!=(x.shape[-1]//2,):
            raise ValueError('Invalid pairwise shape/layout/device')
        p=x.shape[-1]//2;m=x.numel()//(2*p);out=torch.empty_like(x)
        _pair_fwd[(tr.cdiv(m,16),tr.cdiv(p,64))](x,weight,out,m,p,16,64,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(x,weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,w=ctx.saved_tensors;go=go.contiguous();p=x.shape[-1]//2;m=x.numel()//(2*p)
        dx=torch.empty_like(x);parts=torch.empty((1,tr.cdiv(m,16),p,1),device=x.device,dtype=torch.float32)
        _pair_bwd[(tr.cdiv(m,16),tr.cdiv(p,64))](x,w,go,dx,parts,m,p,16,64,num_warps=4,enable_fp_fusion=False)
        dw=reduce_scale_partials(parts).reshape(p)
        return dx,dw


def pairwise_activation(x,weight):
    return _Pairwise.apply(x,weight)


class PairwiseMLP(nn.Module):
    def __init__(self,original,backend='triton'):
        super().__init__()
        self.c_fc=original.c_fc;self.c_proj=original.c_proj
        self.rotation=nn.Parameter(torch.zeros(self.c_fc.weight.shape[0]//2))
        self.backend=backend

    def forward(self,x):
        z=self.c_fc(x)
        h=pairwise_activation(z,self.rotation) if self.backend=='triton' else pairwise_torch(z,self.rotation)
        return self.c_proj(h)
