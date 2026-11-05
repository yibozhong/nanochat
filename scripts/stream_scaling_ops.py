"""Benchmark-only 4/8/16/32-stream kernels. Does not alter production connectors."""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable
from scripts.stream_scaling_sinkhorn import (_mhc_sinkhorn_fwd_fused_recompute,
                                             _mhc_sinkhorn_bwd_fused_recompute)


class Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        n=x.shape[-1];m=x.numel()//(n*n);x=x.contiguous();out=torch.empty_like(x)
        # Keep the number of live matrix entries bounded as n grows.
        bt=max(1,128//(n*n));block=bt*n*n
        _mhc_sinkhorn_fwd_fused_recompute[(tr.cdiv(m,bt),)](
            x,out,n*n,1,n*n,1,m,n,block,20,num_warps=4)
        ctx.save_for_backward(x,out);ctx.bt=bt
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,out=ctx.saved_tensors;n=x.shape[-1];m=x.numel()//(n*n);go=go.contiguous();gx=torch.empty_like(x)
        hf=torch.empty((21,m,n),device=x.device,dtype=torch.float32);hg=torch.empty_like(hf)
        _mhc_sinkhorn_bwd_fused_recompute[(tr.cdiv(m,ctx.bt),)](
            go,out,gx,x,hf,hg,n*n,1,n*n,1,n*n,1,n*n,1,m,n,ctx.bt*n*n,20,num_warps=4)
        return gx


@tr.jit
def _aggregate_fwd(X,P,Y,M:tl.constexpr,C:tl.constexpr,N:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,BC);s=tl.arange(0,N)
    x=tl.load(X+(t*C+c[:,None])*N+s[None,:],c[:,None]<C,0).to(tl.float32)
    p=tl.load(P+t*N+s)
    tl.store(Y+t*C+c,tl.sum(x*p[None,:],1),c<C)


@tr.jit
def _aggregate_bwd(X,P,G,GX,GP,C:tl.constexpr,N:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,BC);s=tl.arange(0,N)
    x=tl.load(X+(t*C+c[:,None])*N+s[None,:],c[:,None]<C,0).to(tl.float32)
    p=tl.load(P+t*N+s);g=tl.load(G+t*C+c,c<C,0).to(tl.float32)
    tl.store(GX+(t*C+c[:,None])*N+s[None,:],g[:,None]*p[None,:],c[:,None]<C)
    tl.store(GP+t*N+s,tl.sum(g[:,None]*x,0))


class Aggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,p):
        x=x.contiguous();p=p.contiguous();m,c,n=x.shape
        y=torch.empty((m,c),device=x.device,dtype=x.dtype)
        _aggregate_fwd[(m,)](x,p,y,m,c,n,tr.next_power_of_2(c),num_warps=4)
        ctx.save_for_backward(x,p);return y

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,p=ctx.saved_tensors;m,c,n=x.shape;gx=torch.empty_like(x);gp=torch.empty_like(p)
        _aggregate_bwd[(m,)](x,p,g.contiguous(),gx,gp,c,n,tr.next_power_of_2(c),num_warps=4)
        return gx,gp


@tr.jit
def _write_fwd(X,Y,P,A,O,C:tl.constexpr,N:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0);c=tl.program_id(1)*BC+tl.arange(0,BC);s=tl.arange(0,N)
    acc=tl.zeros((BC,N),tl.float32)
    for k in tl.static_range(N):
        xk=tl.load(X+(t*C+c)*N+k,c<C,0).to(tl.float32)
        a=tl.load(A+(t*N+k)*N+s)
        acc=tl.fma(xk[:,None],a[None,:],acc)
    y=tl.load(Y+t*C+c,c<C,0).to(tl.float32);p=tl.load(P+t*N+s)
    tl.store(O+(t*C+c[:,None])*N+s[None,:],tl.fma(y[:,None],p[None,:],acc),c[:,None]<C)


@tr.jit
def _write_bwd(X,Y,P,A,G,GX,GY,GP,GA,C:tl.constexpr,N:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,BC);s=tl.arange(0,N)
    x=tl.load(X+(t*C+c[:,None])*N+s[None,:],c[:,None]<C,0).to(tl.float32)
    g=tl.load(G+(t*C+c[:,None])*N+s[None,:],c[:,None]<C,0).to(tl.float32)
    y=tl.load(Y+t*C+c,c<C,0).to(tl.float32);p=tl.load(P+t*N+s)
    dx=tl.zeros((BC,N),tl.float32)
    for k in tl.static_range(N):
        gk=tl.sum(tl.where(s[None,:]==k,g,0.),1)
        ak=tl.load(A+t*N*N+s*N+k)
        dx=tl.fma(gk[:,None],ak[None,:],dx)
        da=tl.sum(x*gk[:,None],0)
        tl.store(GA+t*N*N+s*N+k,da)
    tl.store(GX+(t*C+c[:,None])*N+s[None,:],dx,c[:,None]<C)
    tl.store(GY+t*C+c,tl.sum(g*p[None,:],1),c<C)
    tl.store(GP+t*N+s,tl.sum(g*y[:,None],0))


class Write(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,p,a):
        x=x.contiguous();y=y.contiguous();p=p.contiguous();a=a.contiguous();m,c,n=x.shape
        out=torch.empty_like(x)
        _write_fwd[(m,tr.cdiv(c,64))](x,y,p,a,out,c,n,64,num_warps=4)
        ctx.save_for_backward(x,y,p,a);return out

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,y,p,a=ctx.saved_tensors;m,c,n=x.shape
        gx=torch.empty_like(x);gy=torch.empty_like(y);gp=torch.empty_like(p);ga=torch.empty_like(a)
        _write_bwd[(m,)](x,y,p,a,g.contiguous(),gx,gy,gp,ga,c,n,tr.next_power_of_2(c),num_warps=8)
        return gx,gy,gp,ga


@tr.jit
def _mlp_values(X,WI,WO,row,R:tl.constexpr,N:tl.constexpr,H:tl.constexpr,PN:tl.constexpr):
    s=tl.arange(0,PN);j=tl.arange(0,H)
    x=tl.load(X+row[:,None]*N+s[None,:],(row[:,None]<R)&(s[None,:]<N),0).to(tl.float32)
    wi=tl.load(WI+j[:,None]*N+s[None,:],s[None,:]<N,0)
    wo=tl.load(WO+s[:,None]*H+j[None,:],s[:,None]<N,0)
    z=tl.dot(x,tl.trans(wi),input_precision='tf32x3')
    return x,wi,wo,z,tl.maximum(z,0.)


@tr.jit
def _mlp_fwd(X,Y,P,WI,WO,O,R:tl.constexpr,C:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
             PN:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR);s=tl.arange(0,PN)
    x,wi,wo,z,a=_mlp_values(X,WI,WO,row,R,N,H,PN)
    mixed=tl.dot(a,tl.trans(wo),input_precision='tf32x3')
    y=tl.load(Y+row,row<R,0).to(tl.float32)
    p=tl.load(P+(row[:,None]//C)*N+s[None,:],(row[:,None]<R)&(s[None,:]<N),0)
    tl.store(O+row[:,None]*N+s[None,:],mixed+y[:,None]*p,(row[:,None]<R)&(s[None,:]<N))


@tr.jit
def _mlp_dx(X,WI,WO,G,GX,R:tl.constexpr,N:tl.constexpr,H:tl.constexpr,PN:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0)*BR+tl.arange(0,BR);s=tl.arange(0,PN)
    x,wi,wo,z,a=_mlp_values(X,WI,WO,row,R,N,H,PN)
    g=tl.load(G+row[:,None]*N+s[None,:],(row[:,None]<R)&(s[None,:]<N),0).to(tl.float32)
    dz=tl.where(z>0.,tl.dot(g,wo,input_precision='tf32x3'),0.)
    dx=tl.dot(dz,wi,input_precision='tf32x3')
    tl.store(GX+row[:,None]*N+s[None,:],dx,(row[:,None]<R)&(s[None,:]<N))


@tr.jit
def _mlp_dw(X,WI,WO,G,Parts,R:tl.constexpr,N:tl.constexpr,H:tl.constexpr,PN:tl.constexpr,
            SPLIT:tl.constexpr,BR:tl.constexpr,BH:tl.constexpr):
    tile=tl.program_id(0);rr=tl.arange(0,BR);s=tl.arange(0,PN)
    j=tl.program_id(1)*BH+tl.arange(0,BH)
    wi=tl.load(WI+j[:,None]*N+s[None,:],(j[:,None]<H)&(s[None,:]<N),0)
    wo=tl.load(WO+s[:,None]*H+j[None,:],(s[:,None]<N)&(j[None,:]<H),0)
    dwi=tl.zeros((BH,PN),tl.float32);dwo=tl.zeros((PN,BH),tl.float32)
    for off in range(SPLIT//BR):
        row=tile*SPLIT+off*BR+rr
        x=tl.load(X+row[:,None]*N+s[None,:],(row[:,None]<R)&(s[None,:]<N),0).to(tl.float32)
        g=tl.load(G+row[:,None]*N+s[None,:],(row[:,None]<R)&(s[None,:]<N),0).to(tl.float32)
        z=tl.dot(x,tl.trans(wi),input_precision='tf32x3');a=tl.maximum(z,0.)
        dz=tl.where(z>0.,tl.dot(g,wo,input_precision='tf32x3'),0.)
        dwi=tl.dot(tl.trans(dz),x,dwi,input_precision='tf32x3')
        dwo=tl.dot(tl.trans(g),a,dwo,input_precision='tf32x3')
    tl.store(Parts+tile*(2*N*H)+j[:,None]*N+s[None,:],dwi,(j[:,None]<H)&(s[None,:]<N))
    tl.store(Parts+tile*(2*N*H)+N*H+s[:,None]*H+j[None,:],dwo,(s[:,None]<N)&(j[None,:]<H))


@tr.jit
def _branch_bwd(Y,P,G,GY,GP,C:tl.constexpr,N:tl.constexpr,BC:tl.constexpr):
    t=tl.program_id(0);c=tl.arange(0,BC);s=tl.arange(0,N)
    g=tl.load(G+(t*C+c[:,None])*N+s[None,:],c[:,None]<C,0).to(tl.float32)
    y=tl.load(Y+t*C+c,c<C,0).to(tl.float32);p=tl.load(P+t*N+s)
    tl.store(GY+t*C+c,tl.sum(g*p[None,:],1),c<C)
    tl.store(GP+t*N+s,tl.sum(g*y[:,None],0))


class MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,y,p,wi,wo):
        x=x.contiguous();y=y.contiguous();p=p.contiguous();m,c,n=x.shape;h=wi.shape[0]
        assert n in (4,8,16,32) and h==4*n
        out=torch.empty_like(x)
        br=32 if n>=16 else 64
        _mlp_fwd[(tr.cdiv(m*c,br),)](x,y,p,wi,wo,out,m*c,c,n,h,max(16,n),br,num_warps=4)
        ctx.save_for_backward(x,y,p,wi,wo);return out

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,y,p,wi,wo=ctx.saved_tensors;g=g.contiguous();m,c,n=x.shape;h=wi.shape[0];r=m*c
        gx=torch.empty_like(x);gy=torch.empty_like(y);gp=torch.empty_like(p)
        parts=torch.empty((tr.cdiv(r,1024),2*n*h),device=x.device,dtype=torch.float32)
        br=32 if n>=16 else 64
        _mlp_dx[(tr.cdiv(r,br),)](x,wi,wo,g,gx,r,n,h,max(16,n),br,num_warps=4)
        bh=min(32,h)
        _mlp_dw[(tr.cdiv(r,1024),tr.cdiv(h,bh))](x,wi,wo,g,parts,r,n,h,max(16,n),1024,32,bh,num_warps=4)
        _branch_bwd[(m,)](y,p,g,gy,gp,c,n,tr.next_power_of_2(c),num_warps=4)
        reduced=parts.sum(0)
        return gx,gy,gp,reduced[:n*h].reshape_as(wi).contiguous(),reduced[n*h:].reshape_as(wo).contiguous()
