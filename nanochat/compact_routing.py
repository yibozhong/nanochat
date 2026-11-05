"""Eight-coefficient mHC read/write projection, RMS scaling and sigmoid.

Same formula and precision policy as NVIDIA TE 2.20.2, without the unused
16 residual-mixing coefficients. GEMMs use 16-wide tiles instead of 32;
the eight remaining padded lanes are tensor-core alignment, not parameters.
All reductions are deterministic; no atomic parameter-gradient updates.
"""
import torch
import triton as tr
import triton.language as tl
from torch.autograd.function import once_differentiable


@tr.jit
def _forward(X,Phi,Alpha,Beta,H,U,R,M:tl.constexpr,K:tl.constexpr,
             TMA:tl.constexpr,PREC:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    m=tile*BM+tl.arange(0,BM)
    n=tl.arange(0,16)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,16),tl.float32)
    ms=tl.zeros((BM,BK),tl.float32)
    if TMA:
        xd=tl.make_tensor_descriptor(X,shape=[M,K],strides=[K,1],block_shape=[BM,BK])
        pd=tl.make_tensor_descriptor(Phi,shape=[8,K],strides=[K,1],block_shape=[16,BK])
    for off in range(tl.cdiv(K,BK)):
        kk=off*BK+k
        if TMA:
            x=tl.load_tensor_descriptor(xd,[tile*BM,off*BK])
            p=tl.load_tensor_descriptor(pd,[0,off*BK])
        else:
            x=tl.load(X+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0)
            p=tl.load(Phi+n[:,None]*K+kk[None,:],(n[:,None]<8)&(kk[None,:]<K),0)
        xf=x.to(tl.float32)
        ms+=xf*xf
        acc=tl.dot(xf,tl.trans(p),acc,input_precision=PREC)
    # Match TE's fp32 epsilon and sqrt/reciprocal rather than changing norm.
    r=1./tl.sqrt(tl.sum(ms,1)/K+1.1920928955078125e-7)
    u=acc*r[:,None]
    a=tl.load(Alpha+n//4,n<8,0)
    b=tl.load(Beta+n,n<8,0)
    logits=tl.fma(acc*a[None,:],r[:,None],b[None,:])
    h=tl.sigmoid(logits)*tl.where(n<4,1.,2.)[None,:]
    mask=(m[:,None]<M)&(n[None,:]<8)
    tl.store(H+m[:,None]*8+n[None,:],h,mask)
    tl.store(U+m[:,None]*8+n[None,:],u,mask)
    tl.store(R+m,r,m<M)


@tr.jit
def _backward_scale(G,H,U,R,Alpha,GH,GMS,Parts,M:tl.constexpr,BM:tl.constexpr):
    tile=tl.program_id(0);m=tile*BM+tl.arange(0,BM);n=tl.arange(0,8)
    h=tl.load(H+m[:,None]*8+n[None,:],m[:,None]<M,0)
    u=tl.load(U+m[:,None]*8+n[None,:],m[:,None]<M,0)
    g=tl.load(G+m[:,None]*8+n[None,:],m[:,None]<M,0)
    r=tl.load(R+m,m<M,0)
    alpha=tl.load(Alpha+n//4)
    gz=g*h*(1.-h/tl.where(n<4,1.,2.)[None,:])
    gh=gz*alpha[None,:]*r[:,None]
    gms=-.5*r*r*tl.sum(gz*alpha[None,:]*u,1)
    tl.store(GH+m[:,None]*8+n[None,:],gh,m[:,None]<M)
    tl.store(GMS+m,gms,m<M)
    tl.store(Parts+tile*16+n,tl.sum(gz*u,0))
    tl.store(Parts+tile*16+8+n,tl.sum(gz,0))


@tr.jit
def _backward_x(X,Phi,GH,GMS,GX,M:tl.constexpr,K:tl.constexpr,
                PREC:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.program_id(1)*BK+tl.arange(0,BK);n=tl.arange(0,16)
    gh=tl.load(GH+m[:,None]*8+n[None,:],(m[:,None]<M)&(n[None,:]<8),0)
    p=tl.load(Phi+n[:,None]*K+k[None,:],(n[:,None]<8)&(k[None,:]<K),0)
    x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0).to(tl.float32)
    gms=tl.load(GMS+m,m<M,0)
    acc=x*(gms*(2./K))[:,None]
    gx=tl.dot(gh,p,acc,input_precision=PREC)
    tl.store(GX+m[:,None]*K+k[None,:],gx,(m[:,None]<M)&(k[None,:]<K))


@tr.jit
def _backward_phi(X,GH,Parts,M:tl.constexpr,K:tl.constexpr,PREC:tl.constexpr,
                  BM:tl.constexpr,BK:tl.constexpr,STEP:tl.constexpr):
    tile=tl.program_id(0);k=tl.program_id(1)*BK+tl.arange(0,BK)
    n=tl.arange(0,16);mr=tl.arange(0,STEP)
    acc=tl.zeros((16,BK),tl.float32)
    for off in range(BM//STEP):
        m=tile*BM+off*STEP+mr
        x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0).to(tl.float32)
        gh=tl.load(GH+m[:,None]*8+n[None,:],(m[:,None]<M)&(n[None,:]<8),0)
        acc=tl.dot(tl.trans(gh),x,acc,input_precision=PREC)
    tl.store(Parts+(tile*8+n[:,None])*K+k[None,:],acc,(n[:,None]<8)&(k[None,:]<K))


class _CompactRouting(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,phi,alpha,beta,use_tf32):
        if x.ndim!=2 or x.dtype!=torch.bfloat16 or not x.is_cuda:
            raise ValueError('Compact routing expects CUDA BF16 [tokens,4D]')
        m,k=x.shape
        if k%128 or phi.shape!=(8,k) or alpha.shape!=(2,) or beta.shape!=(8,):
            raise ValueError('Compact routing requires8 read/write coefficients and4D divisible by128')
        if not all(t.is_contiguous() for t in (x,phi,alpha,beta)):
            raise ValueError('Compact routing requires contiguous inputs')
        if any(t.dtype!=torch.float32 for t in (phi,alpha,beta)):
            raise ValueError('Compact routing parameters must be FP32')
        # Keep the same explicit TF32 policy as TE; no BF16 parameter rounding.
        prec='tf32' if use_tf32 else 'tf32x3'
        tma=torch.cuda.get_device_capability(x.device)[0]>=9
        if tma:
            from nanochat.nvidia_ops import nvidia_ops
            nvidia_ops()  # Registers TE's allocator, including autograd contexts.
        h=torch.empty((m,8),device=x.device,dtype=torch.float32)
        u=torch.empty_like(h);r=torch.empty(m,device=x.device,dtype=torch.float32)
        _forward[(tr.cdiv(m,32),)](x,phi,alpha,beta,h,u,r,m,k,tma,prec,32,256,num_warps=4,num_stages=2)
        ctx.save_for_backward(x,phi,alpha,h,u,r);ctx.prec=prec
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        x,phi,alpha,h,u,r=ctx.saved_tensors;m,k=x.shape;go=go.contiguous()
        gh=torch.empty_like(h);gms=torch.empty_like(r);gx=torch.empty_like(x)
        small=torch.empty((tr.cdiv(m,128),16),device=x.device,dtype=torch.float32)
        _backward_scale[(tr.cdiv(m,128),)](go,h,u,r,alpha,gh,gms,small,m,128,num_warps=4)
        _backward_x[(tr.cdiv(m,32),tr.cdiv(k,128))](x,phi,gh,gms,gx,m,k,ctx.prec,32,128,num_warps=4)
        parts=torch.empty((tr.cdiv(m,512),8,k),device=x.device,dtype=torch.float32)
        _backward_phi[(tr.cdiv(m,512),tr.cdiv(k,128))](x,gh,parts,m,k,ctx.prec,512,128,32,num_warps=4,num_stages=2)
        gp=parts.sum(0);reduced=small.sum(0)
        ga=reduced[:8].reshape(2,4).sum(1);gb=reduced[8:].contiguous()
        return gx,gp,ga,gb,None


def compact_routing(x,phi,alpha,beta,*,use_tf32=True):
    return _CompactRouting.apply(x,phi,alpha,beta,use_tf32)
