"""Fused single-query attention over projected previous-token layer K/V.

query[B,T,D], packed_kv[B,T,L,2D]. All three projections receive gradients.
Each memory row is used by just one following query: dK/dV need no atomics.
The masked first query writes zero gradients to the unused final memory row.
"""
import torch
import triton
import triton.language as tl


def projected_read_reference(query, kv, shift=True):
    dtype=torch.float64 if query.dtype==torch.float64 else torch.float32
    key,value=kv.to(dtype).chunk(2,dim=-1)
    if shift:
        key=torch.cat((torch.zeros_like(key[:,:1]),key[:,:-1]),1)
        value=torch.cat((torch.zeros_like(value[:,:1]),value[:,:-1]),1)
    scores=(query.to(dtype).unsqueeze(2)*key).sum(-1)*(query.shape[-1]**-.5)
    return (scores.softmax(-1).unsqueeze(-1)*value).sum(2).to(query.dtype)


@triton.jit
def _forward(Q,KV,OUT,P,T:tl.constexpr,L:tl.constexpr,D:tl.constexpr,
             SHIFT:tl.constexpr,BL:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)
    valid=(row%T>0) if SHIFT else True
    src=row-1 if SHIFT else row
    layers=tl.arange(0,BL);cols=tl.arange(0,BD)
    offsets=(src*L+layers[:,None])*(2*D)+cols[None,:]
    mask=valid & (layers[:,None]<L) & (cols[None,:]<D)
    k=tl.load(KV+offsets,mask=mask,other=0).to(tl.float32)
    v=tl.load(KV+offsets+D,mask=mask,other=0).to(tl.float32)
    q=tl.load(Q+row*D+cols,mask=cols<D,other=0).to(tl.float32)
    scores=tl.sum(k*q[None,:],1)*(D**-.5)
    scores=tl.where(layers<L,scores,float('-inf'))
    p=tl.exp(scores-tl.max(scores,0));p=p/tl.sum(p,0)
    p=tl.where(valid,p,0.)
    out=tl.sum(v*p[:,None],0)
    tl.store(OUT+row*D+cols,out,mask=cols<D)
    tl.store(P+row*L+layers,p,mask=layers<L)


@triton.jit
def _backward(Q,KV,P,GO,GQ,GKV,T:tl.constexpr,L:tl.constexpr,D:tl.constexpr,
              SHIFT:tl.constexpr,BL:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)
    valid=(row%T>0) if SHIFT else True
    # A bijective row permutation covers every output gradient location once.
    src=tl.where(valid,row-1,row+T-1) if SHIFT else row
    layers=tl.arange(0,BL);cols=tl.arange(0,BD)
    offsets=(src*L+layers[:,None])*(2*D)+cols[None,:]
    shape_mask=(layers[:,None]<L) & (cols[None,:]<D)
    k=tl.load(KV+offsets,mask=valid & shape_mask,other=0).to(tl.float32)
    v=tl.load(KV+offsets+D,mask=valid & shape_mask,other=0).to(tl.float32)
    q=tl.load(Q+row*D+cols,mask=cols<D,other=0).to(tl.float32)
    go=tl.load(GO+row*D+cols,mask=cols<D,other=0).to(tl.float32)
    p=tl.load(P+row*L+layers,mask=layers<L,other=0)
    dp=tl.sum(v*go[None,:],1)
    ds=p*(dp-tl.sum(p*dp,0))*(D**-.5)
    dq=tl.sum(ds[:,None]*k,0)
    dk=ds[:,None]*q[None,:]
    dv=p[:,None]*go[None,:]
    tl.store(GQ+row*D+cols,dq,mask=cols<D)
    tl.store(GKV+offsets,dk,mask=shape_mask)
    tl.store(GKV+offsets+D,dv,mask=shape_mask)


class _ProjectedRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,query,kv,shift):
        b,t,d=query.shape;l=kv.shape[2]
        out=torch.empty_like(query)
        prob=torch.empty((b,t,l),device=query.device,dtype=torch.float32)
        bl=triton.next_power_of_2(l);bd=triton.next_power_of_2(d)
        warps=4 if bl*bd<=4096 else 8
        _forward[(b*t,)](query,kv,out,prob,t,l,d,shift,bl,bd,num_warps=warps)
        ctx.save_for_backward(query,kv,prob)
        ctx.options=(b,t,d,l,shift,bl,bd,warps)
        return out

    @staticmethod
    def backward(ctx,grad):
        query,kv,prob=ctx.saved_tensors
        b,t,d,l,shift,bl,bd,warps=ctx.options
        dq=torch.empty_like(query);dkv=torch.empty_like(kv)
        _backward[(b*t,)](query,kv,prob,grad.contiguous(),dq,dkv,t,l,d,shift,bl,bd,num_warps=warps)
        return dq,dkv,None


def projected_read(query,kv,shift=True,backend='triton'):
    if query.ndim!=3 or kv.ndim!=4 or kv.shape[:2]!=query.shape[:2] or kv.shape[-1]!=2*query.shape[-1]:
        raise ValueError('Expected query[B,T,D] and packed KV[B,T,L,2D]')
    if backend=='torch':return projected_read_reference(query,kv,shift)
    if backend!='triton' or not query.is_cuda or query.device!=kv.device:
        raise ValueError('Triton projected read requires CUDA tensors on the same device')
    if query.dtype not in (torch.float32,torch.bfloat16,torch.float16) or kv.dtype!=query.dtype:
        raise ValueError('Query and KV must have matching FP32/BF16/FP16 dtype')
    if min(query.shape)<1 or kv.shape[2]<1:
        raise ValueError('Projected read dimensions must be positive')
    if triton.next_power_of_2(kv.shape[2])*triton.next_power_of_2(query.shape[-1])>32768:
        raise ValueError('Projected read tile exceeds supported reduction size')
    return _ProjectedRead.apply(query.contiguous(),kv.contiguous(),shift)
