"""Two-pass hard segmented pooling; experimental, not selected by HCA yet.

Each 64-token tile first emits only small immutable carry summaries. A second
pass recomputes its local scan, avoiding full-size temporary writes/reads. It retains the original
serial tile carry's addition order, including arbitrarily long chunks. Only the
confidence estimator can use this path: hard starts have no reset derivative.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.autograd.function import once_differentiable
from nanochat.hca_triton import _affine


@tr.jit
def _parts_fwd(K,V,S,G,A,PM,PK,PV,
               T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NB:tl.constexpr,
               BT:tl.constexpr,BD:tl.constexpr):
    block,bh,dc=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=bh//H,bh%H
    rows=tl.arange(0,BT);d=dc*BD+tl.arange(0,BD);t=block*BT+rows
    off=((b*T+t[:,None])*H+h)*D+d[None,:]
    valid=(t[:,None]<T)&(d[None,:]<D)
    s=tl.load(S+off,valid,0).to(tl.float32)
    w=tl.exp(10*libdevice.tanh(s/10))
    k=tl.load(K+off,valid,0).to(tl.float32)
    v=tl.load(V+off,valid,0).to(tl.float32)
    a=1-tl.load(G+b*T+t,t<T,1)
    aa,m,sk,sv=tl.associative_scan((tl.broadcast_to(a[:,None],(BT,BD)),w,w*k,w*v),0,_affine)
    dst=((b*NB+block)*H+h)*D+d
    tl.store(PM+dst,tl.sum(tl.where(rows[:,None]==BT-1,m,0),0),d<D)
    tl.store(PK+dst,tl.sum(tl.where(rows[:,None]==BT-1,sk,0),0),d<D)
    tl.store(PV+dst,tl.sum(tl.where(rows[:,None]==BT-1,sv,0),0),d<D)
    if h==0 and dc==0:
        tl.store(A+b*NB+block,tl.min(a,0))


@tr.jit
def _finish_fwd(K,V,S,G,M,SK,SV,A,PM,PK,PV,OK,OV,
                T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NB:tl.constexpr,
                BT:tl.constexpr,BD:tl.constexpr,BC:tl.constexpr):
    block,bh,dc=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=bh//H,bh%H
    rows=tl.arange(0,BT);d=dc*BD+tl.arange(0,BD);t=block*BT+rows
    candidates=tl.arange(0,BC)
    flags=tl.load(A+b*NB+candidates,candidates<block,1)
    first=tl.maximum(0,tl.max(tl.where((candidates<block)&(flags==0),candidates,-1),0))
    cm=tl.full((BD,),0,tl.float32);ck=cm;cv=cm
    for previous in range(first,block):
        off=((b*NB+previous)*H+h)*D+d
        a=tl.load(A+b*NB+previous).to(tl.float32)
        cm=tl.load(PM+off,d<D,0)+a*cm
        ck=tl.load(PK+off,d<D,0)+a*ck
        cv=tl.load(PV+off,d<D,0)+a*cv
    off=((b*T+t[:,None])*H+h)*D+d[None,:]
    valid=(t[:,None]<T)&(d[None,:]<D)
    s=tl.load(S+off,valid,0).to(tl.float32)
    w=tl.exp(10*libdevice.tanh(s/10))
    k=tl.load(K+off,valid,0).to(tl.float32)
    v=tl.load(V+off,valid,0).to(tl.float32)
    a=1-tl.load(G+b*T+t,t<T,1)
    aa,m,sk,sv=tl.associative_scan((tl.broadcast_to(a[:,None],(BT,BD)),w,w*k,w*v),0,_affine)
    m+=aa*cm[None,:];sk+=aa*ck[None,:];sv+=aa*cv[None,:]
    tl.store(M+off,m,valid);tl.store(SK+off,sk,valid);tl.store(SV+off,sv,valid)
    tl.store(OK+off,sk/m,valid);tl.store(OV+off,sv/m,valid)


@tr.jit
def _parts_bwd(G,M,SK,SV,GK,GV,A,PM,PK,PV,
               T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NB:tl.constexpr,
               BT:tl.constexpr,BD:tl.constexpr):
    block,bh,dc=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=bh//H,bh%H
    rows=tl.arange(0,BT);d=dc*BD+tl.arange(0,BD);t=block*BT+rows
    off=((b*T+t[:,None])*H+h)*D+d[None,:]
    valid=(t[:,None]<T)&(d[None,:]<D)
    m=tl.load(M+off,valid,1)
    sk=tl.load(SK+off,valid,0);sv=tl.load(SV+off,valid,0)
    dk=tl.load(GK+off,valid,0).to(tl.float32)/m
    dv=tl.load(GV+off,valid,0).to(tl.float32)/m
    dm=-(dk*sk+dv*sv)/m
    a=1-tl.load(G+b*T+t+1,t+1<T,1)
    aa,dm,dk,dv=tl.associative_scan((tl.broadcast_to(a[:,None],(BT,BD)),dm,dk,dv),0,_affine,reverse=True)
    dst=((b*NB+block)*H+h)*D+d
    tl.store(PM+dst,tl.sum(tl.where(rows[:,None]==0,dm,0),0),d<D)
    tl.store(PK+dst,tl.sum(tl.where(rows[:,None]==0,dk,0),0),d<D)
    tl.store(PV+dst,tl.sum(tl.where(rows[:,None]==0,dv,0),0),d<D)
    if h==0 and dc==0:
        tl.store(A+b*NB+block,tl.min(a,0))


@tr.jit
def _finish_bwd(K,V,S,G,M,SK,SV,GK,GV,A,PM,PK,PV,OK,OV,OS,
                T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,NB:tl.constexpr,
                BT:tl.constexpr,BD:tl.constexpr,BC:tl.constexpr):
    block,bh,dc=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    b,h=bh//H,bh%H
    rows=tl.arange(0,BT);d=dc*BD+tl.arange(0,BD);t=block*BT+rows
    candidates=tl.arange(0,BC)
    eligible=(candidates>block)&(candidates<NB)
    flags=tl.load(A+b*NB+candidates,eligible,1)
    last=tl.minimum(NB-1,tl.min(tl.where(eligible&(flags==0),candidates,NB),0))
    cm=tl.full((BD,),0,tl.float32);ck=cm;cv=cm
    for distance in range(0,last-block):
        following=last-distance
        off=((b*NB+following)*H+h)*D+d
        a=tl.load(A+b*NB+following).to(tl.float32)
        cm=tl.load(PM+off,d<D,0)+a*cm
        ck=tl.load(PK+off,d<D,0)+a*ck
        cv=tl.load(PV+off,d<D,0)+a*cv
    off=((b*T+t[:,None])*H+h)*D+d[None,:]
    valid=(t[:,None]<T)&(d[None,:]<D)
    m=tl.load(M+off,valid,1)
    sk=tl.load(SK+off,valid,0);sv=tl.load(SV+off,valid,0)
    dk=tl.load(GK+off,valid,0).to(tl.float32)/m
    dv=tl.load(GV+off,valid,0).to(tl.float32)/m
    dm=-(dk*sk+dv*sv)/m
    a=1-tl.load(G+b*T+t+1,t+1<T,1)
    aa,dm,dk,dv=tl.associative_scan((tl.broadcast_to(a[:,None],(BT,BD)),dm,dk,dv),0,_affine,reverse=True)
    dm+=aa*cm[None,:];dk+=aa*ck[None,:];dv+=aa*cv[None,:]
    s=tl.load(S+off,valid,0).to(tl.float32)
    u=libdevice.tanh(s/10);w=tl.exp(10*u)
    k=tl.load(K+off,valid,0).to(tl.float32);v=tl.load(V+off,valid,0).to(tl.float32)
    tl.store(OK+off,dk*w,valid);tl.store(OV+off,dv*w,valid)
    tl.store(OS+off,w*(dm+dk*k+dv*v)*(1-u*u),valid)


class _HardPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx,k,v,score,starts):
        k,v,score,starts=[x.contiguous() for x in (k,v,score,starts)]
        b,t,h,d=k.shape;nb=tr.cdiv(t,64)
        states=[torch.empty_like(k,dtype=torch.float32) for _ in range(3)]
        parts=[torch.empty((b,nb,h,d),device=k.device,dtype=torch.float32) for _ in range(3)]
        flags=torch.empty((b,nb),device=k.device,dtype=torch.int32)
        pk,pv=torch.empty_like(k),torch.empty_like(v)
        grid=(nb,b*h,tr.cdiv(d,32));kw=dict(num_warps=4,enable_fp_fusion=False)
        _parts_fwd[grid](k,v,score,starts,flags,*parts,t,h,d,nb,64,32,**kw)
        _finish_fwd[grid](k,v,score,starts,*states,flags,*parts,pk,pv,t,h,d,nb,64,32,tr.next_power_of_2(nb),**kw)
        ctx.save_for_backward(k,v,score,starts,*states)
        return pk,pv

    @staticmethod
    @once_differentiable
    def backward(ctx,gk,gv):
        k,v,score,starts,m,sk,sv=ctx.saved_tensors
        b,t,h,d=k.shape;nb=tr.cdiv(t,64)
        parts=[torch.empty((b,nb,h,d),device=k.device,dtype=torch.float32) for _ in range(3)]
        flags=torch.empty((b,nb),device=k.device,dtype=torch.int32)
        dk,dv,ds=[torch.empty_like(x) for x in (k,v,score)]
        grid=(nb,b*h,tr.cdiv(d,32));kw=dict(num_warps=4,enable_fp_fusion=False)
        gk,gv=gk.contiguous(),gv.contiguous()
        _parts_bwd[grid](starts,m,sk,sv,gk,gv,flags,*parts,
                        t,h,d,nb,64,32,**kw)
        _finish_bwd[grid](k,v,score,starts,m,sk,sv,gk,gv,flags,*parts,dk,dv,ds,
                         t,h,d,nb,64,32,tr.next_power_of_2(nb),**kw)
        return dk,dv,ds,None


def hard_prefix_pool(k,v,score,starts):
    if starts.requires_grad:
        raise ValueError('Parallel hard pooling cannot compute DMC boundary gradients')
    if not k.is_cuda or k.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError('Parallel hard pooling requires CUDA FP32/BF16')
    if k.shape != v.shape or k.shape != score.shape or starts.shape != k.shape[:2]:
        raise ValueError('Parallel hard pooling shapes do not match')
    return _HardPool.apply(k,v,score,starts)
