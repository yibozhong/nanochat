"""Fixed-length causal MHA port of FLA, with fused value-moment extensions.

BF16 Tensor Core dot operands; standard FP32 online-softmax/dot accumulation.
One forward launch produces mean + statistic without materializing A or phi(V).
Backward uses FLA's preprocess/dQ/dKV structure and analytic moment chain rule.
"""
import torch
import triton
from nanochat import fla_attention_kernels as base
from nanochat import moment_attention_kernels as moment
from nanochat import third_moment_attention_kernels as third

MODES={'none':0,'second_moment':1,'variance':2,'jensen_silu':3,'jensen_abs':4}


def launch_options(q):
    b,t,h,d=q.shape
    return dict(B=b,T=t,H=h,HQ=h,G=1,K=d,V=d,BT=128,BS=64,
                BK=max(16,triton.next_power_of_2(d)),BV=max(16,triton.next_power_of_2(d)),
                W=None,cu_seqlens=None,chunk_indices=None,g_cumsum=None,scale=d**-.5,num_warps=8)


class FusedMomentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mode):
        kw=launch_options(q);b,t,h,d=q.shape
        o=torch.empty_like(v);lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        s=torch.empty_like(v) if mode else torch.empty(0,device=q.device,dtype=q.dtype)
        r=torch.empty_like(s)
        grid=(triton.cdiv(t,128),b*h)
        if mode:
            moment.moment_fwd_kernel[grid](q=q,k=k,v=v,o=o,stat=s,raw=r,MODE=mode,lse=lse,sink_bias=None,**kw)
        else:
            base.parallel_attn_fwd_kernel[grid](q=q,k=k,v=v,o=o,lse=lse,sink_bias=None,**kw)
        ctx.save_for_backward(q,k,v,o,s,r,lse);ctx.mode=mode
        return o,s

    @staticmethod
    def backward(ctx,do,ds):
        q,k,v,o,s,r,lse=ctx.saved_tensors;mode=ctx.mode
        b,t,h,d=q.shape;kw=launch_options(q);grid=(triton.cdiv(t,128),b*h)
        do=do.contiguous()
        delta=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        dq,dk,dv=(torch.empty_like(q) for _ in range(3))
        if mode:
            # Two derivative streams exceed Hopper shared memory at stages=3.
            kw["num_stages"] = 1
            ds=ds.contiguous();dm=torch.empty_like(do);dz=torch.empty_like(do)
            moment.moment_preprocess[(b*t*h,)](o,r,s,do,ds,dm,dz,delta,d,triton.next_power_of_2(d),mode)
            extra=dict(dz=dz,MODE=mode);module=moment
            dq_kernel=module.moment_bwd_kernel_dq;dkv_kernel=module.moment_bwd_kernel_dkv
        else:
            dm=do;extra={}
            base.parallel_attn_bwd_kernel_preprocess[(b*t*h,)](o,do,delta,triton.next_power_of_2(d),d)
            dq_kernel=base.parallel_attn_bwd_kernel_dq;dkv_kernel=base.parallel_attn_bwd_kernel_dkv
        dq_kernel[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=dm,dq=dq,dg_cumsum=None,**extra,**kw)
        dkv_kernel[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=dm,dk=dk,dv=dv,dg_cumsum=None,**extra,**kw)
        return dq,dk,dv,None


class FusedThirdMomentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        kw=launch_options(q);kw.update(BT=64,num_stages=1)
        b,t,h,d=q.shape;grid=(triton.cdiv(t,64),b*h)
        o,s,c3,r2,r3=(torch.empty_like(v) for _ in range(5))
        lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        third.third_fwd_kernel[grid](q=q,k=k,v=v,o=o,stat=s,third=c3,raw=r2,raw3=r3,
            MODE=2,lse=lse,sink_bias=None,**kw)
        ctx.save_for_backward(q,k,v,o,s,r2,r3,lse)
        return o,s,c3

    @staticmethod
    def backward(ctx,do,ds,dt):
        q,k,v,o,s,r2,r3,lse=ctx.saved_tensors
        b,t,h,d=q.shape;kw=launch_options(q);kw.update(BT=64,num_stages=1)
        grid=(triton.cdiv(t,64),b*h)
        do,ds,dt=(x.contiguous() for x in (do,ds,dt))
        dm,d2,d3,dq,dk,dv=(torch.empty_like(v) for _ in range(6))
        delta=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        third.third_preprocess[(b*t*h,)](o,r2,r3,s,do,ds,dt,dm,d2,d3,delta,d,triton.next_power_of_2(d),t,h)
        extra=dict(dz=d2,dz3=d3,MODE=2)
        third.third_bwd_kernel_dq[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=dm,dq=dq,dg_cumsum=None,**extra,**kw)
        third.third_bwd_kernel_dkv[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=dm,dk=dk,dv=dv,dg_cumsum=None,**extra,**kw)
        return dq,dk,dv


def fused_attention(q,k,v,mode='none'):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape:
        raise ValueError('Initial port supports equal-length/equal-head causal MHA in BTHD layout')
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or not q.is_cuda:
        raise ValueError('The training kernel requires CUDA BF16 Q/K/V')
    if q.shape[-1] not in (32,64,128):raise ValueError('Validated head dimensions: 32/64/128')
    if mode=='variance_third':
        return FusedThirdMomentFunction.apply(q.contiguous(),k.contiguous(),v.contiguous())
    return FusedMomentFunction.apply(q.contiguous(),k.contiguous(),v.contiguous(),MODES[mode])
