"""Fixed-length causal vanilla attention using the retained FLA Triton kernels.

BF16 dot operands and standard FP32 online-softmax accumulation. No statistic
branches; the kernel calls are the previous mode=none path unchanged.
"""
import torch
import triton
from torch import nn
from nanochat import fla_attention_kernels as base
from nanochat.gpt import apply_rotary_emb, norm


def launch_options(q):
    b,t,h,d=q.shape
    return dict(B=b,T=t,H=h,HQ=h,G=1,K=d,V=d,BT=128,BS=64,
                BK=max(16,triton.next_power_of_2(d)),BV=max(16,triton.next_power_of_2(d)),
                W=None,cu_seqlens=None,chunk_indices=None,g_cumsum=None,scale=d**-.5,num_warps=8)


class FusedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v):
        kw=launch_options(q);b,t,h,d=q.shape
        o=torch.empty_like(v);lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        grid=(triton.cdiv(t,128),b*h)
        base.parallel_attn_fwd_kernel[grid](q=q,k=k,v=v,o=o,lse=lse,sink_bias=None,**kw)
        ctx.save_for_backward(q,k,v,o,lse)
        return o

    @staticmethod
    def backward(ctx,do):
        q,k,v,o,lse=ctx.saved_tensors
        b,t,h,d=q.shape;kw=launch_options(q);grid=(triton.cdiv(t,128),b*h)
        do=do.contiguous()
        delta=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        dq,dk,dv=(torch.empty_like(q) for _ in range(3))
        base.parallel_attn_bwd_kernel_preprocess[(b*t*h,)](o,do,delta,triton.next_power_of_2(d),d)
        base.parallel_attn_bwd_kernel_dq[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=do,dq=dq,dg_cumsum=None,**kw)
        base.parallel_attn_bwd_kernel_dkv[grid](q=q,k=k,v=v,lse=lse,delta=delta,do=do,dk=dk,dv=dv,dg_cumsum=None,**kw)
        return dq,dk,dv


def fused_attention(q,k,v):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape:
        raise ValueError('Initial port supports equal-length/equal-head causal MHA in BTHD layout')
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or not q.is_cuda:
        raise ValueError('The training kernel requires CUDA BF16 Q/K/V')
    if q.shape[-1] not in (32,64,128):raise ValueError('Validated head dimensions: 32/64/128')
    return FusedAttentionFunction.apply(q.contiguous(),k.contiguous(),v.contiguous())


class FlaAttention(nn.Module):
    """Use the original Q/K/V/O parameters without adding parameters or RNG draws."""
    def __init__(self,original):
        super().__init__()
        for name in ('layer_idx', 'n_head', 'n_kv_head', 'n_embd', 'head_dim',
                     'c_q', 'c_k', 'c_v', 'c_proj', 've_gate_channels', 've_gate'):
            setattr(self,name,getattr(original,name))
        assert self.n_head==self.n_kv_head and self.ve_gate is None

    def forward(self,x,ve,cos_sin,window_size,kv_cache):
        assert ve is None and window_size==(-1,0)
        if kv_cache is not None:
            raise ValueError('Initial FLA backend is for full-sequence training/evaluation; KV decoding is not implemented')
        b,t,_=x.shape
        q=self.c_q(x).view(b,t,self.n_head,self.head_dim)
        k=self.c_k(x).view(b,t,self.n_head,self.head_dim)
        v=self.c_v(x).view(b,t,self.n_head,self.head_dim)
        cos,sin=cos_sin
        q=norm(apply_rotary_emb(q,cos,sin))*1.2
        k=norm(apply_rotary_emb(k,cos,sin))*1.2
        mean=fused_attention(q,k,v)
        return self.c_proj(mean.reshape(b,t,self.n_embd))
