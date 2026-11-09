"""Fused causal sink/local/dilated attention; no extra parameters or QKV GEMM."""
import torch
from torch import nn
from torch.autograd.function import once_differentiable
import triton as tr
from nanochat.dilated_attention_kernels import sparse_query,sparse_delta,sparse_dkv
from nanochat.gpt import apply_rotary_emb,norm


def sparse_reference(q,k,v,stride,core,remote,sinks,query_start=0):
    """Dense mathematical reference and functional KV-cache fallback only."""
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    rows=torch.arange(query_start,query_start+q.shape[1],device=q.device)[:,None]
    cols=torch.arange(k.shape[1],device=q.device)[None,:]
    distance=rows-cols
    local=(distance>=0)&(distance<core)
    distant=(distance>=core)&(distance<=core+(remote-1)*stride)&((distance-core)%stride==0)
    mask=(cols<=rows)&((cols<sinks)|local|distant)
    scores=torch.matmul(q.to(dtype).transpose(1,2),k.to(dtype).transpose(1,2).transpose(-1,-2))*q.shape[-1]**-.5
    p=scores.masked_fill(~mask,float('-inf')).softmax(-1)
    return (p@v.to(dtype).transpose(1,2)).transpose(1,2).to(q.dtype)


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,stride,core,remote,sinks):
        b,t,h,d=q.shape;bm=32 if stride==1 else 16
        grid=(tr.cdiv(t,bm*stride),b*h,stride)
        o=torch.empty_like(q);lse=torch.empty((b,t,h),device=q.device,dtype=torch.float32)
        sparse_query[grid](q,k,v,o,lse,q,lse,q,t,h,d,core,remote,sinks,stride,bm,64,False,num_warps=4)
        ctx.save_for_backward(q,k,v,o,lse)
        ctx.geometry=(stride,core,remote,sinks)
        return o

    @staticmethod
    @once_differentiable
    def backward(ctx,go):
        q,k,v,o,lse=ctx.saved_tensors;go=go.contiguous()
        stride,core,remote,sinks=ctx.geometry;b,t,h,d=q.shape
        bm=32 if stride==1 else 16;grid=(tr.cdiv(t,bm*stride),b*h,stride)
        delta=torch.empty_like(lse);dq,dk,dv=(torch.empty_like(q) for _ in range(3))
        sparse_delta[(tr.cdiv(b*t*h,32),)](o,go,delta,b*t*h,d,32,num_warps=4)
        sparse_query[grid](q,k,v,o,lse,go,delta,dq,t,h,d,core,remote,sinks,stride,bm,64,True,num_warps=4)
        sparse_dkv[grid](q,k,v,go,lse,delta,dk,dv,t,h,d,core,remote,sinks,stride,bm,64,False,num_warps=4)
        # Each sink key is owned by one CTA per batch/head. No atomics or
        # duplicated gradients; non-sink CTAs exclude these output addresses.
        sparse_dkv[(1,b*h,1)](q,k,v,go,lse,delta,dk,dv,t,h,d,core,remote,sinks,stride,16,64,True,num_warps=4)
        return dq,dk,dv,None,None,None,None


def sparse_attention(q,k,v,stride,core,remote,sinks):
    if q.ndim!=4 or q.shape!=k.shape or q.shape!=v.shape or min(q.shape)<1:
        raise ValueError('Sparse training attention requires nonempty matching BTHD MHA tensors')
    if not q.is_cuda or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or k.device!=q.device or v.device!=q.device:
        raise ValueError('Sparse attention requires same-device CUDA BF16 tensors')
    if q.shape[-1] not in (32,64,128) or stride<1 or core<1 or remote<2 or not 1<=sinks<=16:
        raise ValueError('Unsupported sparse attention geometry')
    return _SparseAttention.apply(q.contiguous(),k.contiguous(),v.contiguous(),stride,core,remote,sinks)


class DilatedAttention(nn.Module):
    def __init__(self,original,stride,core,remote,sinks,backend='triton'):
        super().__init__()
        for name in ('layer_idx','n_head','n_kv_head','n_embd','head_dim','c_q','c_k','c_v','c_proj','ve_gate_channels','ve_gate'):
            setattr(self,name,getattr(original,name))
        assert self.n_head==self.n_kv_head and self.ve_gate is None
        self.stride=stride;self.core=core;self.remote=remote;self.sinks=sinks;self.backend=backend

    def forward(self,x,ve,cos_sin,window_size,kv_cache):
        assert ve is None and window_size==(-1,0)
        b,t,_=x.shape
        q=self.c_q(x).view(b,t,self.n_head,self.head_dim)
        k=self.c_k(x).view(b,t,self.n_head,self.head_dim)
        v=self.c_v(x).view(b,t,self.n_head,self.head_dim)
        cos,sin=cos_sin
        q=norm(apply_rotary_emb(q,cos,sin))*1.2
        k=norm(apply_rotary_emb(k,cos,sin))*1.2
        geometry=(self.stride,self.core,self.remote,self.sinks)
        if kv_cache is None:
            y=sparse_attention(q,k,v,*geometry) if self.backend=='triton' else sparse_reference(q,k,v,*geometry)
        else:
            if torch.is_grad_enabled():raise ValueError('KV-cache inference must disable gradients')
            start=kv_cache.get_pos();end=start+t
            kc,vc=kv_cache.get_layer_cache(self.layer_idx)
            kc[:,start:end].copy_(k);vc[:,start:end].copy_(v)
            y=sparse_reference(q,kc[:,:end],vc[:,:end],*geometry,query_start=start)
            # The final unmodified full-attention layer advances cache position.
        return self.c_proj(y.contiguous().view(b,t,self.n_embd))
