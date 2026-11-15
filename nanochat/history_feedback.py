"""Embedding or contextual queries over strictly earlier previous-pass states.

Equal-length shifted Q/KV makes the training read ordinary causal FlashAttention.
Cached inference preserves the finite pass count, including parallel prefill.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.connection_model import ConnectionModel
from nanochat.engine import KVCache
from nanochat.gpt import Linear,apply_rotary_emb,norm
from nanochat.latent_feedback import LatentFeedbackModel


def history_read(q,k,v,backend='flash'):
    """[B,H,T,D]; lower-right causal alignment, with at least one key/query."""
    nq,nk=q.shape[-2],k.shape[-2]
    if backend=='flash':
        if nq==nk:
            return F.scaled_dot_product_attention(q,k,v,is_causal=True)
        if nq==1:
            return F.scaled_dot_product_attention(q,k,v,is_causal=False)
        from torch.nn.attention.bias import causal_lower_right
        return F.scaled_dot_product_attention(q,k,v,attn_mask=causal_lower_right(nq,nk))
    # Explicit reference, also usable for tiny float64 gradient oracles.
    dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
    scores=q.to(dtype)@k.to(dtype).transpose(-1,-2)*q.shape[-1]**-.5
    mask=torch.ones(nq,nk,device=q.device,dtype=torch.bool).tril(nk-nq)
    probabilities=scores.masked_fill(~mask,-float('inf')).softmax(-1)
    return (probabilities@v.to(dtype)).to(q.dtype)


class HistoryFeedbackCache:
    """One backbone cache per finite pass, plus projected history between passes."""
    def __init__(self,batch_size,num_heads,seq_len,head_dim,num_layers,device,dtype,passes):
        if type(passes) is not int or passes<1:raise ValueError('Cache passes must be positive')
        self.passes=passes;self.batch_size=batch_size;self.max_seq_len=seq_len
        self.backbones=[KVCache(batch_size,num_heads,seq_len,head_dim,num_layers,device,dtype) for _ in range(passes)]
        shape=(passes-1,batch_size,seq_len,num_heads,head_dim)
        self.history_k=torch.empty(shape,device=device,dtype=dtype)
        self.history_v=torch.empty_like(self.history_k)

    def get_pos(self):return self.backbones[0].get_pos()

    def reset(self):
        for cache in self.backbones:cache.reset()

    def prefill(self,other):
        if not isinstance(other,HistoryFeedbackCache) or other.passes!=self.passes:
            raise ValueError('Prefill needs the same history-feedback pass count')
        for dst,src in zip(self.backbones,other.backbones):dst.prefill(src)
        end=other.get_pos()
        self.history_k[:,:,:end].copy_(other.history_k[:,:,:end])
        self.history_v[:,:,:end].copy_(other.history_v[:,:,:end])


class HistoryFeedbackModel(LatentFeedbackModel):
    def __init__(self,config):
        # Reuse the pass loop/compile policy without constructing an unused fusion.
        ConnectionModel.__init__(self,config)
        self.history_projections=nn.ModuleDict({name:Linear(config.width,config.width,bias=False)
            for name in ('q','k','v','o')})
        for p in self.history_projections.parameters():nn.init.normal_(p,std=.02)
        self.compiled_mix=None

    def feedback_parameters(self):return self.history_projections.parameters()

    def project_query(self,query,cos,sin):
        b,t,d=query.shape
        q=self.history_projections['q'](query).view(b,t,self.config.heads,d//self.config.heads)
        return apply_rotary_emb(q,cos,sin)

    @staticmethod
    def normalize_history(top):
        return F.rms_norm(top.float(),(top.shape[-1],),eps=1e-5).to(top.dtype)

    def project_history(self,top,cos,sin,normalized=False):
        b,t,d=top.shape
        source=top if normalized else self.normalize_history(top)
        # Keep separate square Muon parameters; one packed KV GEMM.
        weight=torch.cat((self.history_projections['k'].weight,self.history_projections['v'].weight),0).to(top.dtype)
        kv=F.linear(source,weight).view(b,t,2,self.config.heads,d//self.config.heads)
        return apply_rotary_emb(kv[:,:,0],cos,sin),kv[:,:,1]

    def mix(self,embedding,top):
        t=embedding.shape[1]
        if t==1:return embedding
        contextual=self.config.feedback_query=='previous_top'
        source=self.normalize_history(top) if contextual else top
        query=source[:,1:] if contextual else embedding[:,1:]
        q=self.project_query(query,self.cos[:,1:t],self.sin[:,1:t])
        k,v=self.project_history(source[:,:-1],self.cos[:,:t-1],self.sin[:,:t-1],normalized=contextual)
        read=history_read(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),self.config.feedback_attention_backend)
        projected=self.history_projections['o'](read.transpose(1,2).flatten(2))
        return torch.cat((embedding[:,:1],embedding[:,1:]+projected),1)

    def make_kv_cache(self,batch_size,passes=None):
        c=self.config
        return HistoryFeedbackCache(batch_size,c.heads,c.sequence_length,c.width//c.heads,c.depth,
            self.wte.weight.device,self.wte.weight.dtype,c.feedback_eval_passes if passes is None else passes)

    def hidden(self,idx,kv_cache=None,passes=None):
        if kv_cache is None:return super().hidden(idx,passes=passes)
        if torch.is_grad_enabled():raise ValueError('Cached inference requires no_grad')
        if not isinstance(kv_cache,HistoryFeedbackCache):raise TypeError('Use HistoryFeedbackCache')
        if passes is not None and passes!=kv_cache.passes:raise ValueError('Cache pass count cannot change')
        if idx.ndim!=2 or idx.shape[1]<1 or idx.shape[0]!=kv_cache.batch_size:
            raise ValueError('Expected nonempty [cache batch,tokens] input')
        start=kv_cache.get_pos();end=start+idx.shape[1]
        if end>self.config.sequence_length or end>kv_cache.max_seq_len:raise ValueError('Context exceeded')
        if any(c.get_pos()!=start for c in kv_cache.backbones):raise ValueError('Pass caches are out of sync')
        embedding=norm(self.wte(idx));q=None;previous=None
        contextual=self.config.feedback_query=='previous_top'
        # Position zero has no source keys. Every other query has >=1 past key.
        skip=1 if start==0 else 0
        if end>1 and kv_cache.passes>1 and not contextual:
            q=self.project_query(embedding[:,skip:],self.cos[:,start+skip:end],self.sin[:,start+skip:end])
        for index,cache in enumerate(kv_cache.backbones):
            state=embedding
            if index and end>1 and contextual:
                q=self.project_query(previous[:,skip:],self.cos[:,start+skip:end],self.sin[:,start+skip:end])
            if index and q is not None:
                k=kv_cache.history_k[index-1,:,:end-1];v=kv_cache.history_v[index-1,:,:end-1]
                read=history_read(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),self.config.feedback_attention_backend)
                fed=self.history_projections['o'](read.transpose(1,2).flatten(2))
                state=torch.cat((embedding[:,:skip],embedding[:,skip:]+fed),1)
            top=self.backbone(state,cache)
            if index<kv_cache.passes-1:
                previous=self.normalize_history(top) if contextual else top
                k,v=self.project_history(previous,self.cos[:,start:end],self.sin[:,start:end],normalized=contextual)
                kv_cache.history_k[index,:,start:end].copy_(k)
                kv_cache.history_v[index,:,start:end].copy_(v)
        return norm(top)
