"""One retrospective rerun per boundary, reusing the original Transformer.

The preceding chunk's original final block output is the bottom-layer input
for its rerun. Completed chunks expose the LAST ATTENTION layer's existing KV.
Normal passes blend the immediate predecessor's local/deep KV as in LastKV.
Reruns read shared deep history, including the successor, plus causal self KV.
Replacements are functional and available only after successor predictions.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import apply_rotary_emb, norm


def memory_causal_mask(length, memory_length, device):
    # In a rerun memory also includes the completed successor. Its entire KV
    # is visible deliberately; only the local chunk uses a triangular mask.
    q=torch.arange(length,device=device)[:,None]
    k=torch.arange(memory_length+length,device=device)[None,:]-memory_length
    return k<=q


def transformer_chunk(block, x, memory_k, memory_v, cos, sin, mask):
    h=norm(x);b,t,d=h.shape
    shape=(b,t,block.attn.n_head,block.attn.head_dim)
    q=1.2*norm(apply_rotary_emb(block.attn.c_q(h).view(shape),cos,sin))
    k=1.2*norm(apply_rotary_emb(block.attn.c_k(h).view(shape),cos,sin))
    v=block.attn.c_v(h).view(shape)
    all_k=torch.cat((memory_k,k),dim=1)
    all_v=torch.cat((memory_v,v),dim=1)
    attended=F.scaled_dot_product_attention(q.transpose(1,2),all_k.transpose(1,2),
        all_v.transpose(1,2),attn_mask=mask,is_causal=False).transpose(1,2).contiguous()
    x=x+block.attn.c_proj(attended.reshape(b,t,d))
    x=x+block.mlp(norm(x))
    # These are the current layer's ORIGINAL attention KV, computed before its
    # attention/MLP residual updates. The last layer's pair is the deep cache.
    return x,k,v


def blend_previous(deep_k, deep_v, local_k, local_v, logits):
    # LastKV's prev_chunk_alphas: one learned sigmoid gate per layer/KV head,
    # shared by K and V, zero logits => equal local/deep contributions.
    alpha=logits.sigmoid().to(deep_k.dtype).view(1,1,-1,1)
    return alpha*deep_k+(1-alpha)*local_k,alpha*deep_v+(1-alpha)*local_v


@dataclass(frozen=True)
class ChunkState:
    start: int
    length: int
    kv: tuple
    local_kv: tuple | None = None
    final_hidden: torch.Tensor | None = None
    revisions: int = 0


class ChunkRecomputeModel(ConnectionModel):
    def __init__(self, config):
        super().__init__(replace(config,variant='baseline'))
        self.config=config
        # No new projections or RNG draws. Only these D-independent gate logits
        # are added; all original backbone/head parameters stay bitwise matched.
        self.prev_chunk_alphas=nn.Parameter(torch.zeros(config.depth,config.heads))
        self.compiled_chunk=self.compiled_blend=None

    def memory_parameters(self):
        yield self.prev_chunk_alphas

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        self.to(device);self.wte.to(dtype=dtype)
        self.cos=self.cos.to(dtype);self.sin=self.sin.to(dtype)
        if compile_blocks:
            import torch._dynamo.config as compiler_config
            count=math.ceil(self.config.sequence_length/self.config.recompute_chunk_size)
            limit=8*count*self.config.depth
            compiler_config.recompile_limit=max(compiler_config.recompile_limit,limit)
            compiler_config.accumulated_recompile_limit=max(compiler_config.accumulated_recompile_limit,2*limit)
            self.compiled_chunk=torch.compile(transformer_chunk,dynamic=False,fullgraph=True)
            self.compiled_blend=torch.compile(blend_previous,dynamic=False,fullgraph=True)
        return self

    def process_chunk(self, hidden, start, history):
        """Run the shared stack from supplied hidden, retaining original positions."""
        b,t,d=hidden.shape;x=hidden
        cos,sin=self.cos[:,start:start+t],self.sin[:,start:start+t]
        memory_length=sum(chunk.length for chunk in history)
        mask=memory_causal_mask(t,memory_length,x.device)
        # Only the immediate LEFT predecessor of a normal pass is blended.
        # During Lookback the last history entry is the future successor: it
        # must remain the same last-layer KV for every layer, never a blend.
        previous=history[-1] if history and history[-1].start+history[-1].length==start else None
        shared=history[:-1] if previous is not None else history
        if previous is not None:
            assert previous.local_kv is not None
        if shared:
            shared_k=torch.cat([chunk.kv[0] for chunk in shared],dim=1)
            shared_v=torch.cat([chunk.kv[1] for chunk in shared],dim=1)
        else:
            shape=(b,0,self.config.heads,d//self.config.heads)
            shared_k,shared_v=x.new_empty(shape),x.new_empty(shape)
        step=self.compiled_chunk or transformer_chunk
        mix=self.compiled_blend or blend_previous
        layers=[]
        for i,block in enumerate(self.blocks):
            if previous is not None:
                local_k,local_v=previous.local_kv[i]
                mixed_k,mixed_v=mix(*previous.kv,local_k,local_v,self.prev_chunk_alphas[i])
                memory_k=torch.cat((shared_k,mixed_k),dim=1)
                memory_v=torch.cat((shared_v,mixed_v),dim=1)
            else:
                memory_k,memory_v=shared_k,shared_v
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                x,k,v=checkpoint(step,block,x,memory_k,memory_v,cos,sin,mask,use_reentrant=False)
            else:
                x,k,v=step(block,x,memory_k,memory_v,cos,sin,mask)
            layers.append((k,v))
        return x,ChunkState(start,t,layers[-1],local_kv=tuple(layers),final_hidden=x)

    def advance(self, embeddings, start, end, bank):
        original,current=self.process_chunk(embeddings[:,start:end],start,bank)
        output=norm(original)
        if bank and self.config.recompute_previous:
            previous=bank[-1]
            assert previous.revisions==0,'Each chunk may be revised only once'
            assert previous.final_hidden is not None
            # Exclude the obsolete cache of the chunk being rerun. Preserve all
            # older revised chunks, and add the just-completed successor.
            history=bank[:-1]+(current,)
            # Continue from the ORIGINAL final block output, before the readout
            # RMSNorm. Preserve its graph and scale: no detach, norm or embedding
            # reinjection. Each chunk crosses this extra stack exactly once.
            _,revised=self.process_chunk(previous.final_hidden,previous.start,history)
            bank=bank[:-1]+(replace(revised,revisions=1,final_hidden=None,local_kv=None),)
        if not self.config.recompute_previous:
            current=replace(current,final_hidden=None)
            if bank:
                bank=bank[:-1]+(replace(bank[-1],local_kv=None,final_hidden=None),)
        return output,bank+(current,)

    def hidden_from_embeddings(self, embeddings, *, return_bank=False):
        total=embeddings.shape[1]
        if total<1 or total>self.config.sequence_length:
            raise ValueError('Input length must be positive and fit configured context')
        bank=();outputs=[]
        for start in range(0,total,self.config.recompute_chunk_size):
            end=min(total,start+self.config.recompute_chunk_size)
            out,bank=self.advance(embeddings,start,end,bank)
            outputs.append(out)
        result=torch.cat(outputs,dim=1)
        return (result,bank) if return_bank else result

    def hidden(self, idx, kv_cache=None):
        if kv_cache is not None:
            raise ValueError('Chunk recomputation owns a sequence-local versioned KV bank; standard KVCache is unsupported')
        return self.hidden_from_embeddings(norm(self.wte(idx)))
