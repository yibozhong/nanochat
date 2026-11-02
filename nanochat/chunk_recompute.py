"""One retrospective rerun per boundary, reusing the original Transformer.

The state contains each layer's ordinary keys/values, not projections of a
final hidden. All replacements are functional and are available only after the
successor's original predictions have been computed. No new learned tensors.
"""
from dataclasses import dataclass, replace
import math

import torch
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
    # Ordinary per-layer cache: K/V of the block's INPUT, exactly as vanilla.
    return x,k,v


@dataclass(frozen=True)
class ChunkState:
    start: int
    length: int
    kv: tuple
    revisions: int = 0


class ChunkRecomputeModel(ConnectionModel):
    def __init__(self, config):
        super().__init__(replace(config,variant='baseline'))
        self.config=config
        self.compiled_chunk=None

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
        return self

    def process_chunk(self, embedding, start, history):
        """One complete pass, starting from original token embeddings."""
        b,t,d=embedding.shape;x=embedding
        cos,sin=self.cos[:,start:start+t],self.sin[:,start:start+t]
        memory_length=sum(chunk.length for chunk in history)
        mask=memory_causal_mask(t,memory_length,x.device)
        kv=[];step=self.compiled_chunk or transformer_chunk
        for layer,block in enumerate(self.blocks):
            if history:
                memory_k=torch.cat([chunk.kv[layer][0] for chunk in history],dim=1)
                memory_v=torch.cat([chunk.kv[layer][1] for chunk in history],dim=1)
            else:
                shape=(b,0,self.config.heads,d//self.config.heads)
                memory_k,memory_v=x.new_empty(shape),x.new_empty(shape)
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                x,k,v=checkpoint(step,block,x,memory_k,memory_v,cos,sin,mask,use_reentrant=False)
            else:
                x,k,v=step(block,x,memory_k,memory_v,cos,sin,mask)
            kv.append((k,v))
        return x,ChunkState(start,t,tuple(kv))

    def advance(self, embeddings, start, end, bank):
        original,current=self.process_chunk(embeddings[:,start:end],start,bank)
        output=norm(original)
        if bank and self.config.recompute_previous:
            previous=bank[-1]
            assert previous.revisions==0,'Each chunk may be revised only once'
            # Exclude the obsolete cache of the chunk being rerun. Preserve all
            # older revised chunks, and add the just-completed successor.
            history=bank[:-1]+(current,)
            _,revised=self.process_chunk(
                embeddings[:,previous.start:previous.start+previous.length],previous.start,history)
            bank=bank[:-1]+(replace(revised,revisions=1),)
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
