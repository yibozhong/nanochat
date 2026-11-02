"""Chunk-sequential Transformer with once-only, next-chunk historical revision.

The only persistent cross-chunk cache is one shared K/V bank. Revisions are
functional replacements: earlier logits and earlier autograd inputs are never
mutated. A revision can be read only after its source chunk has finished.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import Linear, apply_rotary_emb, norm


def shared_memory_mask(length, history, device):
    """All available history plus ordinary causal attention inside the chunk."""
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(history+length, device=device)[None, :]-history
    return k <= q


def attend(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
        v.transpose(1, 2), attn_mask=mask, is_causal=False).transpose(1, 2).contiguous()


def chunk_block(block, x, memory_k, memory_v, cos, sin, mask):
    h = norm(x)
    b, t, d = h.shape
    heads, head_dim = block.attn.n_head, block.attn.head_dim
    q = block.attn.c_q(h).view(b, t, heads, head_dim)
    k = block.attn.c_k(h).view(b, t, heads, head_dim)
    v = block.attn.c_v(h).view(b, t, heads, head_dim)
    q = 1.2*norm(apply_rotary_emb(q, cos, sin))
    k = 1.2*norm(apply_rotary_emb(k, cos, sin))
    # One joint softmax over history and local context, not two added attentions.
    k = torch.cat((memory_k, k), dim=1)
    v = torch.cat((memory_v, v), dim=1)
    x = x+block.attn.c_proj(attend(q, k, v, mask).reshape(b, t, d))
    return x+block.mlp(norm(x))


class LookbackAttention(nn.Module):
    """One ordinary cross-attention residual, shared across chunk boundaries."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width//heads
        self.c_q = Linear(width, width, bias=False)
        self.c_k = Linear(width, width, bias=False)
        self.c_v = Linear(width, width, bias=False)
        self.c_proj = Linear(width, width, bias=False)
        for p in (self.c_q.weight, self.c_k.weight, self.c_v.weight):
            nn.init.uniform_(p, -math.sqrt(3/width), math.sqrt(3/width))
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, previous, current, previous_cos, previous_sin, cos, sin):
        b, t, d = previous.shape
        prev, new = norm(previous), norm(current)
        q = self.c_q(prev).view(b, t, self.heads, self.head_dim)
        k = self.c_k(new).view(b, current.shape[1], self.heads, self.head_dim)
        v = self.c_v(new).view(b, current.shape[1], self.heads, self.head_dim)
        q = 1.2*norm(apply_rotary_emb(q, previous_cos, previous_sin))
        k = 1.2*norm(apply_rotary_emb(k, cos, sin))
        return previous+self.c_proj(attend(q, k, v).reshape(b, t, d))


@dataclass(frozen=True)
class MemoryBank:
    keys: tuple = ()
    values: tuple = ()
    previous: object = None
    previous_start: int = 0
    positions: int = 0
    revisions: int = 0


class LookbackModel(ConnectionModel):
    def __init__(self, config):
        super().__init__(replace(config, variant='baseline'))
        self.config = config
        # All pre-existing backbone/head draws are identical to vanilla seed42.
        with torch.random.fork_rng(devices=[]):
            self.memory_k = Linear(config.width, config.width, bias=False)
            self.memory_v = Linear(config.width, config.width, bias=False)
            for p in (self.memory_k.weight, self.memory_v.weight):
                nn.init.uniform_(p, -math.sqrt(3/config.width), math.sqrt(3/config.width))
            self.lookback = LookbackAttention(config.width, config.heads) if config.lookback_enabled else None
        self.compiled_chunk = self.compiled_memory = self.compiled_lookback = None

    def memory_parameters(self):
        yield from self.memory_k.parameters()
        yield from self.memory_v.parameters()
        if self.lookback is not None:
            yield from self.lookback.parameters()

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks:
            # Prefix lengths and grad/no-grad are intentional specializations.
            import torch._dynamo.config as compiler_config
            count = math.ceil(self.config.sequence_length/self.config.chunk_size)
            compiler_config.recompile_limit = max(compiler_config.recompile_limit, 4*count*self.config.depth)
            self.compiled_chunk = torch.compile(chunk_block, dynamic=False, fullgraph=True)
            self.compiled_memory = torch.compile(self.project_memory, dynamic=False, fullgraph=True)
            if self.lookback is not None:
                self.compiled_lookback = torch.compile(self.lookback.forward, dynamic=False, fullgraph=True)
        return self

    def project_memory(self, h, cos, sin):
        b, t, _ = h.shape
        normalized = norm(h)
        shape = (b, t, self.config.heads, self.config.width//self.config.heads)
        k = self.memory_k(normalized).view(shape)
        k = 1.2*norm(apply_rotary_emb(k, cos, sin))
        return k, self.memory_v(normalized).view(shape)

    def revise_and_append(self, bank, current):
        """Called after all outputs of current chunk have been computed."""
        start = bank.positions; end = start+current.shape[1]
        cos, sin = self.cos[:, start:end], self.sin[:, start:end]
        project = self.compiled_memory or self.project_memory
        keys, values = bank.keys, bank.values
        revisions = bank.revisions
        if bank.previous is not None and self.lookback is not None:
            old_start = bank.previous_start
            old_end = old_start+bank.previous.shape[1]
            update = self.compiled_lookback or self.lookback.forward
            updated = update(bank.previous, current,
                self.cos[:, old_start:old_end], self.sin[:, old_start:old_end], cos, sin)
            old_k, old_v = project(updated, self.cos[:, old_start:old_end], self.sin[:, old_start:old_end])
            keys, values = keys[:-1]+(old_k,), values[:-1]+(old_v,)
            revisions += 1
        new_k, new_v = project(current, cos, sin)
        return MemoryBank(keys+(new_k,), values+(new_v,), current, start, end, revisions)

    def hidden_from_embeddings(self, embeddings, *, return_bank=False):
        b, total, d = embeddings.shape
        if total == 0:
            raise ValueError('Lookback requires at least one input token')
        if total > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        bank = MemoryBank(); outputs = []
        step = self.compiled_chunk or chunk_block
        for start in range(0, total, self.config.chunk_size):
            end = min(start+self.config.chunk_size, total)
            x = embeddings[:, start:end]
            if bank.keys:
                memory_k, memory_v = torch.cat(bank.keys, dim=1), torch.cat(bank.values, dim=1)
            else:
                shape = (b, 0, self.config.heads, d//self.config.heads)
                memory_k, memory_v = x.new_empty(shape), x.new_empty(shape)
            mask = shared_memory_mask(end-start, start, x.device)
            cos, sin = self.cos[:, start:end], self.sin[:, start:end]
            for block in self.blocks:
                if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                    x = checkpoint(step, block, x, memory_k, memory_v, cos, sin, mask, use_reentrant=False)
                else:
                    x = step(block, x, memory_k, memory_v, cos, sin, mask)
            # Original predictions are immutable. Only the separate bank is revised.
            outputs.append(norm(x))
            bank = self.revise_and_append(bank, x)
        result = torch.cat(outputs, dim=1)
        return (result, bank) if return_bank else result

    def hidden(self, idx, kv_cache=None):
        if kv_cache is not None:
            raise ValueError('Lookback uses its own sequence-local memory bank, not the standard KVCache')
        return self.hidden_from_embeddings(norm(self.wte(idx)))
