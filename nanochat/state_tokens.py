"""Token-local persistent depth memory, with shared cross-attention read/write.

There is no temporal attention here: B*T is the batch, and the sequence axis
contains only the current token's four slots (plus its new branch output).
"""
import math
from functools import partial
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.gpt import Linear, norm
from nanochat.connection_model import ConnectionModel


class SlotAttention(nn.Module):
    def __init__(self, width, heads, backend='auto'):
        super().__init__()
        self.heads = heads
        self.backend = backend
        self.q = Linear(width, width, bias=False)
        self.k = Linear(width, width, bias=False)
        self.v = Linear(width, width, bias=False)
        self.out = Linear(width, width, bias=False)

    def forward(self, query, memory):
        batch, time, nq, width = query.shape
        nk, hd = memory.shape[-2], width // self.heads
        q = self.q(query).reshape(batch*time, nq, self.heads, hd)
        k = self.k(memory).reshape(batch*time, nk, self.heads, hd)
        v = self.v(memory).reshape(batch*time, nk, self.heads, hd)
        from nanochat import flash_attention
        if self.backend == 'auto' and flash_attention.USE_FA3:
            y = flash_attention.flash_attn_func(q, k, v, causal=False)
        else:
            q, k, v = (a.transpose(1, 2) for a in (q, k, v))
            if self.backend == 'math':
                # Small independent reference/debug path. Keep FP64 for audits.
                dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
                logits = (q.to(dtype) @ k.to(dtype).transpose(-1, -2)) / math.sqrt(hd)
                y = (logits.softmax(-1) @ v.to(dtype)).to(v.dtype)
            else:
                # Do not use nanochat's causal SDPA fallback: these slots are
                # simultaneous states of ONE token, not positions in the text.
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
            y = y.transpose(1, 2)
        return self.out(y.reshape(batch, time, nq, width))


class StateTokenResidual(nn.Module):
    def __init__(self, width, heads, backend='auto'):
        super().__init__()
        self.width, self.heads = width, heads
        self.slots = nn.Parameter(torch.empty(4, width))
        self.read = SlotAttention(width, heads, backend)
        self.write = SlotAttention(width, heads, backend)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # A centered tetrahedron in every head: mean=0, RMS=1, cross dot=-1/3.
        # Copies of the same embedding alone would remain exactly symmetric.
        hd = self.width // self.heads
        basis = self.slots.new_tensor([[0, 1, 1, 1], [0, 1, -1, -1],
                                      [0, -1, 1, -1], [0, -1, -1, 1]])
        self.slots.copy_(basis.repeat(1, self.width // 4) * math.sqrt(4/3))
        for attention in (self.read, self.write):
            for projection in (attention.q, attention.k, attention.v, attention.out):
                nn.init.eye_(projection.weight)
        # Read initially averages memory, while retaining a full-rank Q map.
        self.read.q.weight.mul_(.01)
        # Initial self-vs-other-slot logit margin=10 for centered slots. The
        # identity V/O transports memory; random V/O would repeatedly rotate
        # and rescale it. This is weight initialization, not a runtime gate,
        # diagonal attention bias, hard mask, or extra residual connection.
        gain = math.sqrt(10 / ((4/3) * math.sqrt(hd)))
        self.write.q.weight.mul_(gain)
        self.write.k.weight.mul_(gain)

    def initialize(self, embedding):
        return embedding.unsqueeze(-2) + self.slots.to(embedding.dtype)

    def forward(self, h_new, state):
        query = h_new.unsqueeze(-2)
        h_next = h_new + self.read(query, state).squeeze(-2)
        # Both operations read the old state. Write receives h_new, not h_next.
        next_state = self.write(state, torch.cat((state, query), dim=-2))
        return h_next, next_state


class StateTokenModel(ConnectionModel):
    """Unchanged full-attention/ReLU² branches, independent across depth."""
    def __init__(self, config):
        # Construct the entire baseline first, preserving its initialization RNG.
        super().__init__(replace(config, variant='baseline', state_attention_backend='auto'))
        self.config = config
        self.state_residual = StateTokenResidual(config.width, config.heads,
                                                config.state_attention_backend)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks:
            from torch._dynamo import config as compiler_config
            compiler_config.recompile_limit = max(compiler_config.recompile_limit, 4*self.config.depth)
            self.compiled_steps = [torch.compile(partial(self.depth_step, i),
                dynamic=False, fullgraph=True) for i in range(self.config.depth)]
        return self

    def depth_step(self, index, h, state, cos, sin, kv_cache=None):
        block = self.blocks[index]
        h, state = self.state_residual(block.attention_branch(h, cos, sin, kv_cache), state)
        h, state = self.state_residual(block.mlp_branch(h), state)
        return h, state

    def hidden(self, idx, kv_cache=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + idx.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        h = norm(self.wte(idx))
        state = self.state_residual.initialize(h)
        cos, sin = self.cos[:, start:start+idx.shape[1]], self.sin[:, start:start+idx.shape[1]]
        for i in range(self.config.depth):
            fn = self.compiled_steps[i] if self.compiled_steps is not None else partial(self.depth_step, i)
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                h, state = checkpoint(fn, h, state, cos, sin, use_reentrant=False)
            else:
                h, state = fn(h, state, cos, sin, kv_cache)
        return norm(h)
