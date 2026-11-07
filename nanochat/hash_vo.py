"""Fixed token-ID hash on source V and destination O; shared dense Q/K.

Uses the validated DeepGEMM routing and autograd primitives. Each token keeps
full causal attention, and each expert remains an independent 2D Muon matrix.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.connection_model import ConnectionModel
from nanochat.flash_attention import flash_attn
from nanochat.gpt import apply_rotary_emb, norm
from nanochat.hash_qkv import HashQKVModel, _gather_op, _project


class HashVOProjection(nn.Module):
    def __init__(self, original, buckets=16, seed=42, backend='deepgemm', zero_init=False):
        super().__init__()
        if backend not in ('deepgemm', 'reference'):
            raise ValueError(backend)
        self.backend, self.buckets = backend, buckets
        self.width = original.shape[0]
        assert original.shape == (self.width, self.width)
        weights = original.new_empty((buckets, self.width, self.width))
        if zero_init:
            weights.zero_()  # Original backbone initializes every attention O to zero.
        else:
            generator = torch.Generator(device=original.device).manual_seed(seed)
            weights.uniform_(-math.sqrt(3 / self.width), math.sqrt(3 / self.width), generator=generator)
            weights[0].copy_(original.detach())
        self.weights = nn.ParameterList([nn.Parameter(w.clone()) for w in weights])

    def forward(self, x, plan):
        shape = x.shape
        packed = _gather_op(x.reshape(-1, self.width), plan.pack, plan.unpack)
        bank = torch.stack(list(self.weights))
        if self.backend == 'deepgemm':
            y, _ = _project(packed, bank, plan.layout, plan.sizes_cpu, plan.sizes_gpu)
        else:
            parts = packed.split(plan.sizes_cpu.tolist())
            y = torch.cat([F.linear(part, bank[e].to(x.dtype)) for e, part in enumerate(parts)])
        return _gather_op(y, plan.unpack, plan.pack).reshape(shape)


class HashVOAttention(nn.Module):
    def __init__(self, original, buckets, seed, backend):
        super().__init__()
        assert original.n_head == original.n_kv_head and original.ve_gate is None
        for name in ('layer_idx', 'n_head', 'head_dim', 'n_embd', 'c_q', 'c_k'):
            setattr(self, name, getattr(original, name))
        self.v = HashVOProjection(original.c_v.weight, buckets, seed, backend)
        self.o = HashVOProjection(original.c_proj.weight, buckets, seed, backend, zero_init=True)

    def forward(self, x, cos, sin, plan, kv_cache=None):
        b, t, d = x.shape
        # Q/K remain separate 2D parameters while using a single dense GEMM.
        qk = F.linear(x, torch.cat((self.c_q.weight, self.c_k.weight)).to(x.dtype))
        q, k = qk.reshape(b, t, 2, self.n_head, self.head_dim).unbind(2)
        v = self.v(x, plan).reshape(b, t, self.n_head, self.head_dim)
        q, k = norm(apply_rotary_emb(q, cos, sin)) * 1.2, norm(apply_rotary_emb(k, cos, sin)) * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v.contiguous(), causal=True, window_size=(-1, 0))
        else:
            kc, vc = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, kc, vc, k=k, v=v.contiguous(),
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=(-1, 0))
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(t)
        # Source values were routed by their own input IDs. The mixed result is
        # routed by the current destination token's ID, never an attended ID.
        return self.o(y.contiguous().view(b, t, d), plan)


class HashVOBlock(nn.Module):
    def __init__(self, original, buckets, seed, backend):
        super().__init__()
        self.attn = HashVOAttention(original.attn, buckets, seed, backend)
        self.mlp = original.mlp

    def forward(self, x, cos, sin, plan, kv_cache=None):
        x = x + self.attn(norm(x), cos, sin, plan, kv_cache)
        return x + self.mlp(norm(x))


class HashVOModel(HashQKVModel):
    """Reuse vanilla hidden/readout lifecycle, with strict VO checkpoint identity."""
    def __init__(self, config, lookup, buckets=16, seed=42, backend='deepgemm'):
        if config.variant != 'baseline' or config.architecture != 'vanilla' or config.attention_kernel != 'original':
            raise ValueError('Hash VO supports the vanilla baseline backbone')
        table = torch.as_tensor(lookup, dtype=torch.int64).cpu()
        if table.shape != (config.vocab_size,) or table.min() < 0 or table.max() >= buckets:
            raise ValueError('Invalid frozen lookup table')
        ConnectionModel.__init__(self, config)
        self.register_buffer('hash_lookup', table.clone())
        self.register_buffer('hash_buckets', torch.tensor(buckets, dtype=torch.int64))
        self.register_buffer('hash_projection_kind', torch.tensor(2, dtype=torch.int64))
        self.buckets = buckets
        self.blocks = nn.ModuleList([HashVOBlock(b, buckets, seed + 1009 * i, backend)
                                     for i, b in enumerate(self.blocks)])
        self._lookup_cpu = table.numpy().copy()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'hash_projection_kind' not in state_dict or int(state_dict['hash_projection_kind']) != 2:
            raise ValueError('Only a VO-hash checkpoint can be loaded into HashVOModel')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
