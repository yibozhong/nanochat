"""Static RoPE-pair channel weights on normalized attention dot products.

One table per layer/head, shared by the two coordinates of each RoPE pair.
Applying the effective coefficient only to Q avoids redundant Q/K parameters.
The original normalized/1.2-scaled Q is multiplied in FP32, then cast to BF16;
whole-block Inductor compilation fuses the elementwise work. K/V are unchanged.
"""
import torch
from torch import nn
from nanochat.gpt import norm, apply_rotary_emb
from nanochat.flash_attention import flash_attn


def scale_query(q, scale):
    # nanochat RoPE pairs c with c+head_dim/2, not adjacent coordinates.
    paired = q.reshape(*q.shape[:-1], 2, q.shape[-1] // 2)
    work = paired if q.dtype == torch.float64 else paired.float()
    return (work * scale[:, None, :]).to(q.dtype).reshape_as(q)


class QKScaledAttention(nn.Module):
    def __init__(self, attention):
        super().__init__()
        for name in ('layer_idx', 'n_head', 'n_kv_head', 'n_embd', 'head_dim'):
            setattr(self, name, getattr(attention, name))
        assert self.n_head == self.n_kv_head and self.head_dim % 2 == 0
        for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
            setattr(self, name, getattr(attention, name))
        self.ve_gate = None
        self.qk_scale = nn.Parameter(torch.ones(self.n_head, self.head_dim // 2,
                                               device=self.c_q.weight.device))

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        assert ve is None and window_size == (-1, 0)
        b, t, c = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2
        q = scale_query(q, self.qk_scale)
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size)
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(t)
        return self.c_proj(y.contiguous().view(b, t, c))
