"""Value moments for vanilla attention; precision-first compiled reference.

The original mean uses the unchanged efficient attention backend. The added
branch recomputes the SAME mathematical causal probabilities in FP32, without
rounding squared values or moments to BF16 before centering. Query tiling and
activation recomputation bound temporary probability memory; they do not change
the attention support. No new custom kernels or attention parameters are used.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.gpt import Linear, apply_rotary_emb, norm, flash_attn


def moment_tile(q, k, v, query_start, centered, nonlinearity='square'):
    """BHSD inputs; FP64 remains FP64 for independent numerical checks."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    rows = torch.arange(q.shape[-2], device=q.device) + query_start
    cols = torch.arange(k.shape[-2], device=q.device)
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -.5)
    a = scores.masked_fill(cols[None, :] > rows[:, None], -torch.inf).softmax(-1)
    if nonlinearity == 'silu':
        # SiLU is not globally convex: signed gaps are intentional. Both terms
        # use the same phi, values and mathematical attention distribution.
        return a @ F.silu(v) - F.silu(a @ v)
    assert nonlinearity == 'square'
    second = a @ v.square()
    if not centered:
        return second
    mean = a @ v
    variance = second - mean.square()
    # Exact variance is nonnegative. FP32 cancellation may produce small
    # negatives; raw negatives are measured in the gate, not silently ignored.
    return variance.clamp_min(0)


def value_moments(q, k, v, *, centered, query_chunk=256, offset=0, recompute=True, nonlinearity='square'):
    """BTHD -> BTHD, including prefix-cache queries at absolute `offset`."""
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    pieces = []
    for start in range(0, q.shape[-2], query_chunk):
        args = (q[:, :, start:start+query_chunk], k, v, offset+start, centered, nonlinearity)
        if recompute and torch.is_grad_enabled():
            y = checkpoint(moment_tile, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            y = moment_tile(*args)
        pieces.append(y)
    return torch.cat(pieces, dim=2).transpose(1, 2)


class ZeroLinear(Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class MomentAttention(nn.Module):
    """Retain all existing Q/K/V/O Parameters, adding only a zero D x D W_s."""
    def __init__(self, original, mode, query_chunk, backend="original"):
        super().__init__()
        assert mode in ('second_moment', 'variance', 'jensen_silu') or (mode == 'none' and backend == 'fla_triton')
        for name in ('layer_idx', 'n_head', 'n_kv_head', 'n_embd', 'head_dim',
                     'c_q', 'c_k', 'c_v', 'c_proj', 've_gate_channels', 've_gate'):
            setattr(self, name, getattr(original, name))
        assert self.n_head == self.n_kv_head and self.ve_gate is None
        self.mode, self.query_chunk, self.backend = mode, query_chunk, backend
        # Construct without consuming RNG so ALL old initial weights/RNG match.
        self.c_stat = ZeroLinear(self.n_embd, self.n_embd, bias=False,
            device=self.c_proj.weight.device, dtype=self.c_proj.weight.dtype) if mode != 'none' else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        assert ve is None and window_size == (-1, 0)
        b, t, _ = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
        if self.backend == 'fla_triton':
            if kv_cache is not None:
                raise ValueError('Initial FLA backend is for full-sequence training/evaluation; KV decoding is not implemented')
            from nanochat.triton_moment_attention import fused_attention
            mean, stats = fused_attention(q, k, v, self.mode)
            output = self.c_proj(mean.reshape(b, t, self.n_embd))
            if self.c_stat is not None:
                output = output + self.c_stat(stats.reshape(b, t, self.n_embd))
            return output
        offset = 0
        if kv_cache is None:
            mean = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            offset = kv_cache.get_pos()
            kc, vc = kv_cache.get_layer_cache(self.layer_idx)
            mean = flash_attn.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size)
            k, v = kc[:, :offset+t], vc[:, :offset+t]
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(t)
        stats = value_moments(q, k, v, centered=self.mode == 'variance',
                             query_chunk=self.query_chunk, offset=offset,
                             nonlinearity='silu' if self.mode == 'jensen_silu' else 'square')
        mean = mean.contiguous().view(b, t, self.n_embd)
        stats = stats.contiguous().view(b, t, self.n_embd).to(mean.dtype)
        return self.c_proj(mean) + self.c_stat(stats)
