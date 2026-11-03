"""Independent stream attention with a fixed total head/parameter budget.

QKV use a grouped GEMM; attention folds stream into batch. The corrected
independent-write model stores full D-by-D Parameters and uses disjoint
stream slices, preserving the original Muon geometry. The legacy model
stores rectangular Parameters separately. Linearity lets its mHC
merge weights to the attended features, concatenate, then execute all output
projections and their sum in one GEMM without materializing K full-width
branch outputs be merged as sum_s (K * pre_s) * O_s(attention_s(X_s)).
The fixed K compensates for retaining pre_s ~ 1/K after splitting the head
budget; with identical streams and uniform pre, ordinary MHA is recovered.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nanochat.gpt import Linear, apply_rotary_emb, norm


class StreamAttention(nn.Module):
    def __init__(self, width, heads, streams, layer_idx, *, dense_parameters=False):
        super().__init__()
        if heads % streams or width % heads:
            raise ValueError('Total heads must divide evenly across streams')
        self.width, self.heads, self.streams = width, heads, streams
        self.head_dim, self.stream_heads = width // heads, heads // streams
        self.stream_width, self.layer_idx = width // streams, layer_idx
        self.dense_parameters = dense_parameters
        if dense_parameters:
            # Disjoint slices are used by different streams, but autograd and
            # the unmodified optimizer see the original four D-by-D matrices.
            for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
                setattr(self, name, Linear(width, width, bias=False))
        else:
            self.c_q = nn.ModuleList([Linear(width, self.stream_width, bias=False) for _ in range(streams)])
            self.c_k = nn.ModuleList([Linear(width, self.stream_width, bias=False) for _ in range(streams)])
            self.c_v = nn.ModuleList([Linear(width, self.stream_width, bias=False) for _ in range(streams)])
            self.c_proj = nn.ModuleList([Linear(self.stream_width, width, bias=False) for _ in range(streams)])

    @torch.no_grad()
    def copy_from_dense(self, attention):
        """Partition the ordinary attention's rows/columns without extra draws."""
        if self.dense_parameters:
            for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
                getattr(self, name).weight.copy_(getattr(attention, name).weight)
            return
        r = self.stream_width
        for s in range(self.streams):
            for name in ('c_q', 'c_k', 'c_v'):
                getattr(self, name)[s].weight.copy_(getattr(attention, name).weight[s*r:(s+1)*r])
            self.c_proj[s].weight.copy_(attention.c_proj.weight[:, s*r:(s+1)*r])

    def qkv_weights(self):
        if self.dense_parameters:
            return torch.cat([getattr(self, name).weight.reshape(
                self.streams, self.stream_width, self.width) for name in ('c_q', 'c_k', 'c_v')], dim=1)
        return torch.stack([torch.cat([self.c_q[s].weight, self.c_k[s].weight,
            self.c_v[s].weight], dim=0) for s in range(self.streams)])

    def output_weights(self):
        if self.dense_parameters:
            return self.c_proj.weight.reshape(self.width, self.streams, self.stream_width).permute(1, 0, 2)
        return torch.stack([p.weight for p in self.c_proj])

    def merged_output_weight(self):
        if self.dense_parameters:
            return self.c_proj.weight
        return torch.cat([p.weight for p in self.c_proj], dim=1)

    def forward(self, x, pre, cos_sin, kv_cache=None, *, return_streams=False):
        b, t, d, streams = x.shape  # persistent NVIDIA layout
        # Normalize each D-dimensional stream independently, before QKV.
        h = norm(x.permute(3, 0, 1, 2)).reshape(streams, b*t, d)
        weights = self.qkv_weights().to(h.dtype)
        qkv = torch.bmm(h, weights.transpose(1, 2))
        q, k, v = [z.reshape(streams*b, t, self.stream_heads, self.head_dim)
                   for z in qkv.split(self.stream_width, dim=-1)]
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin)) * 1.2
        k = norm(apply_rotary_emb(k, cos, sin)) * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
            y = y.reshape(streams, b, t, self.stream_width).permute(1, 2, 0, 3)
        else:
            # Cache stores the same total H heads as ordinary attention. Packing
            # independent heads into H preserves isolation and avoids copying
            # or maintaining a second cache. Training uses the batch-fold path.
            def pack(z):
                return z.reshape(streams, b, t, self.stream_heads, self.head_dim).permute(
                    1, 2, 0, 3, 4).reshape(b, t, self.heads, self.head_dim)
            q, k, v = pack(q), pack(k), pack(v)
            kc, vc = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=(-1, 0))
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(t)
            y = y.reshape(b, t, streams, self.stream_width)
        if return_streams:
            # Keep each independent output projection in its own residual
            # stream. The next MLP connector performs the first aggregation.
            narrow = y.permute(2, 0, 1, 3).reshape(streams, b*t, self.stream_width)
            weights = self.output_weights().to(x.dtype)
            output = torch.bmm(narrow, weights.transpose(1, 2))
            return output.reshape(streams, b, t, d).permute(1, 2, 3, 0).contiguous()
        # Legacy merged-write experiment, retained for explicit old metadata.
        # pre is tokenwise and applied AFTER attention; it never mixes Q/K/V.
        # Keep the multiply in FP32 before returning to the activation dtype.
        weighted = (y.to(pre.dtype) * (self.streams * pre).unsqueeze(-1)).to(x.dtype).reshape(b, t, d)
        output_weight = self.merged_output_weight().to(x.dtype)
        return F.linear(weighted, output_weight)
