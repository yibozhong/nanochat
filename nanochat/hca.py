"""Correctness-first HCA reference: hard causal chunking, differentiable training.

Fixed and dynamic modes share pooling, RoPE, SWA and inference cache semantics.
Dynamic uses an EXACT causal budget per frame, not a soft ratio-loss promise.
This is a nanochat adaptation, not a reproduction of DeepSeek/DMC/H-Net.
See research/connections/hca/README.md for equations and estimator limitations.
"""
from dataclasses import dataclass, field
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import Linear, apply_rotary_emb, norm


def work_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def boundary_step(content, age, used, position, config, noise=None):
    """End *after* this token. All decisions use only the observed prefix.

    Reserve the final frame token for the last boundary. At most K-1 earlier
    boundaries, and force them if the remaining positions equal the quota.
    No ranking across future tokens, sequence-wide threshold, or end flush.
    """
    b, quota = config.hca_block_size, config.hca_budget_chunks
    frame = b * quota
    if config.attention_mode == 'fixed_hca':
        return torch.full_like(content, float((position + 1) % b == 0))
    # Age prior initializes noiseless routing to the fixed B-token partition.
    logits = content + (age.to(content.dtype) + 1 - b) / max(b / 4, 1)
    soft = torch.sigmoid((logits if noise is None else logits + noise) / config.hca_temperature)
    hard = (soft >= .5).to(soft.dtype)
    gate = hard + (soft - soft.detach())  # hard forward, biased STE backward
    offset = position % frame
    remaining = quota - used
    slots = frame - offset
    forced_end = (remaining == slots) | (offset == frame - 1)
    allowed = remaining > 1  # reserve one nonempty final chunk
    return torch.where(forced_end, torch.ones_like(gate),
                       torch.where(allowed, gate, torch.zeros_like(gate)))


def pool_step(k, v, score, previous, previous_end):
    """Per-channel softmax accumulation, shared weights for separate K and V.

    The learned logits are bounded smoothly to [-10,10]. This avoids overflow
    in the STE reset derivative even when the forward reset is exactly zero.
    Accumulation stays FP32 (FP64 for numerical oracles).
    """
    dtype = work_dtype(k)
    weight = torch.exp(10 * torch.tanh(score.to(dtype) / 10))
    mass, sk, sv = previous
    keep = (1 - previous_end).unsqueeze(-1).unsqueeze(-1)
    mass = keep * mass + weight
    sk = keep * sk + weight * k.to(dtype)
    sv = keep * sv + weight * v.to(dtype)
    return (mass, sk, sv), sk / mass, sv / mass


def weighted_attention(q, k, v, weights, eligible=None):
    """Exact masked softmax forward, finite derivative at a zero STE gate.

    q/k/v: [batch,time,heads,dim]; weights: [batch,query,key]. A zero
    weight removes the entry completely. Unlike log(gate), its derivative
    remains defined at zero. Every query must include at least its local KV.
    """
    dtype = work_dtype(q)
    scores = torch.einsum('bthd,bshd->bhts', q.to(dtype), k.to(dtype)) / math.sqrt(q.shape[-1])
    # Include causal candidate logits (including zero gates) in the shift.
    # Future values must not influence even the numerical stabilization.
    if eligible is not None:
        scores = scores.masked_fill(~eligible[:, None], -torch.inf)
    shifted = scores - scores.detach().amax(dim=-1, keepdim=True)
    numer = shifted.exp() * weights[:, None].to(dtype)
    probs = numer / numer.sum(dim=-1, keepdim=True)
    return torch.einsum('bhts,bshd->bthd', probs, v.to(dtype)).to(q.dtype)


@dataclass
class LayerCache:
    position: int = 0
    age: int = 0
    used: int = 0
    previous_end: float = 1.
    pool: tuple = field(default_factory=tuple)
    local_k: torch.Tensor | None = None
    local_v: torch.Tensor | None = None
    summary_k: torch.Tensor | None = None
    summary_v: torch.Tensor | None = None
    ends: list = field(default_factory=list)


class HCAKVCache:
    """Actually compressed cache: local window + completed chunks + 3 accumulators.

    Ragged per-example storage avoids padding the summary count to sequence
    length. Equal-length batch inputs; per-example chunk boundaries may differ.
    No autograd graph is retained. Partial chunks survive arbitrary API splits.
    """
    def __init__(self, config, batch_size=1):
        if config.attention_mode == 'full' or batch_size < 1:
            raise ValueError('HCA cache requires HCA config and positive batch size')
        self.config = copy.deepcopy(config)
        self.batch_size = batch_size
        self.n_layers = config.depth
        self.position = 0
        self.layers = [[LayerCache() for _ in range(batch_size)] for _ in range(self.n_layers)]

    def get_pos(self):
        return self.position

    def advance(self, count):
        self.position += count

    def reset(self):
        self.position = 0
        self.layers = [[LayerCache() for _ in range(self.batch_size)] for _ in range(self.n_layers)]

    def state_dict(self):
        from dataclasses import asdict
        return dict(config=asdict(self.config), batch_size=self.batch_size,
                    position=self.position, layers=copy.deepcopy(self.layers))

    def load_state_dict(self, state):
        from dataclasses import asdict
        if state['config'] != asdict(self.config) or state['batch_size'] != self.batch_size:
            raise ValueError('Cache configuration mismatch')
        if len(state['layers']) != self.n_layers or any(len(x) != self.batch_size for x in state['layers']):
            raise ValueError('Cache layer/batch mismatch')
        if any(s.position != state['position'] for layer in state['layers'] for s in layer):
            raise ValueError('Incomplete cache transaction')
        self.position = state['position']
        self.layers = copy.deepcopy(state['layers'])

    def prefill(self, other):
        if self.position or self.config != other.config or other.batch_size not in (1, self.batch_size):
            raise ValueError('Cannot broadcast this cache')
        self.position = other.position
        self.layers = [[copy.deepcopy(layer[0 if other.batch_size == 1 else b])
                        for b in range(self.batch_size)] for layer in other.layers]

    def statistics(self):
        storage = {}
        counts, tails = [], []
        tensor_bytes = 0
        for layer in self.layers:
            counts.append([len(s.ends) for s in layer])
            tails.append([s.age for s in layer])
            for s in layer:
                for t in (*s.pool, s.local_k, s.local_v, s.summary_k, s.summary_v):
                    if t is not None:
                        tensor_bytes += t.numel() * t.element_size()
                        st = t.untyped_storage()
                        storage[(str(t.device), st.data_ptr())] = st.nbytes()
        return dict(tokens=self.position, completed_chunks=counts, open_chunk_tokens=tails,
                    tensor_bytes=tensor_bytes, storage_bytes=sum(storage.values()),
                    local_window=self.config.hca_window,
                    note='Tensor storage only; excludes Python metadata, model and allocator overhead')


class HCAAttention(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.config = config
        self.layer_idx = original.layer_idx
        self.n_head, self.n_kv_head = original.n_head, original.n_kv_head
        self.head_dim, self.n_embd = original.head_dim, original.n_embd
        for name in ('c_q', 'c_k', 'c_v', 'c_proj'):
            setattr(self, name, getattr(original, name))
        self.ve_gate = None
        self.pool_score = Linear(config.width, config.width, bias=False)
        nn.init.zeros_(self.pool_score.weight)
        self.boundary = nn.Linear(config.width, 1) if config.attention_mode == 'dynamic_hca' else None
        if self.boundary is not None:
            nn.init.zeros_(self.boundary.weight)
            nn.init.zeros_(self.boundary.bias)

    def project(self, x, cos_sin):
        shape = (*x.shape[:2], self.n_head, self.head_dim)
        raw_k, v = self.c_k(x).view(shape), self.c_v(x).view(shape)
        q = norm(apply_rotary_emb(self.c_q(x).view(shape), *cos_sin)) * 1.2
        local_k = norm(apply_rotary_emb(raw_k, *cos_sin)) * 1.2
        # Pool/router projections use FP32 even with BF16 backbone activations.
        dtype = work_dtype(x)
        score = F.linear(x.to(dtype), self.pool_score.weight.to(dtype)).view(shape)
        content = (F.linear(x.to(dtype), self.boundary.weight.to(dtype), self.boundary.bias.to(dtype)).squeeze(-1)
                   if self.boundary is not None else x.new_zeros(x.shape[:2], dtype=dtype))
        return q, raw_k, local_k, v, score, content

    def dense(self, q, raw_k, local_k, v, score, content, cos_sin, noise=None):
        batch, length, heads, dim = q.shape
        dtype = work_dtype(q)
        age = torch.zeros(batch, dtype=torch.long, device=q.device)
        used = torch.zeros_like(age)
        end = torch.ones(batch, dtype=dtype, device=q.device)
        pool = tuple(torch.zeros((batch, heads, dim), dtype=dtype, device=q.device) for _ in range(3))
        keys, values, ends = [], [], []
        for t in range(length):
            pool, pk, pv = pool_step(raw_k[:, t], v[:, t], score[:, t], pool, end)
            end = boundary_step(content[:, t], age, used, t, self.config, None if noise is None else noise[:, t])
            hard = end.detach().bool()
            age = torch.where(hard, 0, age + 1)
            used = used + hard.long()
            if (t + 1) % (self.config.hca_block_size * self.config.hca_budget_chunks) == 0:
                used = torch.zeros_like(used)
            keys.append(pk); values.append(pv); ends.append(end)
        pk = torch.stack(keys, 1).to(q.dtype)
        pv = torch.stack(values, 1).to(v.dtype)
        pk = norm(apply_rotary_emb(pk, *cos_sin)) * 1.2
        ends = torch.stack(ends, 1)
        pos = torch.arange(length, device=q.device)
        local = (pos[:, None] >= pos) & (pos[:, None] - pos < self.config.hca_window)
        completed = (pos[:, None] > pos).to(dtype)[None] * ends[:, None, :]
        weights = torch.cat((local[None].expand(batch, -1, -1).to(dtype), completed), -1)
        eligible = torch.cat((local, pos[:, None] > pos), -1)[None].expand(batch, -1, -1)
        y = weighted_attention(q, torch.cat((local_k, pk), 1), torch.cat((v, pv), 1), weights, eligible)
        return y, ends

    def cached(self, q, raw_k, local_k, v, score, content, cos_sin, cache):
        if torch.is_grad_enabled() or self.training:
            raise ValueError('HCA cache is inference-only: use eval() and no_grad()')
        if not isinstance(cache, HCAKVCache) or cache.config != self.config or cache.batch_size != q.shape[0]:
            raise ValueError('Use a matching HCAKVCache for compressed attention')
        batch, length, heads, dim = q.shape
        dtype = work_dtype(q)
        results = []
        for b in range(batch):
            state = cache.layers[self.layer_idx][b]
            if state.position != cache.get_pos():
                raise ValueError('Layer cache position mismatch')
            outputs = []
            for t in range(length):
                if not state.pool:
                    state.pool = tuple(torch.zeros((1, heads, dim), dtype=dtype, device=q.device) for _ in range(3))
                previous_end = q.new_tensor([state.previous_end], dtype=dtype)
                state.pool, pk, pv = pool_step(raw_k[b:b+1, t], v[b:b+1, t], score[b:b+1, t], state.pool, previous_end)
                lk, lv = local_k[b, t:t+1], v[b, t:t+1]
                state.local_k = (lk.clone() if state.local_k is None else torch.cat((state.local_k, lk))[-self.config.hca_window:].clone())
                state.local_v = (lv.clone() if state.local_v is None else torch.cat((state.local_v, lv))[-self.config.hca_window:].clone())
                keys, vals = state.local_k, state.local_v
                if state.summary_k is not None:
                    keys = torch.cat((keys, state.summary_k))
                    vals = torch.cat((vals, state.summary_v))
                weights = q.new_ones((1, 1, keys.shape[0]), dtype=dtype)
                outputs.append(weighted_attention(q[b:b+1, t:t+1], keys[None], vals[None], weights))
                age = torch.tensor([state.age], device=q.device)
                used = torch.tensor([state.used], device=q.device)
                end = boundary_step(content[b:b+1, t], age, used, state.position, self.config)
                closed = bool(end.item())
                if closed:
                    cos, sin = (r[:, t:t+1] for r in cos_sin)
                    sk = (norm(apply_rotary_emb(pk[:, None].to(q.dtype), cos, sin)) * 1.2)[0]
                    sv = pv.to(v.dtype)
                    state.summary_k = sk.clone() if state.summary_k is None else torch.cat((state.summary_k, sk))
                    state.summary_v = sv.clone() if state.summary_v is None else torch.cat((state.summary_v, sv))
                    state.ends.append(state.position)
                state.previous_end = float(closed)
                state.age = 0 if closed else state.age + 1
                state.used += int(closed)
                state.position += 1
                if state.position % (self.config.hca_block_size * self.config.hca_budget_chunks) == 0:
                    state.used = 0
            results.append(torch.cat(outputs, 1))
        if self.layer_idx == cache.n_layers - 1:
            cache.advance(length)
        return torch.cat(results)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if ve is not None:
            raise ValueError('Initial HCA port supports vanilla without value embeddings')
        if x.shape[1] == 0:
            raise ValueError('Empty HCA input')
        args = self.project(x, cos_sin)
        if kv_cache is None:
            noise = None
            if self.training and self.boundary is not None and self.config.hca_noise:
                u = torch.rand_like(args[-1]).clamp(1e-6, 1 - 1e-6)
                noise = u.log() - torch.log1p(-u)  # difference of two Gumbels
            y, _ = self.dense(*args, cos_sin, noise)
        else:
            y = self.cached(*args, cos_sin, kv_cache)
        return self.c_proj(y.reshape(*x.shape[:2], self.n_embd))
