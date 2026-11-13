"""Correctness-first HCA reference: hard causal chunking, differentiable training.

Fixed and dynamic modes share pooling, RoPE, SWA and inference cache semantics.
Dynamic supports the legacy DMC reset STE or an output-confidence surrogate.
Each uses a global soft rate objective, with no boundary quota.
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

USE_FUSED_ROTARY = True
USE_FUSED_ROUTING = True


def work_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def boundary_step(content, position, config, noise=None):
    """Start a chunk BEFORE the current token; 1-alpha in DMC's notation.

    Returns a hard-forward STE gate and its continuous relaxation. Only the
    first token must start a chunk. Dynamic has no age rule, length cap, quota,
    lookahead or end-of-call flush. The current hidden state is causal.
    """
    if config.attention_mode == 'fixed_hca':
        hard = torch.full_like(content, float(position % config.hca_block_size == 0))
        return hard, hard
    if position == 0:
        first = torch.ones_like(content)
        return first, first
    soft = torch.sigmoid((content if noise is None else content + noise) / config.hca_temperature)
    hard = (soft >= .5).to(soft.dtype)
    if config.hca_dynamic_gradient == 'confidence':
        return hard, soft
    gate = hard + (soft - soft.detach())  # hard forward, biased STE backward
    return gate, soft


def compression_loss(soft_count, count, ratio):
    """DMC eq.11: ReLU(mean(1-alpha) - 1/R), globally before ReLU.

    Count includes the compulsory first entry. Same number of KV heads in
    every layer makes our shared-per-layer routing average identical to a
    head-weighted average. This is a soft target, not a rate guarantee.
    """
    return F.relu(soft_count / count - 1. / ratio)


def balanced_compression_loss(soft_count, hard_count, count, ratio):
    """H-Net's load-balancing objective; hard rate controls gradient direction.

    F is detached realized keep fraction, G is differentiable mean probability.
    The derivative changes sign at F=1/R, so both over/under-compression are
    corrected. It is not a quota and does not guarantee an exact achieved rate.
    R=1 reduces to an objective encouraging all boundaries.
    """
    f = (torch.as_tensor(hard_count, dtype=soft_count.dtype, device=soft_count.device)/count).detach()
    g = soft_count/count
    if ratio == 1:
        return 1-g
    return ratio/(ratio-1) * ((ratio-1)*f*g+(1-f)*(1-g))


class _Confidence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, selected):
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        output, = ctx.saved_tensors
        if USE_FUSED_ROUTING and output.is_cuda and output.dtype == torch.bfloat16:
            from nanochat.hca_routing import confidence_gradient
            return grad, confidence_gradient(grad, output)
        dtype = work_dtype(output)
        score_grad = (grad.to(dtype)*output.to(dtype)).sum((-1, -2))
        return grad, score_grad


def confidence_output(output, starts, soft):
    """H-Net-inspired confidence STE, adapted to the HCA attention branch.

    Forward multiplier is one; backward multiplier derivative is one. Unlike
    DMC reset/visibility STE, this surrogate never differentiates chunk mass
    resets or absent KV entries. This is not a reproduction of H-Net's EMA
    dechunker, and must be evaluated as a distinct estimator.
    """
    selected = torch.where(starts.detach() > .5, soft, 1-soft)
    return _Confidence.apply(output, selected)


def pool_step(k, v, score, previous, start):
    """Per-channel softmax accumulation, shared weights for separate K and V.

    The learned logits are bounded smoothly to [-10,10], keeping weights finite.
    This does not prevent the legacy STE reset derivative from amplifying them.
    Accumulation stays FP32 (FP64 for numerical oracles).
    """
    dtype = work_dtype(k)
    weight = torch.exp(10 * torch.tanh(score.to(dtype) / 10))
    mass, sk, sv = previous
    keep = (1 - start).unsqueeze(-1).unsqueeze(-1)
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
    pool: tuple = field(default_factory=tuple)
    pool_tile_k: torch.Tensor | None = None
    pool_tile_v: torch.Tensor | None = None
    pool_tile_score: torch.Tensor | None = None
    pool_tile_starts: torch.Tensor | None = None
    pool_carry: tuple = field(default_factory=tuple)
    local_k: torch.Tensor | None = None
    local_v: torch.Tensor | None = None
    summary_k: torch.Tensor | None = None
    summary_v: torch.Tensor | None = None
    open_k: torch.Tensor | None = None
    open_v: torch.Tensor | None = None
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
        counts, tails, entries = [], [], []
        tensor_bytes = 0
        for layer in self.layers:
            counts.append([len(s.ends) for s in layer])
            tails.append([s.age for s in layer])
            entries.append([len(s.ends) + int(s.position > 0) for s in layer])
            for s in layer:
                for t in (*s.pool, *s.pool_carry, s.pool_tile_k, s.pool_tile_v,
                          s.pool_tile_score, s.pool_tile_starts, s.local_k, s.local_v,
                          s.summary_k, s.summary_v, s.open_k, s.open_v):
                    if t is not None:
                        tensor_bytes += t.numel() * t.element_size()
                        st = t.untyped_storage()
                        storage[(str(t.device), st.data_ptr())] = st.nbytes()
        return dict(tokens=self.position, completed_chunks=counts, open_chunk_tokens=tails,
                    compressed_entries=entries,
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
            if config.hca_dynamic_gradient == 'confidence':
                # Content-dependent initial boundaries; balanced rate loss
                # subsequently learns the target. No hard quota or length cap.
                nn.init.zeros_(self.boundary.bias)
            else:
                nn.init.zeros_(self.boundary.weight)
                # DMC's mostly uncompressed initialization is retained.
                nn.init.constant_(self.boundary.bias, 2.)

    def rotated(self, x, cos_sin):
        if (USE_FUSED_ROTARY and self.config.hca_backend == 'triton' and x.is_cuda
                and x.dtype == torch.bfloat16
                and (self.boundary is None or self.config.hca_dynamic_gradient == 'confidence')):
            from nanochat.hca_rotary import rotary_norm
            return rotary_norm(x, *cos_sin)
        return norm(apply_rotary_emb(x, *cos_sin)) * 1.2

    def project(self, x, cos_sin):
        shape = (*x.shape[:2], self.n_head, self.head_dim)
        raw_k, v = self.c_k(x).view(shape), self.c_v(x).view(shape)
        q = self.rotated(self.c_q(x).view(shape), cos_sin)
        local_k = self.rotated(raw_k, cos_sin)
        # Pool/router projections use FP32 even with BF16 backbone activations.
        dtype = work_dtype(x)
        if (self.config.hca_backend == 'triton' and x.is_cuda and x.dtype == torch.bfloat16
                and (self.boundary is None or self.config.hca_dynamic_gradient == 'confidence')):
            from nanochat.hca_projection import score_linear
            score = score_linear(x, self.pool_score.weight).view(shape)
        else:
            score = F.linear(x.to(dtype), self.pool_score.weight.to(dtype)).view(shape)
        if self.boundary is None:
            content = x.new_zeros(x.shape[:2], dtype=dtype)
        elif (USE_FUSED_ROUTING and self.config.hca_backend == 'triton' and x.is_cuda
              and x.dtype == torch.bfloat16 and self.config.hca_dynamic_gradient == 'confidence'):
            from nanochat.hca_routing import router_linear
            content = router_linear(x, self.boundary.weight, self.boundary.bias)
        else:
            content = F.linear(x.to(dtype), self.boundary.weight.to(dtype), self.boundary.bias.to(dtype)).squeeze(-1)
        return q, raw_k, local_k, v, score, content

    def dense(self, q, raw_k, local_k, v, score, content, cos_sin, noise=None):
        batch, length, heads, dim = q.shape
        dtype = work_dtype(q)
        pool = tuple(torch.zeros((batch, heads, dim), dtype=dtype, device=q.device) for _ in range(3))
        keys, values, starts, soft_starts = [], [], [], []
        for t in range(length):
            start, soft = boundary_step(content[:, t], t, self.config, None if noise is None else noise[:, t])
            pool, pk, pv = pool_step(raw_k[:, t], v[:, t], score[:, t], pool, start)
            keys.append(pk); values.append(pv); starts.append(start); soft_starts.append(soft)
        pk = torch.stack(keys, 1).to(q.dtype)
        pv = torch.stack(values, 1).to(v.dtype)
        pk = self.rotated(pk, cos_sin)
        starts = torch.stack(starts, 1)
        soft_starts = torch.stack(soft_starts, 1)
        pos = torch.arange(length, device=q.device)
        local = (pos[:, None] >= pos) & (pos[:, None] - pos < self.config.hca_window)
        # Candidate j becomes immutable/visible at j+1 iff j+1 starts a chunk.
        # Each query also sees its current unfinished prefix (the diagonal).
        # Never expose a later version of this prefix to an earlier query.
        closed = torch.cat((starts[:, 1:], starts.new_zeros(batch, 1)), 1)
        memory = (pos[:, None] > pos).to(dtype)[None] * closed[:, None, :]
        memory = memory + torch.eye(length, dtype=dtype, device=q.device)[None]
        weights = torch.cat((local[None].expand(batch, -1, -1).to(dtype), memory), -1)
        eligible = torch.cat((local, pos[:, None] >= pos), -1)[None].expand(batch, -1, -1)
        y = weighted_attention(q, torch.cat((local_k, pk), 1), torch.cat((v, pv), 1), weights, eligible)
        if self.config.hca_dynamic_gradient == 'confidence':
            y = confidence_output(y, starts, soft_starts)
        return y, starts, soft_starts

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
                start, _ = boundary_step(content[b:b+1, t], state.position, self.config)
                begins = bool(start.item())
                if begins and state.open_k is not None:
                    state.summary_k = state.open_k.clone() if state.summary_k is None else torch.cat((state.summary_k, state.open_k))
                    state.summary_v = state.open_v.clone() if state.summary_v is None else torch.cat((state.summary_v, state.open_v))
                    state.ends.append(state.position - 1)
                if not state.pool:
                    state.pool = tuple(torch.zeros((1, heads, dim), dtype=dtype, device=q.device) for _ in range(3))
                fast_cache = (self.config.hca_backend == 'triton' and q.dtype == torch.bfloat16
                              and (self.boundary is None or self.config.hca_dynamic_gradient == 'confidence'))
                if fast_cache:
                    from nanochat.hca_triton import cached_pool
                    pk, pv = cached_pool(raw_k[b:b+1, t], v[b:b+1, t], score[b:b+1, t], start,
                                         state, self.config.hca_block_size if self.boundary is None else 0)
                else:
                    state.pool, pk, pv = pool_step(raw_k[b:b+1, t], v[b:b+1, t], score[b:b+1, t], state.pool, start)
                cos, sin = (r[:, t:t+1] for r in cos_sin)
                state.open_k = self.rotated(pk[:, None].to(q.dtype), (cos, sin))[0].clone()
                state.open_v = pv.to(v.dtype).clone()
                lk, lv = local_k[b, t:t+1], v[b, t:t+1]
                state.local_k = (lk.clone() if state.local_k is None else torch.cat((state.local_k, lk))[-self.config.hca_window:].clone())
                state.local_v = (lv.clone() if state.local_v is None else torch.cat((state.local_v, lv))[-self.config.hca_window:].clone())
                if fast_cache:
                    from nanochat.hca_triton import cached_attention
                    outputs.append(cached_attention(q[b:b+1, t:t+1], state.local_k, state.local_v,
                        state.summary_k, state.summary_v, state.open_k, state.open_v,
                        state.position, self.config.hca_window,
                        self.config.hca_block_size if self.boundary is None else 0))
                    state.age = 1 if begins else state.age + 1
                    state.position += 1
                    continue
                keys, vals = state.local_k, state.local_v
                if state.summary_k is not None:
                    keys = torch.cat((keys, state.summary_k))
                    vals = torch.cat((vals, state.summary_v))
                keys = torch.cat((keys, state.open_k))
                vals = torch.cat((vals, state.open_v))
                weights = q.new_ones((1, 1, keys.shape[0]), dtype=dtype)
                outputs.append(weighted_attention(q[b:b+1, t:t+1], keys[None], vals[None], weights))
                state.age = 1 if begins else state.age + 1
                state.position += 1
            results.append(torch.cat(outputs, 1))
        if self.layer_idx == cache.n_layers - 1:
            cache.advance(length)
        return torch.cat(results)

    def accelerated(self, q, raw_k, local_k, v, score, content, cos_sin, noise=None):
        # FP64 stays on the independent oracle path; ordinary FP32/BF16 use
        # explicit fused forward/backward kernels, including router gradients.
        if q.dtype == torch.float64:
            return self.dense(q, raw_k, local_k, v, score, content, cos_sin, noise)
        from nanochat.hca_triton import prefix_pool, attention
        if self.boundary is None:
            positions = torch.arange(q.shape[1], device=q.device)
            starts = (positions % self.config.hca_block_size == 0).to(score.dtype)[None].expand(q.shape[0], -1)
            soft = starts
            ratio = self.config.hca_block_size
        elif q.shape[1] == 1:
            # The first boundary is compulsory and independent of the router,
            # including its autograd connectivity (None, not a synthetic zero).
            starts = torch.ones_like(content)
            soft = starts
            ratio = 0
        else:
            logits = content if noise is None else content + noise
            relaxed = torch.sigmoid(logits / self.config.hca_temperature)
            soft = torch.cat((torch.ones_like(relaxed[:, :1]), relaxed[:, 1:]), 1)
            starts = (soft >= .5).to(soft.dtype)
            if self.config.hca_dynamic_gradient == 'dmc':
                starts = starts + (soft-soft.detach())
            ratio = 0
        pk, pv = prefix_pool(raw_k, v, score, starts, ratio)
        pk = self.rotated(pk, cos_sin)
        y = attention(q, local_k, v, pk, pv, starts, self.config.hca_window, ratio,
                      compact=self.config.hca_dynamic_gradient == 'confidence')
        if self.config.hca_dynamic_gradient == 'confidence':
            y = confidence_output(y, starts, soft)
        return y, starts, soft

    def forward_with_stats(self, x, cos_sin):
        if x.shape[1] == 0:
            raise ValueError('Empty HCA input')
        args = self.project(x, cos_sin)
        noise = None
        if self.training and self.boundary is not None and self.config.hca_noise:
            u = torch.rand_like(args[-1]).clamp(1e-6, 1 - 1e-6)
            noise = u.log() - torch.log1p(-u)  # difference of two Gumbels
        execute = self.accelerated if self.config.hca_backend == 'triton' else self.dense
        y, starts, soft = execute(*args, cos_sin, noise)
        stats = torch.stack((soft.sum(), starts.detach().sum(), soft.new_tensor(soft.numel())))
        return self.c_proj(y.reshape(*x.shape[:2], self.n_embd)), stats

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if ve is not None:
            raise ValueError('Initial HCA port supports vanilla without value embeddings')
        if x.shape[1] == 0:
            raise ValueError('Empty HCA input')
        if kv_cache is None:
            return self.forward_with_stats(x, cos_sin)[0]
        y = self.cached(*self.project(x, cos_sin), cos_sin, kv_cache)
        return self.c_proj(y.reshape(*x.shape[:2], self.n_embd))
