"""Fixed, frequency-balanced token routing for independent Q/K/V projections.

The attention/O/MLP mathematics and original 2D Muon updates are preserved.
Only the new standalone HashQKVModel imports this module; existing models do not.
DeepGEMM is loaded lazily on the worker. No learned routing or dropped tokens.
"""
import heapq
import math
from typing import NamedTuple

import numpy as np
import torch
import triton as tr
import triton.language as tl
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.flash_attention import flash_attn
from nanochat.gpt import apply_rotary_emb, norm


def balanced_lookup(frequencies, buckets=16):
    """Greedy LPT assignment; ties use token ID and then bucket ID.

    Statistics must come from training INPUT IDs only. The table is frozen and
    checkpointed. Balances estimated occurrence mass, not every future batch.
    """
    frequencies = np.asarray(frequencies, dtype=np.int64)
    if frequencies.ndim != 1 or np.any(frequencies < 0) or not 1 <= buckets <= len(frequencies):
        raise ValueError('Invalid training frequencies or bucket count')
    order = np.lexsort((np.arange(len(frequencies)), -frequencies))
    heap = [(0, 0, e) for e in range(buckets)]
    table = np.empty(len(frequencies), dtype=np.int64)
    for token in order:
        mass, count, expert = heapq.heappop(heap)
        table[token] = expert
        heapq.heappush(heap, (mass + int(frequencies[token]), count + 1, expert))
    return table


class RoutingPlan(NamedTuple):
    pack: torch.Tensor
    unpack: torch.Tensor
    layout: torch.Tensor
    sizes_cpu: torch.Tensor
    sizes_gpu: torch.Tensor


def make_plan(input_ids, lookup, buckets=16, device='cuda'):
    """Prepare once from CPU input IDs, before H2D; reused by every layer.

    Fixed worst-case 128-row padding avoids a different compiled shape for each
    batch. Unused tail belongs to the last group and is filled with zeros. No
    expert capacity limit and no token dropping. CPU planning is timed by the
    full-model harness, rather than hidden outside its step measurement.
    """
    ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    table = np.asarray(lookup, dtype=np.int64)
    if not len(ids) or ids.min() < 0 or ids.max() >= len(table):
        raise ValueError('Input IDs outside the frozen lookup vocabulary')
    assignment = table[ids]
    counts = np.bincount(assignment, minlength=buckets)
    sizes = ((counts + 127) // 128) * 128
    capacity = ((len(ids) + 127) // 128 + buckets - 1) * 128
    sizes[-1] += capacity - sizes.sum()
    order = np.argsort(assignment, kind='stable')
    pack = np.full(capacity, -1, dtype=np.int64)
    unpack = np.empty(len(ids), dtype=np.int64)
    start = source = 0
    for count, size in zip(counts, sizes):
        rows = order[source:source + count]
        pack[start:start + count] = rows
        unpack[rows] = np.arange(start, start + count)
        start += size
        source += count
    cpu_sizes = torch.from_numpy(sizes.astype(np.int64))
    return RoutingPlan(torch.from_numpy(pack).to(device), torch.from_numpy(unpack).to(device),
        torch.from_numpy(np.repeat(np.arange(buckets, dtype=np.int32), sizes)).to(device),
        cpu_sizes, cpu_sizes.to(device=device, dtype=torch.int32))


@tr.jit
def _gather(X, Index, Y, ROWS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = p // WIDTH, p % WIDTH
    source = tl.load(Index + row, row < ROWS, other=-1)
    value = tl.load(X + source * WIDTH + col, (row < ROWS) & (source >= 0), other=0)
    tl.store(Y + p, value, row < ROWS)


def _gather_impl(x, index):
    y = torch.empty((index.numel(), x.shape[1]), device=x.device, dtype=x.dtype)
    _gather[(tr.cdiv(y.numel(), 1024),)](x.contiguous(), index, y, y.shape[0], y.shape[1], 1024)
    return y


@torch.library.custom_op('hash_qkv::gather', mutates_args=())
def _gather_op(x: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    return _gather_impl(x, index)


@_gather_op.register_fake
def _gather_fake(x, index, inverse):
    return x.new_empty((index.numel(), x.shape[1]))


def _gather_setup(ctx, inputs, output):
    _, index, inverse = inputs
    ctx.save_for_backward(inverse, index)


def _gather_backward(ctx, dy):
    inverse, index = ctx.saved_tensors
    return _gather_op(dy, inverse, index), None, None


_gather_op.register_autograd(_gather_backward, setup_context=_gather_setup)


@torch.library.custom_op('hash_qkv::project', mutates_args=())
def _project(x: torch.Tensor, master: torch.Tensor, layout: torch.Tensor,
             sizes_cpu: torch.Tensor, sizes_gpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm as dg
    w = master.to(torch.bfloat16)
    y = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)
    dg.m_grouped_bf16_gemm_nt_contiguous(x.contiguous(), w, y, layout)
    return y, w


@_project.register_fake
def _project_fake(x, master, layout, sizes_cpu, sizes_gpu):
    return x.new_empty((x.shape[0], master.shape[1])), master.new_empty(master.shape, dtype=torch.bfloat16)


@torch.library.custom_op('hash_qkv::project_backward', mutates_args=())
def _project_backward_op(dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor,
                         layout: torch.Tensor, sizes_cpu: torch.Tensor,
                         sizes_gpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm as dg
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    dw = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
    dg.m_grouped_bf16_gemm_nn_contiguous(dy, w, dx, layout)
    dg.k_grouped_bf16_gemm_tn_contiguous(dy, x, dw, sizes_cpu.tolist(), sizes_gpu, dw)
    return dx, dw


@_project_backward_op.register_fake
def _project_backward_fake(dy, x, w, layout, sizes_cpu, sizes_gpu):
    return torch.empty_like(x), w.new_empty(w.shape, dtype=torch.float32)


def _project_setup(ctx, inputs, output):
    x, master, layout, sizes_cpu, sizes_gpu = inputs
    y, w = output
    ctx.save_for_backward(x, w, layout, sizes_cpu, sizes_gpu)
    ctx.mark_non_differentiable(w)


def _project_backward(ctx, dy, ignored):
    x, w, layout, sizes_cpu, sizes_gpu = ctx.saved_tensors
    dx, dw = _project_backward_op(dy, x, w, layout, sizes_cpu, sizes_gpu)
    return dx, dw, None, None, None


_project.register_autograd(_project_backward, setup_context=_project_setup)


class HashQKVProjection(nn.Module):
    def __init__(self, originals, buckets=16, seed=42, backend='deepgemm'):
        super().__init__()
        if backend not in ('deepgemm', 'reference'):
            raise ValueError(backend)
        self.backend, self.buckets = backend, buckets
        self.width = originals[0].shape[0]
        assert all(p.shape == (self.width, self.width) for p in originals)
        generator = torch.Generator(device=originals[0].device).manual_seed(seed)
        weights = torch.empty(buckets, 3, self.width, self.width, device=originals[0].device)
        weights.uniform_(-math.sqrt(3 / self.width), math.sqrt(3 / self.width), generator=generator)
        # Expert 0 matches the baseline, other experts have independent draws.
        for j, p in enumerate(originals):
            weights[0, j].copy_(p.detach())
        # Independent 2D parameters preserve the original Muon matrix geometry.
        self.weights = nn.ParameterList([nn.Parameter(p.clone()) for p in weights.flatten(0, 1)])

    def forward(self, x, plan):
        shape = x.shape
        flat = x.reshape(-1, self.width)
        packed = _gather_op(flat, plan.pack, plan.unpack)
        bank = torch.stack(list(self.weights)).reshape(self.buckets, 3 * self.width, self.width)
        if self.backend == 'deepgemm':
            y, _ = _project(packed, bank, plan.layout, plan.sizes_cpu, plan.sizes_gpu)
        else:
            # Independent ordinary-GEMM oracle, never used for the DG timing.
            parts = packed.split(plan.sizes_cpu.tolist())
            y = torch.cat([F.linear(part, bank[e].to(x.dtype)) for e, part in enumerate(parts)])
        return _gather_op(y, plan.unpack, plan.pack).reshape(*shape[:-1], 3 * self.width)


class HashQKVAttention(nn.Module):
    def __init__(self, original, buckets, seed, backend):
        super().__init__()
        assert original.n_head == original.n_kv_head and original.ve_gate is None
        for name in ('layer_idx', 'n_head', 'head_dim', 'n_embd', 'c_proj'):
            setattr(self, name, getattr(original, name))
        self.qkv = HashQKVProjection([original.c_q.weight, original.c_k.weight, original.c_v.weight],
                                     buckets, seed, backend)

    def forward(self, x, cos, sin, plan, kv_cache=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x, plan).reshape(b, t, 3, self.n_head, self.head_dim).unbind(2)
        q, k = norm(apply_rotary_emb(q, cos, sin)) * 1.2, norm(apply_rotary_emb(k, cos, sin)) * 1.2
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v.contiguous(), causal=True, window_size=(-1, 0))
        else:
            kc, vc = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(q, kc, vc, k=k, v=v.contiguous(),
                cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=(-1, 0))
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(t)
        return self.c_proj(y.contiguous().view(b, t, d))


class HashQKVBlock(nn.Module):
    def __init__(self, original, buckets, seed, backend):
        super().__init__()
        self.attn = HashQKVAttention(original.attn, buckets, seed, backend)
        self.mlp = original.mlp

    def forward(self, x, cos, sin, plan, kv_cache=None):
        x = x + self.attn(norm(x), cos, sin, plan, kv_cache)
        return x + self.mlp(norm(x))


class HashQKVModel(ConnectionModel):
    """Standalone vanilla model; the frozen table is part of state_dict.

    Call make_routing_plan on CPU input IDs in the loader and pass routing=plan.
    Omitting routing is a convenience path that copies GPU IDs back to CPU.
    """
    def __init__(self, config, lookup, buckets=16, seed=42, backend='deepgemm'):
        if config.variant != 'baseline' or config.architecture != 'vanilla' or config.attention_kernel != 'original':
            raise ValueError('Initial hash QKV supports the vanilla baseline backbone')
        table = torch.as_tensor(lookup, dtype=torch.int64).cpu()
        if table.shape != (config.vocab_size,) or table.min() < 0 or table.max() >= buckets:
            raise ValueError('Invalid frozen lookup table')
        super().__init__(config)
        self.register_buffer('hash_lookup', table.clone())
        self.register_buffer('hash_buckets', torch.tensor(buckets, dtype=torch.int64))
        self.buckets = buckets
        self.blocks = nn.ModuleList([HashQKVBlock(b, buckets, seed + 1009 * i, backend)
                                     for i, b in enumerate(self.blocks)])
        self._lookup_cpu = table.numpy().copy()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'hash_buckets' not in state_dict or int(state_dict['hash_buckets']) != self.buckets:
            raise ValueError('Checkpoint hash bucket geometry does not match')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._lookup_cpu = self.hash_lookup.detach().cpu().numpy().copy()
        return result

    def make_routing_plan(self, cpu_ids, device=None):
        if isinstance(cpu_ids, torch.Tensor):
            cpu_ids = cpu_ids.detach().cpu().numpy()
        return make_plan(cpu_ids, self._lookup_cpu, self.buckets, device or self.wte.weight.device)

    def hidden(self, idx, kv_cache=None, routing=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + idx.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        if routing is None:
            routing = self.make_routing_plan(idx)
        if routing.unpack.numel() != idx.numel():
            raise ValueError('Routing plan token count mismatch')
        x = norm(self.wte(idx))
        cos, sin = self.cos[:, start:start + idx.shape[1]], self.sin[:, start:start + idx.shape[1]]
        for i, block in enumerate(self.blocks):
            fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else block.forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                x = checkpoint(fn, x, cos, sin, routing, use_reentrant=False)
            else:
                x = fn(x, cos, sin, routing, kv_cache)
        return norm(x)

    def forward(self, idx, targets=None, dense=False, kv_cache=None, routing=None):
        h = self.hidden(idx, kv_cache, routing)
        w = self.lm_head.weight.to(h.dtype)
        if targets is None or dense:
            logits = 15 * torch.tanh(F.linear(h, w).float() / 15)
            return logits if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(h, w, targets, softcap=15., shift=0,
            reduction='mean', filter_eps=None, filter_e_grad=False, filter_c_grad=False,
            accum_e_fp32=True, accum_c_fp32=True)
