"""Full-attention vanilla/mHC on the existing nanochat attention and MLP.

This research model keeps native GPT and its SSSL/extras path unchanged.
The residual option matches the audited 58.72M full-attention vanilla baseline.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.gpt import Block, GPTConfig, Linear, norm
from nanochat.manifold_connections import ManifoldHyperConnection, ManifoldHyperHead, expand_streams, check_official_device
from nanochat.optim import MuonAdamW


@dataclass
class ConnectionConfig:
    variant: str = 'baseline'
    depth: int = 8
    width: int = 512
    heads: int = 4
    sequence_length: int = 2048
    vocab_size: int = 32768
    streams: int = 4
    sinkhorn_iterations: int = 20
    activation_checkpoint: bool = False

    def __post_init__(self):
        if self.variant not in ('baseline', 'mhc', 'mhc_no_manifold'):
            raise ValueError(f'Unknown connection variant: {self.variant}')
        if min(self.depth, self.width, self.heads, self.sequence_length, self.vocab_size, self.streams) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('RoPE requires an even integral head dimension')
        if self.sinkhorn_iterations < 1:
            raise ValueError('Sinkhorn needs at least one iteration')
        if self.variant != 'baseline' and (self.streams != 4 or self.width % 128):
            raise ValueError('Official mHC kernels require four streams and width divisible by 128')


class ConnectionBlock(Block):
    def __init__(self, native, index, config):
        super().__init__(native, index)
        self.attn.ve_gate = None
        self.variant = config.variant
        self.compiled_attention = self.compiled_mlp = None
        if self.variant != 'baseline':
            self.attn_connection = ManifoldHyperConnection(config.width, config.streams, 2*index,
                config.sinkhorn_iterations, manifold=config.variant == 'mhc')
            self.mlp_connection = ManifoldHyperConnection(config.width, config.streams, 2*index+1,
                config.sinkhorn_iterations, manifold=config.variant == 'mhc')

    def attention_branch(self, h, cos, sin, kv_cache=None):
        return self.attn(norm(h), None, (cos, sin), (-1, 0), kv_cache)

    def mlp_branch(self, h):
        return self.mlp(norm(h))

    def forward(self, x, cos, sin, kv_cache=None):
        if self.variant == 'baseline':
            return super().forward(x, None, (cos, sin), (-1, 0), kv_cache)
        h, skip, post = self.attn_connection.read(x)
        y = (self.compiled_attention or self.attention_branch)(h, cos, sin, kv_cache)
        x = self.attn_connection.write(y, skip, post)
        h, skip, post = self.mlp_connection.read(x)
        y = (self.compiled_mlp or self.mlp_branch)(h)
        return self.mlp_connection.write(y, skip, post)


class ConnectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        native = GPTConfig(sequence_len=config.sequence_length, vocab_size=config.vocab_size,
            n_layer=config.depth, n_head=config.heads, n_kv_head=config.heads,
            n_embd=config.width, window_pattern='L')
        self.wte = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([ConnectionBlock(native, i, config) for i in range(config.depth)])
        self.lm_head = Linear(config.width, config.vocab_size, bias=False)
        nn.init.normal_(self.wte.weight, std=.8)
        nn.init.normal_(self.lm_head.weight, std=.001)
        s = math.sqrt(3 / config.width)
        for b in self.blocks:
            for p in (b.attn.c_q.weight, b.attn.c_k.weight, b.attn.c_v.weight):
                nn.init.uniform_(p, -s, s)
            nn.init.zeros_(b.attn.c_proj.weight)
            nn.init.uniform_(b.mlp.c_fc.weight, -s*.4, s*.4)
            nn.init.zeros_(b.mlp.c_proj.weight)
        if config.variant != 'baseline':
            for b in self.blocks:
                b.attn_connection.reset_parameters(); b.mlp_connection.reset_parameters()
        self.stream_head = ManifoldHyperHead(config.width, config.streams) if config.variant != 'baseline' else None
        if self.stream_head is not None:
            self.stream_head.reset_parameters()
        d = config.width // config.heads
        inv = 1 / (100000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        freq = torch.outer(torch.arange(config.sequence_length, dtype=torch.float32), inv)
        self.register_buffer('cos', freq.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', freq.sin()[None, :, None, :], persistent=False)
        self.compiled_steps = None

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        if self.config.variant != 'baseline':
            check_official_device(device)
            if dtype != torch.bfloat16:
                raise ValueError('Official mHC kernels require BF16 activations')
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks and self.config.variant == 'baseline':
            self.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True) for b in self.blocks]
        elif compile_blocks:
            # TileKernels supplies its own compiled kernels and custom backward.
            # Compile only the unchanged attention/MLP branches with Inductor.
            for b in self.blocks:
                b.compiled_attention = torch.compile(b.attention_branch, dynamic=False, fullgraph=True)
                b.compiled_mlp = torch.compile(b.mlp_branch, dynamic=False, fullgraph=True)
        return self

    def hidden(self, idx, kv_cache=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + idx.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        x = norm(self.wte(idx))
        if self.config.variant != 'baseline':
            x = expand_streams(x, self.config.streams)
        cos, sin = self.cos[:, start:start+idx.shape[1]], self.sin[:, start:start+idx.shape[1]]
        for i, b in enumerate(self.blocks):
            fn = self.compiled_steps[i] if self.compiled_steps is not None else b.forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                x = checkpoint(fn, x, cos, sin, use_reentrant=False)
            else:
                x = fn(x, cos, sin, kv_cache)
        if self.stream_head is not None:
            x = self.stream_head(x)
        return norm(x)

    def forward(self, idx, targets=None, dense=False, kv_cache=None):
        h = self.hidden(idx, kv_cache)
        w = self.lm_head.weight.to(h.dtype)
        if targets is None or dense:
            logits = 15 * torch.tanh(F.linear(h, w).float() / 15)
            if targets is None:
                return logits
            return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(h, w, targets, softcap=15., shift=0,
            reduction='mean', filter_eps=None, filter_e_grad=False, filter_c_grad=False,
            accum_e_fp32=True, accum_c_fp32=True)


def optimizer_for(model, batch_tokens=32768):
    """Same backbone native LR rules; connector parameters get separate AdamW."""
    c = model.config
    batch_scale = math.sqrt(batch_tokens / 2**19)
    width_scale = (c.width / 768) ** -.5
    scaling_params = 12*c.depth*c.width*c.width + c.vocab_size*c.width
    ref_scaling_params = 12*12*768*768 + c.vocab_size*768
    wd = .28*batch_scale*(ref_scaling_params/scaling_params)
    groups = [
        dict(kind='adamw', params=[model.lm_head.weight], lr=.008*batch_scale*width_scale,
             betas=(.8, .96), eps=1e-10, weight_decay=.01),
        dict(kind='adamw', params=[model.wte.weight], lr=.3*batch_scale*width_scale,
             betas=(.8, .995), eps=1e-10, weight_decay=.001)]
    matrices, routing = [], []
    for b in model.blocks:
        for name, p in b.named_parameters():
            (routing if '_connection.' in name else matrices).append(p)
    if model.stream_head is not None:
        routing.extend(model.stream_head.parameters())
    if routing:
        groups.append(dict(kind='adamw', params=routing, lr=.008*batch_scale*width_scale,
                           betas=(.8, .96), eps=1e-10, weight_decay=0.))
    for shape in sorted({p.shape for p in matrices}):
        groups.append(dict(kind='muon', params=[p for p in matrices if p.shape == shape],
            lr=.02*batch_scale, momentum=.95, ns_steps=5, beta2=.9, weight_decay=wd))
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    opt = MuonAdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt, wd


def schedule(opt, step, total_steps, wd):
    down = round(.65*total_steps)
    if step < 40:
        lr_mult = (step+1)/40
    elif step <= total_steps-down:
        lr_mult = 1.
    else:
        progress = (total_steps-step)/down
        lr_mult = progress+(1-progress)*.05
    if step < 400:
        momentum = .85+(step/400)*(.97-.85)
    elif step >= total_steps-down:
        momentum = .97-(step-(total_steps-down))/down*.07
    else:
        momentum = .97
    for g in opt.param_groups:
        g['lr'] = g['initial_lr']*lr_mult
        if g['kind'] == 'muon':
            g['momentum'] = momentum
            g['weight_decay'] = wd*.5*(1+math.cos(math.pi*step/total_steps))
    return lr_mult
