"""Full-attention vanilla and NVIDIA mHC on nanochat attention and MLP.

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
from nanochat.nvidia_connections import NvidiaHyperConnection, NvidiaHyperHead, expand_streams, check_nvidia_device
from nanochat.optim import MuonAdamW

CONNECTION_VARIANTS = ('baseline', 'mhc', 'mhc_no_sinkhorn', 'mhc_channel_no_sinkhorn', 'attnres', 'scaled_hc', 'scaled_hc_all', 'skip_scale', 'attnres_scale')


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
    connection_backend: str = 'nvidia'
    fused_connection_grad: bool = False
    architecture: str = 'vanilla'
    window_pattern: str = 'SSSL'  # native only; vanilla is always full attention
    attention_kernel: str = 'original'
    compact_routing: bool = False
    attnres_block_size: int = 2  # Moonshot paper's S: Attention and MLP each count once
    attnres_block_size_unit: str = 'sublayers'
    latent_feedback: bool = False
    feedback_eval_passes: int = 4
    feedback_source: str = 'top'
    feedback_attention_backend: str = 'triton'

    def __post_init__(self):
        if self.feedback_source not in ('top','layers','embedding_qkv','per_layer_qkv') or self.feedback_attention_backend not in ('torch','triton'):
            raise ValueError('Unknown feedback source or attention backend')
        if self.feedback_source != 'top' and not self.latent_feedback:
            raise ValueError('Layer feedback requires latent_feedback')
        if type(self.feedback_eval_passes) is not int or self.feedback_eval_passes < 1:
            raise ValueError('Feedback evaluation passes must be a positive integer')
        if self.latent_feedback and (self.architecture != 'vanilla' or self.variant != 'baseline'
                                    or self.attention_kernel != 'original'):
            raise ValueError('Latent feedback uses the vanilla baseline and original attention')
        if self.variant in ('skip_scale','attnres_scale'):
            if self.architecture != 'vanilla' or self.streams != 1 or self.fused_connection_grad:
                raise ValueError('Skip scale uses vanilla, exactly one residual stream, and ordinary gradients')
        if self.attnres_block_size_unit != 'sublayers':
            raise ValueError('AttnRes block size uses sublayers; load legacy metadata with from_checkpoint')
        if type(self.attnres_block_size) is not int or self.attnres_block_size < 2 or self.attnres_block_size % 2:
            raise ValueError('AttnRes block size counts Attention/MLP sublayers and must be even and >=2; Full AttnRes (S=1) is not implemented')
        if self.variant in ('attnres','attnres_scale'):
            if self.width > 16384:
                raise ValueError('NVIDIA AttnRes supports hidden width at most16384')
            if self.architecture != 'vanilla' or self.fused_connection_grad:
                raise ValueError('Initial AttnRes port uses vanilla and its own NVIDIA autograd')
            if math.ceil(self.depth / self.attnres_block_layers) > 16:
                raise ValueError('NVIDIA AttnRes supports at most16 completed history entries')
        if self.compact_routing and self.variant != 'mhc_channel_no_sinkhorn':
            raise ValueError('Compact routing applies only to the8-coefficient channel connector')
        if self.attention_kernel not in ('original','fla_triton'):
            raise ValueError('Unknown attention kernel')
        if self.attention_kernel == 'fla_triton' and (self.architecture != 'vanilla' or self.variant != 'baseline'):
            raise ValueError('Initial FLA port supports vanilla baseline backbone only')
        if self.architecture not in ('vanilla', 'native'):
            raise ValueError(f'Unknown backbone architecture: {self.architecture}')
        if not self.window_pattern or any(c not in 'SL' for c in self.window_pattern):
            raise ValueError('window_pattern must contain S and/or L')
        if self.connection_backend not in ('nvidia', 'nvidia_precise'):
            raise ValueError('Only the full NVIDIA mHC implementation is supported')
        if self.variant not in CONNECTION_VARIANTS:
            raise ValueError(f'Unknown connection variant: {self.variant}')
        if self.variant == 'mhc_channel_no_sinkhorn' and (self.architecture != 'vanilla' or self.fused_connection_grad):
            raise ValueError('Local channel mixing initially supports vanilla and unfused gradient accumulation')
        if self.variant in ('scaled_hc','scaled_hc_all') and (self.architecture != 'vanilla' or self.fused_connection_grad):
            raise ValueError('Scaled HC supports vanilla and ordinary gradient accumulation')
        if self.sinkhorn_iterations != 20:
            raise ValueError('The NVIDIA composite API requires 20 Sinkhorn iterations')
        if min(self.depth, self.width, self.heads, self.sequence_length, self.vocab_size, self.streams) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('RoPE requires an even integral head dimension')
        if self.sinkhorn_iterations < 1:
            raise ValueError('Sinkhorn needs at least one iteration')
        if self.variant not in ('baseline', 'attnres', 'attnres_scale', 'skip_scale') and (self.streams != 4 or self.width % 128):
            raise ValueError('NVIDIA mHC kernels require four streams and width divisible by 128')

    @property
    def attnres_block_layers(self):
        """Complete Transformer layers per block, for NeMo's layer-boundary recurrence."""
        return self.attnres_block_size // 2

    @classmethod
    def from_checkpoint(cls, config):
        """Read pre-cleanup NVIDIA/vanilla metadata; reject retired model weights."""
        config = dict(config)
        if config.pop('vocab_gate', False):
            raise ValueError('Checkpoint belongs to the retired vocabulary gate experiment')
        for name in ('vocab_gate_alpha', 'vocab_gate_backend', 'vocab_gate_chunk'):
            config.pop(name, None)
        if config.pop('ffn_readout_width', 0):
            raise ValueError('Checkpoint belongs to the retired FFN readout experiment')
        config.pop('ffn_readout_backend', None)
        config.pop('ffn_readout_init', None)
        if config.pop('attention_mode', 'full') != 'full':
            raise ValueError('Checkpoint belongs to the retired HCA experiment')
        for name in list(config):
            if name.startswith('hca_'):
                config.pop(name)
        if config.pop('window_final_full', True) is not True:
            raise ValueError('Checkpoint belongs to the retired native frontloaded-attention experiment')
        if config.pop('attention_pattern', 'full') != 'full':
            raise ValueError('Checkpoint belongs to the retired dilated-attention experiment')
        for name in ('sparse_budget', 'sparse_core', 'sparse_sinks', 'sparse_backend'):
            config.pop(name, None)  # inert defaults stored by retained full-attention models
        config.pop('kalman_formula', None)  # retired route; other variants stored this inert default
        if config.pop('mlp_memory', 'none') != 'none':
            raise ValueError('Checkpoint belongs to the retired play-memory experiment')
        config.pop('play_radius_init', None)
        # Metadata written before the unit marker counted complete Transformer
        # layers. Convert once, without changing grouping or checkpoint tensors.
        unit = config.get('attnres_block_size_unit', 'transformer_layers')
        if unit == 'transformer_layers':
            size = config.get('attnres_block_size', 1)
            if type(size) is not int or size < 1:
                raise ValueError('Invalid legacy AttnRes block size')
            config['attnres_block_size'] = 2 * size
            config['attnres_block_size_unit'] = 'sublayers'
        if config.pop('attention_statistic', 'none') != 'none':
            raise ValueError('Checkpoint belongs to a retired attention-statistics experiment')
        config.pop('moment_query_chunk', None)
        if config.pop('swa_looping', False):
            raise ValueError('Checkpoint belongs to the retired SWA looping experiment')
        for name in ('swa_loops', 'swa_window'):
            config.pop(name, None)
        if config.pop('attention_looping', False):
            raise ValueError('Checkpoint belongs to the retired attention looping experiment')
        for name in ('attention_loops', 'attention_loop_window'):
            config.pop(name, None)
        if config.pop('attention_output_ns', False):
            raise ValueError('Checkpoint belongs to the retired attention output NS experiment')
        if config.get('variant', 'baseline') not in CONNECTION_VARIANTS:
            raise ValueError('Checkpoint belongs to a retired model variant')
        if config.pop('stream_mlp_kernel', 'legacy') != 'legacy':
            raise ValueError('Checkpoint belongs to the retired stream MLP experiment')
        # Legacy metadata was inert on the retained native/vanilla/mHC models.
        for name in ("chunk_size", "lookback_enabled", "lookback_update",
                     "recompute_chunk_size", "recompute_previous", "recompute_input",
                     "recompute_memory"):
            config.pop(name, None)
        projection = config.pop('projection_backend', 'official')
        if projection != 'official':
            raise ValueError('Projection-only checkpoints belong to a retired implementation')
        if config.get('variant', 'baseline') == 'baseline':
            config['connection_backend'] = 'nvidia'  # unused by the vanilla model
        return cls(**config)


class ConnectionBlock(Block):
    def __init__(self, native, index, config):
        super().__init__(native, index)
        self.attn.ve_gate = None
        self.variant = config.variant
        self.compiled_attention = self.compiled_mlp = None
        if self.variant == 'skip_scale':
            self.attn_skip_scale = nn.Parameter(torch.ones(config.width))
            self.mlp_skip_scale = nn.Parameter(torch.ones(config.width))
        elif self.variant != 'baseline':
            connection = NvidiaHyperConnection
            if self.variant == 'mhc_channel_no_sinkhorn':
                from nanochat.channel_connections import ChannelHyperConnection
                connection = ChannelHyperConnection
            if self.variant == 'scaled_hc':
                from nanochat.scaled_connections import ScaledHyperConnection
                connection = ScaledHyperConnection
            if self.variant == 'scaled_hc_all':
                from nanochat.scaled_all_connections import AllScaledHyperConnection
                connection = AllScaledHyperConnection
            kwargs = dict(use_tf32=config.connection_backend == 'nvidia', fused_grad_acc=config.fused_connection_grad,
                          sinkhorn=config.variant != 'mhc_no_sinkhorn')
            if config.variant == 'mhc_channel_no_sinkhorn':
                kwargs['compact_routing']=config.compact_routing
            self.attn_connection = connection(config.width, config.streams, 2*index, config.sinkhorn_iterations, **kwargs)
            self.mlp_connection = connection(config.width, config.streams, 2*index+1, config.sinkhorn_iterations, **kwargs)

    def attention_branch(self, h, cos, sin, kv_cache=None):
        return self.attn(norm(h), None, (cos, sin), (-1, 0), kv_cache)

    def mlp_branch(self, h):
        return self.mlp(norm(h))

    def forward(self, x, cos, sin, kv_cache=None):
        if self.variant == 'baseline':
            return super().forward(x, None, (cos, sin), (-1, 0), kv_cache)
        if self.variant == 'skip_scale':
            from nanochat.skip_scale import scaled_skip
            x = scaled_skip(x, self.attention_branch(x, cos, sin, kv_cache), self.attn_skip_scale)
            return scaled_skip(x, self.mlp_branch(x), self.mlp_skip_scale)
        h, skip, post = self.attn_connection.read(x)
        y = (self.compiled_attention or self.attention_branch)(h, cos, sin, kv_cache)
        x = self.attn_connection.write(y, skip, post)
        h, skip, post = self.mlp_connection.read(x)
        y = (self.compiled_mlp or self.mlp_branch)(h)
        return self.mlp_connection.write(y, skip, post)


class ConnectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.variant in ('attnres','attnres_scale'):
            raise ValueError('Use build_connection_model for AttnRes')
        if config.architecture != 'vanilla':
            raise ValueError('Use build_connection_model for native nanochat')
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
        if config.variant not in ('baseline', 'skip_scale'):
            for b in self.blocks:
                b.attn_connection.reset_parameters(); b.mlp_connection.reset_parameters()
        if config.variant in ('baseline', 'skip_scale'):
            self.stream_head = None
        else:
            head_class=NvidiaHyperHead
            if config.variant=='scaled_hc_all':
                from nanochat.scaled_all_connections import ScaledHyperHead
                head_class=ScaledHyperHead
            self.stream_head = head_class(config.width, config.streams, use_tf32=config.connection_backend == 'nvidia')
        if self.stream_head is not None:
            self.stream_head.reset_parameters()
        d = config.width // config.heads
        inv = 1 / (100000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        freq = torch.outer(torch.arange(config.sequence_length, dtype=torch.float32), inv)
        self.register_buffer('cos', freq.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', freq.sin()[None, :, None, :], persistent=False)
        self.compiled_steps = None
        if config.attention_kernel == 'fla_triton':
            from nanochat.fla_attention import FlaAttention
            for block in self.blocks:
                block.attn = FlaAttention(block.attn)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        if self.config.variant not in ('baseline', 'skip_scale'):
            check_nvidia_device(device)
            if dtype != torch.bfloat16:
                raise ValueError('NVIDIA mHC kernels require BF16 activations')
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks and self.config.variant in ('baseline', 'skip_scale'):
            self.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True) for b in self.blocks]
        elif compile_blocks:
            # NVIDIA TE supplies its own kernels and custom backward.
            # Compile only the unchanged attention/MLP branches with Inductor.
            for b in self.blocks:
                b.compiled_attention = torch.compile(b.attention_branch, dynamic=False, fullgraph=True)
                b.compiled_mlp = torch.compile(b.mlp_branch, dynamic=False, fullgraph=True)
        return self

    def hidden(self, idx, kv_cache=None):
        return norm(self.backbone(norm(self.wte(idx)), kv_cache))

    def backbone(self, x, kv_cache=None, return_layers=False, before_block=None):
        """Unnormalized top state, also used by detached latent feedback passes."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + x.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        if self.config.variant not in ('baseline', 'skip_scale'):
            x = expand_streams(x, self.config.streams)
        cos, sin = self.cos[:, start:start+x.shape[1]], self.sin[:, start:start+x.shape[1]]
        states = [] if return_layers else None
        for i, b in enumerate(self.blocks):
            if before_block is not None:
                x = before_block(i, x)
            # Cache positions mutate across calls and stay outside static
            # fullgraph training capture; attention keeps its GPU kernels.
            fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else b.forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                x = checkpoint(fn, x, cos, sin, use_reentrant=False)
            else:
                x = fn(x, cos, sin, kv_cache)
            if return_layers:
                states.append(x)
        if self.stream_head is not None:
            x = self.stream_head(x)
        return (x, torch.stack(states, dim=2)) if return_layers else x

    def forward(self, idx, targets=None, dense=False, kv_cache=None):
        h = self.hidden(idx, kv_cache)
        return self.readout(h, targets, dense)

    def readout(self, h, targets=None, dense=False):
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
    if model.config.architecture == 'native':
        return model.setup_optimizer(batch_tokens)
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
            (routing if '_connection.' in name or name in ('attn_skip_scale', 'mlp_skip_scale') else matrices).append(p)
    if model.stream_head is not None:
        routing.extend(model.stream_head.parameters())
    if c.variant in ('attnres','attnres_scale'):
        routing.extend(model.residual_parameters())
    if routing:
        groups.append(dict(kind='adamw', params=routing, lr=.008*batch_scale*width_scale,
                           betas=(.8, .96), eps=1e-10, weight_decay=0.))
    for shape in sorted({p.shape for p in matrices}):
        groups.append(dict(kind='muon', params=[p for p in matrices if p.shape == shape],
            lr=.02*batch_scale, momentum=.95, ns_steps=5, beta2=.9, weight_decay=wd))
    if c.latent_feedback:
        # Own group: single-pass updates must not advance momentum or decay W.
        feedback_params=list(model.feedback_parameters())
        groups.append(dict(kind='muon', params=feedback_params, conditional=True,
            lr=.02*batch_scale, momentum=.95, ns_steps=5, beta2=.9, weight_decay=wd))
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    if c.variant in ('mhc_channel_no_sinkhorn', 'scaled_hc', 'scaled_hc_all', 'attnres_scale'):
        # Like native.setup_optimizer: the unchanged compiled AdamW function
        # specializes once per routing shape/dtype. Local generators introduce
        # two extra tensor ranks; this is a compiler cache budget, not an LR or
        # optimizer change, and must be set before the first optimizer step.
        from torch._dynamo import config as compiler_config
        signatures = {(tuple(p.shape), p.dtype) for g in groups if g['kind'] == 'adamw' for p in g['params']}
        compiler_config.recompile_limit = max(compiler_config.recompile_limit, 2*len(signatures))
    opt = MuonAdamW(groups)
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt, wd


def build_connection_model(config):
    if config.feedback_source in ('embedding_qkv','per_layer_qkv'):
        from nanochat.projected_feedback import ProjectedFeedbackModel
        return ProjectedFeedbackModel(config)
    if config.latent_feedback:
        from nanochat.latent_feedback import LatentFeedbackModel
        return LatentFeedbackModel(config)
    if config.variant == 'attnres_scale':
        from nanochat.scaled_attention_residuals import ScaledAttnResModel
        return ScaledAttnResModel(config)
    if config.variant == 'attnres':
        from nanochat.attention_residuals import AttnResModel
        return AttnResModel(config)
    if config.architecture == 'native':
        from nanochat.native_connection_model import NativeConnectionModel
        return NativeConnectionModel(config)
    return ConnectionModel(config)


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
