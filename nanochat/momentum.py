"""Heavy-Ball depth streams and nGPT-inspired normalized residual interpolation.

New normalized momentum uses tangent velocities, scalar RMS normalization and
short-arc parallel transport. The ambient recurrence remains explicit for old
checkpoints. Pure nGPT uses one hidden stream without allocating a velocity.
"""
from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import norm


def state_dtype(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def inverse_softplus(value):
    return value + math.log(-math.expm1(-value))


def normalize_residual(x, rms_floor=1e-6):
    """Normalize to RMS=1 outside a small-norm floor, smooth at zero.

    Clamp the squared norm BEFORE rsqrt so zero vectors have finite first and
    second derivatives. A larger force floor preserves zero output init.
    """
    x = state_dtype(x)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(rms_floor**2))


def ngpt_momentum_step(h, v, force, beta, gamma, eta, rms_floor):
    """Normalized target interpolation plus ambient-coordinate Heavy-Ball.

    beta=0 recovers scalar nGPT LERP with alpha=eta*gamma, provided the branch
    is above its norm floor. h is already normalized on entering each sublayer.
    Carry u itself: normalization of h is a state update, not a change of basis.
    """
    h, v = state_dtype(h), state_dtype(v)
    target = normalize_residual(force, rms_floor)
    u = beta*v + gamma*(target-h)
    return normalize_residual(h + eta*u), u


def tangent_project(h, x):
    """Project onto T_h S with radius sqrt(d); h must already have RMS=1."""
    return x - (h*x).mean(-1, keepdim=True)*h


def sphere_transport(h, new_h, v):
    """Exact short-arc parallel transport of tangent v, with O(d) work/storage.

    Our tangent retraction has dot(h,new_h)>=0, so the denominator is >=1;
    an antipodal fallback or denominator clipping is neither needed nor used.
    """
    scale = (v*new_h).mean(-1, keepdim=True) / (1 + (h*new_h).mean(-1, keepdim=True))
    return v - scale*(h+new_h)


def transported_momentum_step(h, v, force, beta, gamma, eta, force_floor, velocity_floor,
                              velocity_norm=None):
    """Retract with a tangent velocity and transport that same velocity to the new point."""
    h, v = state_dtype(h), state_dtype(v)
    target = normalize_residual(force, force_floor)
    mixed = beta*v + gamma*target
    if velocity_norm is None:
        speed = normalize_residual(tangent_project(h, mixed), velocity_floor)
    else:
        # Centering and channel gains can introduce a radial component.
        # Project last so tangency and the learned step magnitude both survive.
        speed = tangent_project(h, velocity_norm(mixed))
    new_h = normalize_residual(h + eta*speed)
    return new_h, sphere_transport(h, new_h, speed)


class HeavyBall(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.speed_floor = config.momentum_velocity_rms_floor if config.residual_mode == 'heavy_ball_rms' else None
        b = config.momentum_beta_init
        # Preserve the existing Heavy-Ball parameter names/shapes and optimizer.
        self.fixed_step = config.residual_mode == 'ngpt_hbformer'
        raw = [math.log(b / (1 - b)), inverse_softplus(config.momentum_gamma_init)]
        if not self.fixed_step:
            raw.append(inverse_softplus(config.momentum_eta_init))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def coefficients(self):
        eta = self.raw.new_ones(()) if self.fixed_step else F.softplus(self.raw[2])
        return self.raw[0].sigmoid(), F.softplus(self.raw[1]), eta

    def forward(self, h, v, force):
        beta, gamma, eta = self.coefficients()
        u = beta * state_dtype(v) + gamma * state_dtype(force)
        if self.speed_floor is not None:
            u = normalize_residual(u, self.speed_floor)
        return state_dtype(h) + eta * u, u


class NormalizedHeavyBall(HeavyBall):
    def __init__(self, config):
        super().__init__(config)
        self.rms_floor = config.ngpt_force_rms_floor
        self.velocity_mode = config.ngpt_velocity_mode
        self.velocity_floor = config.momentum_velocity_rms_floor
        self.velocity_norm = (nn.LayerNorm(config.width, eps=1e-5, bias=False)
                              if self.velocity_mode == 'transport_layernorm' else None)

    def forward(self, h, v, force):
        if self.velocity_mode in ('transport_rms', 'transport_layernorm'):
            return transported_momentum_step(h, v, force, *self.coefficients(),
                                             self.rms_floor, self.velocity_floor, self.velocity_norm)
        return ngpt_momentum_step(h, v, force, *self.coefficients(), self.rms_floor)


def residual_update(config, *, velocity_consumed=True, fixed_unit_mu=False):
    if config.chord_six_controls == 'small_dynamic':
        from nanochat.chord_token_controls import TokenControlledSite
        return TokenControlledSite(config, velocity_consumed=velocity_consumed)
    if config.ngpt_velocity_mode == 'lookahead_token_single':
        from nanochat.chord_single_step import SingleStepSite
        return SingleStepSite(config, velocity_consumed)
    if config.residual_mode == 'lookahead_only':
        from nanochat.lookahead import AdditiveLookaheadResidual
        return AdditiveLookaheadResidual(config)
    if config.residual_mode == 'ngpt_lookahead':
        from nanochat.lookahead import LookaheadResidual
        return LookaheadResidual(config, velocity_consumed=velocity_consumed, fixed_unit_mu=fixed_unit_mu,
                                 split_velocity=config.chord_six_controls == 'static')
    return {'heavy_ball': HeavyBall, 'heavy_ball_rms': HeavyBall, 'ngpt_heavy_ball': NormalizedHeavyBall,
            'ngpt_hbformer': NormalizedHeavyBall, 'ngpt': NormalizedResidual}[config.residual_mode](config)


class NormalizedResidual(nn.Module):
    """Scalar nGPT normalized interpolation, with no velocity state or parameters."""
    def __init__(self, config):
        super().__init__()
        self.initial_alpha = config.ngpt_alpha_init
        self.initial_raw = inverse_softplus(self.initial_alpha)
        size = config.width if config.channel_chord_coefficients else 1
        self.raw = nn.Parameter(torch.full((size,), self.initial_raw, dtype=torch.float32))
        self.signed_chord_steps = config.signed_chord_steps
        self.initial_slope = -math.expm1(-self.initial_alpha)
        self.rms_floor = config.ngpt_force_rms_floor

    def step_size(self, raw):
        value = raw.squeeze(0)
        positive = F.softplus(value)
        if self.signed_chord_steps:
            # Extend the softplus tangent below initialization. Both the value
            # and derivative match at the join, and the slope stays nonzero at zero.
            tangent = self.initial_alpha + self.initial_slope * (value - self.initial_raw)
            return torch.where(value >= self.initial_raw, positive, tangent)
        return positive

    def alpha(self):
        return self.step_size(self.raw)

    def forward(self, h, force):
        h = state_dtype(h)
        target = normalize_residual(force, self.rms_floor)
        return normalize_residual(h + self.alpha()*(target-h))


class NormalizedResidualModel(ConnectionModel):
    """Single-stream implementation reusing the existing backbone and KV cache."""
    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        super().prepare(device, dtype, compile_blocks=False)
        if compile_blocks:
            self.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True,
                                   options={'emulate_precision_casts': True}) for b in self.blocks]
        return self

    def hidden(self, idx, kv_cache=None):
        h = self.backbone(normalize_residual(self.wte(idx)), kv_cache)
        return norm(h.to(self.wte.weight.dtype))

    @torch.no_grad()
    def residual_metrics(self):
        values = torch.stack([m.alpha() for b in self.blocks
                              for m in (b.attn_residual, b.mlp_residual)]).cpu().tolist()
        return {f'residual/sublayer_{i}/alpha': a for i, a in enumerate(values)}


class MomentumModel(ConnectionModel):
    def __init__(self, config):
        super().__init__(config)
        # Allocate after the backbone to preserve seed-matched initialization.
        self.velocity_embedding = None
        self.velocity_position_embedding = None
        if config.velocity_init in ('token', 'token_position'):
            self.velocity_embedding = nn.Embedding(config.vocab_size, config.width)
            nn.init.normal_(self.velocity_embedding.weight, std=config.velocity_embedding_std)
        if config.velocity_init == 'token_position':
            self.velocity_position_embedding = nn.Embedding(config.sequence_length, config.width)
            nn.init.normal_(self.velocity_position_embedding.weight, std=config.velocity_embedding_std)
        if config.ngpt_weight_norm:
            self.normalize_backbone_weights()

    @torch.no_grad()
    def normalize_backbone_weights(self):
        """Project nGPT backbone vectors onto the unit sphere, including the untied head."""
        def project(weight, dim):
            lengths = torch.linalg.vector_norm(weight, dim=dim, keepdim=True, dtype=torch.float32)
            weight.div_(lengths.clamp_min_(1e-12))

        project(self.wte.weight, 1)
        project(self.lm_head.weight, 1)
        for b in self.blocks:
            for layer in (b.attn.c_q, b.attn.c_k, b.attn.c_v, b.mlp.c_fc):
                project(layer.weight, 1)
            for layer in (b.attn.c_proj, b.mlp.c_proj):
                project(layer.weight, 0)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False, *, compile_lookahead_branches=False):
        if compile_lookahead_branches and self.config.residual_mode != 'lookahead_only':
            raise ValueError('Separate lookahead branch compilation requires lookahead_only')
        super().prepare(device, dtype, compile_blocks=False)
        if self.velocity_embedding is not None:
            self.velocity_embedding.to(dtype=dtype)
        if self.velocity_position_embedding is not None:
            self.velocity_position_embedding.to(dtype=dtype)
        self.compiled_steps = None
        if compile_blocks:
            options = dict(dynamic=False, fullgraph=True, options={'emulate_precision_casts': True})
            if self.blocks[0].token_controls:
                for block in self.blocks:
                    block.compiled_attention = torch.compile(block.attn.forward, **options)
                    block.compiled_mlp = torch.compile(block.mlp.forward, **options)
            elif compile_lookahead_branches:
                from nanochat.lookahead import compile_additive_step
                self.compiled_steps = [compile_additive_step(b, options) for b in self.blocks]
            elif self.config.ngpt_velocity_mode == 'transport_layernorm' or self.config.residual_mode == 'lookahead_only':
                # Keep branch reads and BF16 attention/FFN backward at reference
                # precision. Compile only each site's residual and velocity update.
                self.compiled_steps = [partial(b.momentum_forward,
                    attn_update=torch.compile(b.attn_residual.forward, **options),
                    mlp_update=torch.compile(b.mlp_residual.forward, **options)) for b in self.blocks]
            else:
                self.compiled_steps = [torch.compile(b.momentum_forward, **options) for b in self.blocks]
        return self

    def hidden(self, idx, kv_cache=None):
        h = self.backbone(self.wte(idx), kv_cache, token_ids=idx)
        return norm(h.to(self.wte.weight.dtype))

    def initial_state(self, x, token_ids=None, start=0):
        h = normalize_residual(x) if self.config.residual_mode in ('ngpt_heavy_ball', 'ngpt_hbformer', 'ngpt_lookahead') else state_dtype(norm(x))
        if self.velocity_embedding is None:
            return h, torch.zeros_like(h)
        if token_ids is None:
            raise ValueError('Token IDs are required for velocity embeddings')
        v = state_dtype(self.velocity_embedding(token_ids))
        if self.velocity_position_embedding is not None:
            # The first sublayer projects the mixture. Keep HBFormer's raw v0 here.
            positions = self.velocity_position_embedding.weight[start:start+x.shape[1]]
            v = v + state_dtype(positions)
        elif self.config.residual_mode == 'ngpt_heavy_ball' and self.config.ngpt_velocity_mode == 'transport_rms':
            v = normalize_residual(tangent_project(h, v), self.config.momentum_velocity_rms_floor)
        elif self.config.residual_mode == 'heavy_ball_rms':
            v = normalize_residual(v, self.config.momentum_velocity_rms_floor)
        return h, v

    def backbone(self, x, kv_cache=None, return_layers=False, token_ids=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + x.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        h, v = self.initial_state(x, token_ids, start)
        state = (h, v)
        cos, sin = self.cos[:, start:start+x.shape[1]], self.sin[:, start:start+x.shape[1]]
        states = [] if return_layers else None
        for i, b in enumerate(self.blocks):
            fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else b.momentum_forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                state = checkpoint(fn, *state, cos, sin, use_reentrant=False)
            else:
                state = fn(*state, cos, sin, kv_cache)
            if return_layers:
                states.append(state[0])
        # Both streams are local to one token's depth traversal; no extra
        # persistent KV/cache or checkpoint state is added.
        h = state[0]
        return (h, torch.stack(states, dim=2)) if return_layers else h

    @torch.no_grad()
    def residual_metrics(self):
        values = torch.stack([torch.stack(m.coefficients()) for b in self.blocks
                              for m in (b.attn_residual, b.mlp_residual)]).cpu()
        result = {}
        for i, row in enumerate(values.tolist()):
            for name, value in zip(('beta', 'gamma', 'eta'), row):
                result[f'residual/sublayer_{i}/{name}'] = value
            if self.config.residual_mode == 'ngpt_heavy_ball' and self.config.ngpt_velocity_mode == 'ambient':
                result[f'residual/sublayer_{i}/alpha'] = row[1]*row[2]
        if self.config.ngpt_velocity_mode == 'transport_layernorm':
            gains = torch.stack([m.velocity_norm.weight for b in self.blocks
                                 for m in (b.attn_residual, b.mlp_residual)])
            stats = torch.stack((gains.mean(-1), gains.std(-1, unbiased=False)), dim=-1)
            for i, (mean, std) in enumerate(stats.tolist()):
                result[f'residual/sublayer_{i}/velocity_gain_mean'] = mean
                result[f'residual/sublayer_{i}/velocity_gain_std'] = std
        if self.velocity_embedding is not None:
            result['residual/velocity_embedding_rms'] = self.velocity_embedding.weight.float().square().mean().sqrt().item()
        if self.velocity_position_embedding is not None:
            result['residual/velocity_position_embedding_rms'] = self.velocity_position_embedding.weight.float().square().mean().sqrt().item()
        return result
