"""Heavy-ball depth stream on the radius-sqrt(d) sphere, with parallel transport.

No learned matrices, velocity normalization, stop-gradients, or token recurrence.
FP64 is preserved for independent numerical checks; production state is FP32.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import norm


def geometry_dtype(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def sphere_project(x):
    """Radial projection to sqrt(d); nonzero inputs are required geometrically."""
    x = geometry_dtype(x)
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(1e-20))


def tangent_project(h, u):
    # Actual norm handles FP32 sphere roundoff. On the manifold h.h = d.
    return u - h * ((h * u).sum(-1, keepdim=True)
                    / h.square().sum(-1, keepdim=True).clamp_min(1e-20))


def parallel_transport(h, h_new, u):
    """Exact shortest-geodesic transport of u in T_h, using O(d) arithmetic.

    Retraction of a tangent step has h.h_new > 0, so the denominator is at
    least d (there is no antipodal singularity on this execution path).
    Transport the unscaled velocity, NOT the chord h_new-h or eta*u.
    """
    denominator = h.shape[-1] + (h * h_new).sum(-1, keepdim=True)
    coefficient = (u * h_new).sum(-1, keepdim=True) / denominator
    return u - coefficient * (h + h_new)


def momentum_step(h, v, force, beta, gamma, eta):
    h, v, force = geometry_dtype(h), geometry_dtype(v), geometry_dtype(force)
    # P_h(beta*v + gamma*F) = beta*v + gamma*P_h(F) for tangent v.
    # The combined projection also removes accumulated floating-point radial
    # error, without silently normalizing or changing the velocity's scale.
    u = tangent_project(h, beta * v + gamma * force)
    h_new = sphere_project(h + eta * u)
    return h_new, parallel_transport(h, h_new, u)


def inverse_softplus(value):
    return value + math.log(-math.expm1(-value))


class SphereMomentum(nn.Module):
    def __init__(self, config):
        super().__init__()
        b = config.momentum_beta_init
        self.raw = nn.Parameter(torch.tensor([
            math.log(b / (1 - b)), inverse_softplus(config.momentum_gamma_init),
            inverse_softplus(config.momentum_eta_init)], dtype=torch.float32))

    def coefficients(self):
        return self.raw[0].sigmoid(), F.softplus(self.raw[1]), F.softplus(self.raw[2])

    def forward(self, h, v, force):
        return momentum_step(h, v, force, *self.coefficients())


class RiemannianMomentumModel(ConnectionModel):
    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        super().prepare(device, dtype, compile_blocks=False)
        if compile_blocks:
            self.compiled_steps = [torch.compile(b.momentum_forward, dynamic=False, fullgraph=True)
                                   for b in self.blocks]
        return self

    def hidden(self, idx, kv_cache=None):
        # Replace only the residual stream's initial radial normalization.
        h = self.backbone(self.wte(idx), kv_cache)
        return norm(h.to(self.wte.weight.dtype))

    def backbone(self, x, kv_cache=None, return_layers=False):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + x.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        h = sphere_project(x)
        v = torch.zeros_like(h)
        cos, sin = self.cos[:, start:start+x.shape[1]], self.sin[:, start:start+x.shape[1]]
        states = [] if return_layers else None
        for i, b in enumerate(self.blocks):
            fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else b.momentum_forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                h, v = checkpoint(fn, h, v, cos, sin, use_reentrant=False)
            else:
                h, v = fn(h, v, cos, sin, kv_cache)
            if return_layers:
                states.append(h)
        # Velocity is local to this token's depth traversal. KV caching needs no
        # persistent velocity, and resets it for each new token/chunk.
        return (h, torch.stack(states, dim=2)) if return_layers else h

    @torch.no_grad()
    def residual_metrics(self):
        values = torch.stack([torch.stack(m.coefficients()) for b in self.blocks
                              for m in (b.attn_residual, b.mlp_residual)]).cpu()
        result = {}
        for i, row in enumerate(values.tolist()):
            for name, value in zip(('beta', 'gamma', 'eta'), row):
                result[f'residual/sublayer_{i}/{name}'] = value
        return result
