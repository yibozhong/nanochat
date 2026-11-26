"""Surviving token-dependent controls in the two from-rest Chord ablations."""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.chord_token_controls import GATE_EPS, SMALL_DYNAMIC_SCALE
from nanochat.gpt import norm
from nanochat.lookahead import AdditiveLookaheadResidual
from nanochat.momentum import NormalizedResidual, normalize_residual


def modulation(h, weight, scales):
    # Same FP32 h-only read, zero-initialized rows and learned group scales as full Chord.
    x = F.rms_norm(h.float(), (h.size(-1),), eps=GATE_EPS)
    return F.linear(x, weight * scales[:, None])


class SphereOnlySite(NormalizedResidual):
    def __init__(self, config):
        super().__init__(config)
        self.token_weight = nn.Parameter(torch.zeros(1, config.width))
        self.mix_scale = nn.Parameter(torch.full((1,), SMALL_DYNAMIC_SCALE))

    def forward(self, h, force):
        alpha = F.softplus(self.raw + modulation(h, self.token_weight, self.mix_scale))
        update = normalize_residual(force, self.rms_floor, soft_tau=self.soft_tau)
        return normalize_residual(h.float() + alpha * update).to(h.dtype)


class LookaheadOnlySite(AdditiveLookaheadResidual):
    def __init__(self, config, velocity_consumed):
        super().__init__(config, velocity_consumed)
        self.writes = nn.Parameter(torch.full((2,), 1 - self.decay)) if velocity_consumed else None
        self.token_weight = nn.Parameter(torch.zeros(4 if velocity_consumed else 1, config.width))
        self.mix_scales = nn.Parameter(torch.full((2 if velocity_consumed else 1,), SMALL_DYNAMIC_SCALE))

    def read(self, h, v):
        scales = self.mix_scales
        if not self.terminal:
            scales = torch.cat((scales[:1], scales[1:].expand(3)))
        z = modulation(h, self.token_weight, scales)
        mu = self.mu + z[..., :1]
        query = norm((h.float() + mu * v.float()).to(h.dtype))
        return query, z[..., 1:]

    def write(self, h, v, force, z):
        new = h + force
        if self.terminal:
            return new, None
        beta = (self.raw_beta + z[..., :1]).sigmoid()
        parallel, perpendicular = (self.writes + z[..., 1:]).split(1, dim=-1)
        old, d = v.float(), new.float() - h.float()
        along = (old*d).mean(-1, True) / old.square().mean(-1, True).clamp_min(1e-12)
        velocity = (beta + (parallel - perpendicular) * along) * old + perpendicular * d
        return new, velocity.to(v.dtype)
