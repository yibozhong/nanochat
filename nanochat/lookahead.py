"""Lookahead branch reads for spherical and vanilla additive residuals."""
import torch
from torch import nn

from nanochat.momentum import MomentumModel, NormalizedResidual, state_dtype, tangent_project


class LookaheadResidual(NormalizedResidual):
    def __init__(self, config):
        super().__init__(config)
        self.mu = nn.Parameter(torch.tensor([config.lookahead_mu_init], dtype=torch.float32))
        self.decay = config.momentum_beta_init

    def query(self, h, v):
        # h has RMS 1 and v is tangent. This form is exactly h when mu=0.
        offset = self.mu[0] * v
        return (h + offset) * torch.rsqrt(1 + offset.square().mean(-1, keepdim=True))

    def forward(self, h, v, force):
        new_h = super().forward(h, force)
        # The Householder reflector with normal h + new_h gives short-arc
        # parallel transport. At opposite poles the two tangent planes coincide.
        bisector = h + new_h
        norm2 = bisector.square().mean(-1, keepdim=True)
        scale = 2 * (v * bisector).mean(-1, keepdim=True) / norm2.clamp_min(1e-12)
        carried = v - scale * bisector
        v = tangent_project(new_h, self.decay * carried + (1 - self.decay) * (new_h - h))
        return new_h, v


class AdditiveLookaheadResidual(nn.Module):
    """Read ahead in Euclidean space and write through the vanilla residual."""
    def __init__(self, config):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor([config.lookahead_mu_init], dtype=torch.float32))
        self.decay = config.momentum_beta_init

    def query(self, h, v):
        return state_dtype(h) + self.mu[0] * v

    def forward(self, h, v, force):
        new_h = h + force
        # Track the realized displacement of the vanilla (normally BF16) sum.
        displacement = state_dtype(new_h) - state_dtype(h)
        return new_h, self.decay * v + (1 - self.decay) * displacement


class LookaheadModel(MomentumModel):
    def initial_state(self, x, token_ids=None, start=0):
        h, v = super().initial_state(x, token_ids, start)
        if self.config.residual_mode == 'lookahead_only':
            return h.to(x.dtype), v
        return h, tangent_project(h, v)

    @torch.no_grad()
    def residual_metrics(self):
        sites = [site for b in self.blocks for site in (b.attn_residual, b.mlp_residual)]
        if self.config.residual_mode == 'lookahead_only':
            mus = torch.stack([site.mu[0] for site in sites]).tolist()
            return {f'residual/sublayer_{i}/mu': value for i, value in enumerate(mus)}
        values = torch.stack([torch.stack((site.alpha(), site.mu[0])) for site in sites]).tolist()
        return {f'residual/sublayer_{i}/{key}': value
                for i, row in enumerate(values) for key, value in zip(('alpha', 'mu'), row)}
