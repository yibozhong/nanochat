"""Lookahead branch reads for spherical and vanilla additive residuals."""
import math

import torch
from torch import nn

from nanochat.geodesic_residual import geodesic_split_write
from nanochat.momentum import MomentumModel, NormalizedResidual, normalize_residual, state_dtype, tangent_project


class LookaheadResidual(NormalizedResidual):
    def __init__(self, config, *, velocity_consumed=True):
        super().__init__(config)
        self.mu = nn.Parameter(torch.tensor([config.lookahead_mu_init], dtype=torch.float32))
        self.kappa = (nn.Parameter(torch.zeros(1, dtype=torch.float32))
                      if config.ngpt_velocity_mode == 'lookahead_axis_write' else None)
        self.geodesic = config.ngpt_velocity_mode == 'lookahead_geodesic_split'
        self.raw_parallel = (nn.Parameter(self.raw.detach().clone())
                             if self.geodesic or config.split_chord_write else None)
        self.decay = config.momentum_beta_init
        self.gate_slopes = nn.Parameter(torch.zeros(2, dtype=torch.float32)) if config.token_chord_gates else None
        self.raw_beta = None
        if config.ngpt_velocity_mode == 'lookahead_split_learned_beta' and velocity_consumed:
            logit = math.log(self.decay / (1 - self.decay))
            self.raw_beta = nn.Parameter(torch.tensor([logit], dtype=torch.float32))

    def beta(self):
        return self.decay if self.raw_beta is None else self.raw_beta[0].sigmoid()

    def query(self, h, v):
        # h has RMS 1 and v is tangent. This form is exactly h when mu=0.
        offset = self.mu[0] * v
        return (h + offset) * torch.rsqrt(1 + offset.square().mean(-1, keepdim=True))

    def forward(self, h, v, force):
        if (self.gate_slopes is not None and h.is_cuda and h.dtype == v.dtype == torch.float32
                and force.dtype in (torch.bfloat16, torch.float32) and h.size(-1) in (128, 512)):
            from nanochat.chord_triton import fused_gated_write
            beta = self.beta() if self.raw_beta is not None else self.raw.new_tensor(self.decay)
            params = torch.stack((self.alpha(), self.step_size(self.raw_parallel), beta,
                                  self.gate_slopes[0], self.gate_slopes[1]))
            return fused_gated_write(h, v, force, params, self.rms_floor)
        return self.reference_write(h, v, force)

    def reference_write(self, h, v, force):
        if self.geodesic:
            return geodesic_split_write(h, v, force, self.alpha(), self.step_size(self.raw_parallel),
                                        self.decay, self.rms_floor)
        if self.kappa is None and self.raw_parallel is None:
            new_h = super().forward(h, force)
        else:
            h, v = state_dtype(h), state_dtype(v)
            delta = normalize_residual(force, self.rms_floor) - h
            # v is tangent at h. Use an RMS-squared floor to keep the write
            # and its derivatives finite even for an exactly zero velocity.
            dot = (delta * v).mean(-1, keepdim=True)
            v2 = v.square().mean(-1, keepdim=True).clamp_min(1e-12)
            along = dot / v2
            if self.raw_parallel is None:
                new_h = normalize_residual(h + self.alpha() * (delta + (self.kappa[0] * along) * v))
            else:
                # a_perp*(delta - P) + a_parallel*P, grouped so equal steps
                # reproduce the original Arc write without roundoff in delta-P.
                perp = self.alpha()
                parallel = self.step_size(self.raw_parallel)
                if self.gate_slopes is not None:
                    inv_delta = delta.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
                    cosine = dot * (v2.rsqrt() * inv_delta)
                    perp = perp * (2 * torch.sigmoid(self.gate_slopes[0] * cosine))
                    parallel = parallel * (2 * torch.sigmoid(self.gate_slopes[1] * cosine))
                new_h = normalize_residual((h + perp * delta) + ((parallel - perp) * along) * v)
        # The Householder reflector with normal h + new_h gives short-arc
        # parallel transport. At opposite poles the two tangent planes coincide.
        bisector = h + new_h
        norm2 = bisector.square().mean(-1, keepdim=True)
        scale = 2 * (v * bisector).mean(-1, keepdim=True) / norm2.clamp_min(1e-12)
        carried = v - scale * bisector
        beta = self.beta()
        v = tangent_project(new_h, beta * carried + (1 - beta) * (new_h - h))
        return new_h, v


def additive_branch_input(h, v, mu):
    # Keep the BF16 query boundary visible to the branch backward.
    return (state_dtype(h) + mu[0] * v).to(h.dtype)


def compile_additive_step(block, options):
    read = torch.compile(additive_branch_input, **options)
    attention = torch.compile(block.attention_branch, **options)
    ffn = torch.compile(block.mlp_branch, **options)
    attention_write = torch.compile(block.attn_residual.forward, **options)
    ffn_write = torch.compile(block.mlp_residual.forward, **options)

    def step(h, v, cos, sin, cache=None):
        q = read(h, v, block.attn_residual.mu)
        h, v = attention_write(h, v, attention(q, cos, sin, cache))
        q = read(h, v, block.mlp_residual.mu)
        return ffn_write(h, v, ffn(q))
    return step


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
        if sites[0].raw_parallel is not None:
            values = torch.stack([torch.stack((s.alpha(), s.step_size(s.raw_parallel), s.mu[0]))
                                  for s in sites]).tolist()
            metrics = {f'residual/sublayer_{i}/{key}': value for i,row in enumerate(values)
                       for key,value in zip(('alpha_perp','alpha_parallel','mu'),row)}
            if sites[0].raw_beta is not None:
                betas = torch.stack([s.beta() for s in sites if s.raw_beta is not None]).tolist()
                metrics.update({f'residual/sublayer_{i}/beta': value for i,value in enumerate(betas)})
            if sites[0].gate_slopes is not None:
                slopes = torch.stack([s.gate_slopes for s in sites]).tolist()
                metrics.update({f'residual/sublayer_{i}/gate_{axis}_slope': value
                                for i,row in enumerate(slopes) for axis,value in zip(('perp','parallel'),row)})
            return metrics
        values = torch.stack([torch.stack((site.alpha(), site.mu[0])) for site in sites]).tolist()
        metrics = {f'residual/sublayer_{i}/{key}': value
                   for i, row in enumerate(values) for key, value in zip(('alpha', 'mu'), row)}
        if sites[0].kappa is not None:
            kappas = torch.stack([site.kappa[0] for site in sites]).tolist()
            metrics.update({f'residual/sublayer_{i}/kappa': k for i, k in enumerate(kappas)})
        return metrics
