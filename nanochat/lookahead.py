"""Lookahead branch reads for spherical and vanilla additive residuals."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanochat.geodesic_residual import geodesic_split_write
from nanochat.momentum import MomentumModel, NormalizedResidual, normalize_residual, state_dtype, tangent_project


def _read(h, v, mu, ambient):
    offset = mu[0] * state_dtype(v)
    read = state_dtype(h) + offset
    if ambient:
        return normalize_residual(read)
    return read * torch.rsqrt(1 + offset.square().mean(-1, keepdim=True))


class LookaheadResidual(NormalizedResidual):
    def __init__(self, config, *, velocity_consumed=True):
        super().__init__(config)
        self.mu = nn.Parameter(torch.tensor([config.lookahead_mu_init], dtype=torch.float32))
        self.kappa = (nn.Parameter(torch.zeros(1, dtype=torch.float32))
                      if config.ngpt_velocity_mode == 'lookahead_axis_write' else None)
        self.geodesic = config.ngpt_velocity_mode == 'lookahead_geodesic_split'
        self.raw_parallel = (nn.Parameter(self.raw.detach().clone())
                             if self.geodesic or config.split_chord_write else None)
        self.velocity_geometry = config.chord_velocity_geometry
        self.decay = config.momentum_beta_init
        self.gate_slopes = nn.Parameter(torch.zeros(2, dtype=torch.float32)) if config.token_chord_gates else None
        self.raw_beta = None
        self.beta_gate_slope = None
        if config.ngpt_velocity_mode == 'lookahead_split_learned_beta' and velocity_consumed:
            logit = math.log(self.decay / (1 - self.decay))
            self.raw_beta = nn.Parameter(torch.full_like(self.raw, logit))
            if config.token_beta_gate:
                self.beta_gate_slope = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def beta(self):
        return self.decay if self.raw_beta is None else self.raw_beta.squeeze(0).sigmoid()

    def query(self, h, v):
        # The tangent-only denominator omits the vanishing cross term.
        ambient = self.velocity_geometry == 'ambient'
        if h.dtype == torch.bfloat16 and h.is_cuda and h.size(-1) in (128, 512):
            from nanochat.chord_triton import bf16_query
            return bf16_query(h, v, self.mu, ambient)
        if h.dtype == torch.bfloat16 and torch.is_grad_enabled():
            return checkpoint(_read, h, v, self.mu, ambient, use_reentrant=False, preserve_rng_state=False).to(h.dtype)
        return _read(h, v, self.mu, ambient).to(h.dtype)

    def branch_input(self, h, v, dtype, fuse_norm):
        if fuse_norm and h.is_cuda and h.dtype == dtype == torch.bfloat16 and h.size(-1) in (128, 512):
            from nanochat.chord_triton import bf16_query
            return bf16_query(h, v, self.mu, self.velocity_geometry == 'ambient', normalize=True)
        q = self.query(h, v).to(dtype)
        return torch.nn.functional.rms_norm(q, (q.size(-1),))

    def projected_write(self, h, v, features, layer, recompute):
        if not recompute:
            return self(h, v, layer(features))
        if not h.is_cuda or h.size(-1) not in (128, 512):
            return checkpoint(lambda x, y, z: self(x, y, layer(z)), h, v, features,
                              use_reentrant=False, preserve_rng_state=False)
        weight = layer.weight.to(features.dtype)
        force = torch.nn.functional.linear(features, weight)
        return self(h, v, force, projection=(features, weight))

    def forward(self, h, v, force, projection=()):
        if (self.geodesic or not h.is_cuda or h.dtype not in (torch.bfloat16, torch.float32) or v.dtype != h.dtype
                or force.dtype not in (torch.bfloat16, torch.float32) or h.size(-1) not in (128, 512)):
            if h.dtype == torch.bfloat16 and torch.is_grad_enabled():
                return checkpoint(self.reference_write, h, v, force, use_reentrant=False, preserve_rng_state=False)
            return self.reference_write(h, v, force)
        from nanochat.chord_triton import fused_axis_write, fused_beta_write, fused_gated_write, fused_split_write, fused_write
        perp = self.alpha()
        if self.kappa is not None:
            params = torch.stack((perp, self.kappa[0]))
            return fused_axis_write(h, v, force, params, self.decay, self.rms_floor)
        if self.raw_parallel is None:
            return fused_write(h, v, force, perp, self.decay, self.rms_floor)
        parallel = self.step_size(self.raw_parallel)
        if self.beta_gate_slope is not None:
            params = torch.stack((perp, parallel, self.raw_beta[0], self.beta_gate_slope[0]))
            return fused_beta_write(h, v, force, params, self.rms_floor)
        beta = self.beta() if self.raw_beta is not None else torch.full_like(perp, self.decay)
        if self.gate_slopes is not None:
            params = torch.stack((perp, parallel, beta, self.gate_slopes[0], self.gate_slopes[1]))
            return fused_gated_write(h, v, force, params, self.rms_floor)
        return fused_split_write(h, v, force, torch.stack((perp, parallel, beta)), self.rms_floor, self.velocity_geometry, projection)

    def reference_write(self, h, v, force):
        if self.geodesic:
            return geodesic_split_write(h, v, force, self.alpha(), self.step_size(self.raw_parallel),
                                        self.decay, self.rms_floor)
        storage_dtype = h.dtype
        h, v = state_dtype(h), state_dtype(v)
        if self.kappa is None and self.raw_parallel is None:
            new_h = super().forward(h, force)
        else:
            delta = normalize_residual(force, self.rms_floor) - h
            # Use an RMS-squared floor to keep the write
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
        if self.velocity_geometry == 'parallel_transport':
            bisector = h + new_h
            norm2 = bisector.square().mean(-1, keepdim=True)
            scale = 2 * (v * bisector).mean(-1, keepdim=True) / norm2.clamp_min(1e-12)
            carried = v - scale * bisector
        else:
            carried = v
        chord = new_h - h
        if self.beta_gate_slope is None:
            beta = self.beta()
        else:
            # Compare the transported history and realized step in the same tangent space.
            recent = tangent_project(new_h, chord)
            inv_history = carried.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            inv_recent = recent.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            alignment = (carried * recent).mean(-1, keepdim=True) * (inv_history * inv_recent)
            beta = torch.sigmoid(self.raw_beta[0] + self.beta_gate_slope[0] * alignment)
        velocity = beta * carried + (1 - beta) * chord
        if self.velocity_geometry != 'ambient':
            velocity = tangent_project(new_h, velocity)
        return new_h.to(storage_dtype), velocity.to(storage_dtype)


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


def compile_split_block(block, next_mu, *, first, preview, terminal):
    """Carry the next BF16 branch input across block boundaries with the configured residual stream precision."""
    from nanochat.chord_triton import split_boundary
    attn, mlp = block.attn_residual, block.mlp_residual

    def step(h, v, q, cos, sin, cache=None):
        force = block.attention_branch(q, cos, sin, cache)
        if preview:
            h, v, q = split_boundary(attn, h, v, force, mlp.mu)
        else:
            h, v = attn(h, v, force)
            q = mlp.query(h, v).to(cos.dtype)
        force = block.mlp_branch(q)
        if terminal:
            return (split_boundary(mlp, h, v, force, None),)
        if preview and next_mu is not None:
            return split_boundary(mlp, h, v, force, next_mu)
        return mlp(h, v, force)

    def starting_step(h, v, cos, sin, cache=None):
        q = attn.query(h, v).to(cos.dtype)
        return step(h, v, q, cos, sin, cache)

    return torch.compile(starting_step if first else step, dynamic=False, fullgraph=True,
                         options={'emulate_precision_casts': True})


class LookaheadModel(MomentumModel):
    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False, *,
                compile_lookahead_branches=False, chord_kernel='current'):
        if compile_blocks and torch.device(device).type == 'cuda' and self.config.residual_mode == 'ngpt_lookahead':
            from nanochat import chord_triton  # Register the backward GEMM before Dynamo tracing.
        super().prepare(device, dtype, compile_blocks, compile_lookahead_branches=compile_lookahead_branches)
        if chord_kernel == 'current':
            return self
        if chord_kernel not in ('terminal', 'boundary', 'both'):
            raise ValueError(f'Unknown Chord kernel {chord_kernel}')
        c = self.config
        if c.channel_chord_coefficients:
            raise ValueError('Channel alpha and beta use the current Chord write kernel')
        if c.fuse_chord_read_norm or c.recompute_chord_force:
            raise ValueError('Fused read/norm and force recomputation use the current write kernel')
        if (c.ngpt_velocity_mode != 'lookahead_split_learned_beta' or c.signed_chord_steps or
                c.token_beta_gate or c.token_chord_gates or c.width not in (128, 512) or
                torch.device(device).type != 'cuda' or dtype != torch.bfloat16 or not compile_blocks):
            raise ValueError('Boundary kernels require compiled BF16 CUDA shared positive split steps and layer beta')
        preview = chord_kernel in ('boundary', 'both')
        for i, block in enumerate(self.blocks):
            last = i + 1 == len(self.blocks)
            if preview or last:
                next_mu = None if last else self.blocks[i+1].attn_residual.mu
                self.compiled_steps[i] = compile_split_block(
                    block, next_mu, first=(i == 0 or not preview), preview=preview,
                    terminal=(last and chord_kernel in ('terminal', 'both')))
        return self

    def initial_state(self, x, token_ids=None, start=0):
        if self.config.chord_state_dtype == 'bfloat16' and torch.is_grad_enabled():
            # Recompute this one projection from the embedding lookup on backward.
            # Keep its FP32 normalization intermediates out of the activation tape.
            return checkpoint(self._initial_state, x, token_ids, start, use_reentrant=False, preserve_rng_state=False)
        return self._initial_state(x, token_ids, start)

    def _initial_state(self, x, token_ids, start):
        h, v = super().initial_state(x, token_ids, start)
        if self.config.residual_mode == 'lookahead_only':
            return h.to(x.dtype), v
        if self.config.chord_velocity_geometry != 'ambient':
            v = tangent_project(h, v)
        if self.config.chord_state_dtype == 'bfloat16':
            return h.bfloat16(), v.bfloat16()
        return h, v

    @torch.no_grad()
    def residual_metrics(self):
        sites = [site for b in self.blocks for site in (b.attn_residual, b.mlp_residual)]
        if self.config.residual_mode == 'lookahead_only':
            mus = torch.stack([site.mu[0] for site in sites]).tolist()
            return {f'residual/sublayer_{i}/mu': value for i, value in enumerate(mus)}
        if sites[0].raw_parallel is not None:
            values = torch.stack([torch.stack((s.alpha().mean(), s.step_size(s.raw_parallel).mean(), s.mu[0]))
                                  for s in sites]).tolist()
            metrics = {f'residual/sublayer_{i}/{key}': value for i,row in enumerate(values)
                       for key,value in zip(('alpha_perp','alpha_parallel','mu'),row)}
            if sites[0].raw_beta is not None:
                betas = torch.stack([s.beta().mean() for s in sites if s.raw_beta is not None]).tolist()
                key = 'beta_at_zero_alignment' if sites[0].beta_gate_slope is not None else 'beta'
                metrics.update({f'residual/sublayer_{i}/{key}': value for i,value in enumerate(betas)})
            if self.config.channel_chord_coefficients:
                for key, vectors in (('alpha_perp', [s.alpha() for s in sites]),
                                     ('alpha_parallel', [s.step_size(s.raw_parallel) for s in sites]),
                                     ('beta', [s.beta() for s in sites if s.raw_beta is not None])):
                    spreads = torch.stack(vectors).std(dim=-1, unbiased=False).tolist()
                    metrics.update({f'residual/sublayer_{i}/{key}_std': x for i, x in enumerate(spreads)})
            if sites[0].beta_gate_slope is not None:
                slopes = torch.stack([s.beta_gate_slope[0] for s in sites if s.beta_gate_slope is not None]).tolist()
                metrics.update({f'residual/sublayer_{i}/beta_gate_slope': value for i,value in enumerate(slopes)})
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
