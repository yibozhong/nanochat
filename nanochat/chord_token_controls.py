"""Content-conditioned Chord preview and write coefficients."""
import math

import torch
from torch.nn import functional as F

from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project

GATE_EPS = 1e-6
CONTROL_NAMES = ('mu', 'alpha_perp', 'alpha_parallel', 'history', 'parallel_write', 'perpendicular_write')


def reference_read(h, v, weight, mu, steps, beta, gains, bounded=False):
    position = h.to(weight.dtype)
    velocity = v.to(weight.dtype)
    if bounded:
        joined = torch.cat((position, velocity), dim=-1)
        gate = F.rms_norm(joined, (joined.size(-1),), eps=GATE_EPS)
        z = F.linear(gate, weight.flatten(1))
    else:
        gate_v = velocity * (velocity.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
        z = F.linear(position, weight[:, 0]) + F.linear(gate_v, weight[:, 1])
    m = z[..., :1] + mu
    raw_steps = z[..., 1:3] + steps
    alpha = raw_steps.sigmoid() if bounded else F.softplus(raw_steps)
    if bounded:
        m = 3 * (m / 3).tanh()
    if beta is None:
        b, g = torch.full_like(m, .5), torch.full_like(alpha, .5)
    else:
        b, g = torch.sigmoid(z[..., 3:4] + beta), z[..., 4:] + gains
        if bounded:
            g = 2 * g.sigmoid()
    offset = m * velocity
    read = ((h.to(weight.dtype) + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()).to(h.dtype)
    q = F.rms_norm(read, (h.shape[-1],), eps=torch.finfo(torch.float32).eps)
    return q, torch.cat((m, alpha, b, g), dim=-1)


def reference_write(h, v, force, controls, floor, terminal):
    dtype = h.dtype
    h, v, f = h.to(controls.dtype), v.to(controls.dtype), force.to(controls.dtype)
    _, perp, par, a, b, c = controls.unsqueeze(-1).unbind(-2)
    delta = normalize_residual(f, floor) - h
    along = (delta*v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    n = normalize_residual((h + perp*delta) + ((par-perp)*along)*v)
    if terminal:
        return n.to(dtype), None
    u, d = tangent_project(n, v), tangent_project(n, n-h)
    par_d = u * ((u*d).mean(-1, keepdim=True) / u.square().mean(-1, keepdim=True).clamp_min(1e-12))
    return n.to(dtype), (a*u + b*par_d + c*(d-par_d)).to(dtype)


class TokenControlledSite(LookaheadResidual):
    def __init__(self, config, *, velocity_consumed=True):
        super().__init__(config, velocity_consumed=velocity_consumed, split_velocity=True)
        self.terminal = not velocity_consumed
        self.bounded = config.native_chord_write == 'token_bounded'
        if self.bounded:
            # Convert physical initial values to logits before the first token update.
            alpha = config.ngpt_alpha_init
            gain = 1 - config.momentum_beta_init
            with torch.no_grad():
                self.raw.fill_(math.log(alpha / (1 - alpha)))
                self.raw_parallel.copy_(self.raw)
                self.mu.fill_(3 * math.atanh(config.lookahead_mu_init / 3))
                if self.writes is not None:
                    self.writes.fill_(math.log(gain / (2 - gain)))
        self.token_weight = torch.nn.Parameter(torch.zeros(3 if self.terminal else 6, 2, config.width))
        self.eval_control_moments = None

    def read(self, h, v):
        steps = torch.cat((self.raw, self.raw_parallel))
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nanochat.chord_token_read import TokenRead
            read = TokenRead.apply
        else:
            read = reference_read
        q, controls = read(h, v, self.token_weight, self.mu, steps, self.raw_beta, self.writes, self.bounded)
        if not self.training and not torch.is_grad_enabled():
            # Keep just the token dispersion and mean of controls from the validation minibatch.
            self.eval_control_moments = torch.std_mean(controls.flatten(0, -2), dim=0, correction=0)
        return q, controls

    def write(self, h, v, force, controls):
        if h.is_cuda and h.dtype == torch.bfloat16 and h.size(-1) in (128, 512):
            from nanochat.chord_triton import TokenWrite
            return TokenWrite.apply(h, v, force, controls, self.rms_floor, self.terminal)
        return reference_write(h, v, force, controls, self.rms_floor, self.terminal)
