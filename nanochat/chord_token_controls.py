"""A content-conditioned Chord site. Only the velocity sent to the gate is RMS-normalized."""
import torch
from torch.nn import functional as F

from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project

GATE_EPS = 1e-6
CONTROL_NAMES = ('mu', 'alpha_perp', 'alpha_parallel', 'history', 'parallel_write', 'perpendicular_write')


def reference_read(h, v, weight, mu, steps, beta, gains):
    velocity = v.to(weight.dtype)
    gate_v = velocity * (velocity.square().mean(-1, keepdim=True) + GATE_EPS).rsqrt()
    z = F.linear(h.to(weight.dtype), weight[:, 0]) + F.linear(gate_v, weight[:, 1])
    m = z[..., :1] + mu
    alpha = F.softplus(z[..., 1:3] + steps)
    if beta is None:
        b, g = torch.full_like(m, .5), torch.full_like(alpha, .5)
    else:
        b, g = torch.sigmoid(z[..., 3:4] + beta), z[..., 4:] + gains
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
        self.token_weight = torch.nn.Parameter(torch.zeros(3 if self.terminal else 6, 2, config.width))
        self.eval_control_moments = None

    def read(self, h, v):
        steps = torch.cat((self.raw, self.raw_parallel))
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nanochat.chord_token_read import TokenRead
            read = TokenRead.apply
        else:
            read = reference_read
        q, controls = read(h, v, self.token_weight, self.mu, steps, self.raw_beta, self.writes)
        if not self.training and not torch.is_grad_enabled():
            # Keep just the token dispersion and mean of controls from the validation minibatch.
            self.eval_control_moments = torch.std_mean(controls.flatten(0, -2), dim=0, correction=0)
        return q, controls

    def write(self, h, v, force, controls):
        if h.is_cuda and h.dtype == torch.bfloat16 and h.size(-1) in (128, 512):
            from nanochat.chord_triton import TokenWrite
            return TokenWrite.apply(h, v, force, controls, self.rms_floor, self.terminal)
        return reference_write(h, v, force, controls, self.rms_floor, self.terminal)
