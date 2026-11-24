"""Vanilla Chord with one token-conditioned lookahead, step and retention."""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.momentum import inverse_softplus, normalize_residual, tangent_project

GATE_EPS = 1e-6


def reference_read(h, v, weight, mu, step, beta):
    position = h.to(weight.dtype)
    z = F.linear(F.rms_norm(position, (h.size(-1),), eps=GATE_EPS), weight)
    preview = z[..., :1] + mu
    alpha = F.softplus(z[..., 1:2] + step)
    retention = .5 * torch.ones_like(alpha) if beta is None else (beta + z[..., 2:3]).sigmoid()
    offset = preview * v.to(weight.dtype)
    read = ((position + offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()).to(h.dtype)
    query = F.rms_norm(read, (h.size(-1),), eps=torch.finfo(torch.float32).eps)
    return query, torch.cat((preview, alpha, retention), dim=-1)


def reference_write(h, v, force, controls, floor):
    position = h.to(controls.dtype)
    alpha, beta = controls[..., 1:2], controls[..., 2:3]
    delta = normalize_residual(force.to(controls.dtype), floor) - position
    new = normalize_residual(position + alpha * delta)
    velocity = beta * v.to(controls.dtype) + (1 - beta) * (new - position)
    return new.to(h.dtype), tangent_project(new, velocity).to(v.dtype)


class SingleStepSite(nn.Module):
    def __init__(self, config, velocity_consumed):
        super().__init__()
        count = 3 if velocity_consumed else 2
        self.weight = nn.Parameter(torch.zeros(count, config.width))
        self.mu = nn.Parameter(torch.tensor([config.lookahead_mu_init], dtype=torch.float32))
        self.step = nn.Parameter(torch.tensor([inverse_softplus(config.ngpt_alpha_init)]))
        self.beta = (nn.Parameter(torch.tensor([config.momentum_beta_init]).logit())
                     if velocity_consumed else None)
        self.floor = config.ngpt_force_rms_floor
        self.eval_moments = None

    def read(self, h, v):
        read = reference_read
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nanochat.chord_single_step_triton import TokenRead
            read = TokenRead.apply
        q, controls = read(h, v, self.weight, self.mu, self.step, self.beta)
        if not self.training and not torch.is_grad_enabled():
            self.eval_moments = torch.std_mean(controls.flatten(0, -2), dim=0, correction=0)
        return q, controls

    def write(self, h, v, force, controls):
        if h.is_cuda and h.dtype == torch.bfloat16:
            from nanochat.chord_single_step_triton import TokenWrite
            return TokenWrite.apply(h, v, force, controls, self.floor)
        return reference_write(h, v, force, controls, self.floor)
