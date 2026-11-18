"""Arc: look ahead along transported weight displacements on a Frobenius sphere.

Run the forward and backward inside opt.gradient_point(), then call opt.step().
Evaluation and checkpoints use the base weights.
"""
from contextlib import contextmanager
import math

import torch

from nanochat.optim import MuonAdamW


@torch.compile(dynamic=False, fullgraph=True)
def _query(w, v, radius, mu):
    # v is tangent, so ||w + mu*v||² = radius² + mu² ||v||².
    return (w + mu * v) * torch.rsqrt(1 + (mu / radius).square() * v.square().sum())


@torch.compile(dynamic=False, fullgraph=True)
def _update(w, grad, v, radius, step_length, beta):
    norm = torch.linalg.vector_norm(grad)
    trial = w - (step_length / norm.clamp_min(1e-30)) * grad
    trial_norm = torch.linalg.vector_norm(trial)
    # Leave the base untouched for zero updates or undefined retractions.
    move = (norm > 0) & (step_length > 0) & (trial_norm > 0)
    new = torch.where(move, trial * (radius / trial_norm.clamp_min(1e-30)), w)
    bisector = w + new
    carried = v - bisector * (2 * (v * bisector).sum() / bisector.square().sum().clamp_min(1e-30))
    velocity = torch.lerp(carried, new - w, 1 - beta)
    velocity = velocity - new * ((velocity * new).sum() / radius.square())
    v.copy_(velocity)
    w.copy_(new)


class ArcAdamW(MuonAdamW):
    """Single-GPU Arc for backbone matrices and nanochat AdamW for other groups."""

    def __init__(self, groups, mu=1., beta=.5):
        if not math.isfinite(mu) or not 0 <= beta < 1:
            raise ValueError('Arc requires finite mu and 0 <= beta < 1')
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('Arc currently supports single-rank training')
        groups = [dict(kind='arc', params=g['params'], lr=g['lr'], beta=beta, mu=mu)
                  if g['kind'] == 'muon' else g for g in groups]
        super().__init__(groups)
        self._query_active = False
        self._gradient_ready = False
        radii = []
        for group in self.param_groups:
            if group['kind'] != 'arc':
                continue
            for p in group['params']:
                if p.ndim != 2 or p.dtype != torch.float32:
                    raise ValueError('Arc expects FP32 matrix weights')
                radius = torch.linalg.vector_norm(p.detach())
                radii.append(radius)
                self.state[p].update(radius=radius, velocity=torch.zeros_like(p))
        radii = torch.stack(radii)
        if not bool((torch.isfinite(radii) & (radii > 0)).all()):
            raise ValueError('Arc needs finite, nonzero initial matrix norms')

    @contextmanager
    def gradient_point(self):
        """Install detached query weights across all microbatches and restore exactly."""
        if self._query_active:
            raise RuntimeError('Nested Arc gradient points')
        self._query_active = True
        self._gradient_ready = False
        bases = []
        try:
            with torch.no_grad():
                for group in self.param_groups:
                    if group['kind'] != 'arc' or group['mu'] == 0:
                        continue
                    for p in group['params']:
                        state = self.state[p]
                        bases.append((p, p.detach().clone()))
                        p.copy_(_query(p, state['velocity'], state['radius'], group['mu']))
            yield
            self._gradient_ready = True
        finally:
            with torch.no_grad():
                for p, base in bases:
                    p.copy_(base)
            self._query_active = False

    def state_dict(self):
        if self._query_active:
            raise RuntimeError('Cannot checkpoint transient Arc query weights')
        return super().state_dict()

    @torch.no_grad()
    def step(self):
        if self._query_active or not self._gradient_ready:
            raise RuntimeError('Compute gradients inside Arc.gradient_point() before step()')
        self._gradient_ready = False
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                info = self._reduce_adamw(group, world_size=1)
                self._compute_adamw(group, info, [], rank=0, world_size=1)
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Muon's nominal update norm on an [out,in] matrix is lr*sqrt(out).
                self._muon_lr_t.fill_(group['lr'] * math.sqrt(p.shape[0]))
                _update(p, p.grad, state['velocity'], state['radius'], self._muon_lr_t, group['beta'])

    @torch.no_grad()
    def metrics(self):
        angles, norm_errors = [], []
        for group in self.param_groups:
            if group['kind'] == 'arc':
                for p in group['params']:
                    state = self.state[p]
                    radius = state['radius']
                    angles.append(torch.atan(abs(group['mu']) * state['velocity'].norm() / radius))
                    norm_errors.append((p.norm() / radius - 1).abs())
        return {'optimizer/mean_query_degrees': torch.stack(angles).mean().rad2deg().item(),
                'optimizer/max_relative_radius_error': torch.stack(norm_errors).max().item()}
