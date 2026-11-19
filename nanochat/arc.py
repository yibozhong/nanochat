"""Arc: look ahead along transported weight displacements on a Frobenius sphere.

Run the forward and backward inside opt.gradient_point(), then call opt.step().
Evaluation and checkpoints use the base weights.
"""
from contextlib import contextmanager
from itertools import groupby
import math

import torch

from nanochat.optim import MuonAdamW, muon_direction

_compiled_muon_direction = torch.compile(muon_direction, dynamic=False, fullgraph=True)


@torch.compile(dynamic=False, fullgraph=True)
def _query(w, v, radius, mu):
    # v is tangent, so ||w + mu*v||² = radius² + mu² ||v||².
    return (w + mu * v) * torch.rsqrt(1 + (mu / radius).square() * v.square().sum())


@torch.compile(dynamic=False, fullgraph=True)
def _update(w, grad, v, radius, step_length, beta, relative_step=False):
    if relative_step:
        step_length = step_length * radius
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

    def __init__(self, groups, mu=1., beta=.5, direction='raw', step_scale='fan_out'):
        if step_scale not in ('fan_out', 'radius'):
            raise ValueError('Unknown Arc step scale')
        if direction not in ('raw', 'muon'):
            raise ValueError('Unknown Arc write direction')
        if not math.isfinite(mu) or not 0 <= beta < 1:
            raise ValueError('Arc requires finite mu and 0 <= beta < 1')
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError('Arc currently supports single-rank training')
        groups = list(groups)
        for i, g in enumerate(groups):
            if g['kind'] == 'muon':
                arc = dict(kind='arc', params=g['params'], lr=g['lr'], beta=beta, mu=mu)
                if step_scale == 'radius':
                    arc['step_scale'] = step_scale
                if direction == 'muon':
                    arc.update(direction=direction, momentum=g['momentum'], beta2=g['beta2'], ns_steps=g['ns_steps'])
                    if len({p.shape for p in g['params']}) != 1:
                        raise ValueError('Batched Muon directions require matrices of the same shape')
                groups[i] = arc
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

    def _write(self, p, direction, group):
        state = self.state[p]
        relative_step = group.get('step_scale') == 'radius'
        scale = 1. if relative_step else math.sqrt(p.shape[0])
        self._muon_lr_t.fill_(group['lr'] * scale)
        _update(p, direction, state['velocity'], state['radius'], self._muon_lr_t, group['beta'], relative_step)

    def _step_muon_direction(self, group):
        params = group['params']
        p = params[0]
        rows, cols = p.shape
        red_dim = -1 if rows >= cols else -2
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = p.new_zeros((len(params), rows, cols))
            factor_shape = (rows, 1) if rows >= cols else (1, cols)
            state['second_momentum_buffer'] = p.new_zeros((len(params), *factor_shape))
        self._muon_momentum_t.fill_(group['momentum'])
        self._muon_beta2_t.fill_(group['beta2'])
        # Contiguous runs let the dense training path update one stack in place;
        # a missing gradient also leaves that matrix's two Muon moments untouched.
        for active, run in groupby(range(len(params)), key=lambda i: params[i].grad is not None):
            if active:
                indices = list(run)
                span = slice(indices[0], indices[-1] + 1)
                grads = torch.stack([params[i].grad for i in indices])
                directions = _compiled_muon_direction(
                    grads, state['momentum_buffer'][span], state['second_momentum_buffer'][span],
                    self._muon_momentum_t, self._muon_beta2_t, group['ns_steps'], red_dim, p.dtype)
                for i, direction in zip(indices, directions.unbind(0)):
                    self._write(params[i], direction, group)

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
            if group.get('direction') == 'muon':
                self._step_muon_direction(group)
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                self._write(p, p.grad, group)

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
