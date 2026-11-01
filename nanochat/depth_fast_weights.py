"""Per-token depth memory with analytic inner gradients and functional Muon.

SwiGLU gradient structure follows VideoChat3 FastWeightSwiGLU at
004d5eff3fbd20c80a5bdbb60f838825c8cb683e. NS5 coefficients/orientation and
EMA/Nesterov momentum follow MiniWorld muon.py at
4e043041341038bfc9933ad3e052e206725e99b7. No reference code is executed.

Unlike VideoLACT, states are independent per (batch, token), updates have a
post-NS learning rate, carry momentum across depth, and do not normalize weights.
All runtime states are functional tensors, never optimizer-owned Parameters.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.gpt import Linear


def swiglu(theta, key):
    gate_w, up_w, down_w = theta
    gate = (gate_w @ key.unsqueeze(-1)).squeeze(-1)
    up = (up_w @ key.unsqueeze(-1)).squeeze(-1)
    hidden = F.silu(gate) * up
    output = (down_w @ hidden.unsqueeze(-1)).squeeze(-1)
    return output, (gate, up, hidden)


def binding_gradient(theta, key, delta, *, stopped_old=None):
    """Partial gradient wrt each theta, holding key/target fixed INNER arguments.

Outer autograd still sees key, delta, theta and the Jacobian of the gradient.
Do not replace error by -delta: old-stopgrad(old) has a nonzero derivative.
stopped_old is only an oracle hook for finite differences with frozen anchors.
"""
    old, (gate, up, hidden) = swiglu(theta, key)
    anchor = old.detach() if stopped_old is None else stopped_old.detach()
    error = old - (anchor + delta)
    down_w = theta[2]
    d_hidden = (down_w.mT @ error.unsqueeze(-1)).squeeze(-1)
    sigmoid = gate.sigmoid()
    d_gate = d_hidden * up * sigmoid * (1 + gate * (1 - sigmoid))
    d_up = d_hidden * F.silu(gate)
    gradients = (
        d_gate.unsqueeze(-1) * key.unsqueeze(-2),
        d_up.unsqueeze(-1) * key.unsqueeze(-2),
        error.unsqueeze(-1) * hidden.unsqueeze(-2),
    )
    return gradients, old, .5 * error.square().sum(-1)


def newton_schulz(matrix, steps=5, eps=1e-7, bf16=False):
    """Batched NS5; reductions never mix independent token states."""
    x = matrix.to(torch.bfloat16) if bf16 else matrix
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = x @ x.mT
        polynomial = -4.7750 * gram + (2.0315 * gram) @ gram
        x = 3.4445 * x + polynomial @ x
    return (x.mT if transpose else x).to(matrix.dtype)


def inner_muon(theta, momentum, gradients, learning_rates, *, beta=.95,
               ns_steps=5, ns_eps=1e-7, ns_bf16=False):
    """EMA momentum, Nesterov direction, NS5, shape scale, functional update.

The lr is applied AFTER NS, so positive rescaling is not normalized away.
No in-place writes, decay, detach, gradient clipping, or optimizer.step().
"""
    next_theta, next_momentum = [], []
    for weight, previous, gradient, lr in zip(theta, momentum, gradients, learning_rates):
        velocity = beta * previous + (1 - beta) * gradient
        direction = (1 - beta) * gradient + beta * velocity
        orthogonal = newton_schulz(direction, ns_steps, ns_eps, ns_bf16)
        scale = max(1., weight.shape[-2] / weight.shape[-1]) ** .5
        next_theta.append(weight - lr * scale * orthogonal)
        next_momentum.append(velocity)
    return tuple(next_theta), tuple(next_momentum)


def state_memory_estimate(batch, tokens, width, bottleneck, depth, bytes_per_value=4):
    matrices = batch * tokens * 3 * width * bottleneck * bytes_per_value
    return dict(fast_weight_bytes=matrices, momentum_bytes=matrices,
        live_state_bytes=2*matrices,
        checkpoint_input_states_upper_bound_bytes=(depth+1)*2*matrices,
        note='State-only estimate with one checkpoint per two-sublayer block; excludes activations, NS recomputation, gradients, optimizer, allocator and temporary tensors.')


class DepthFastWeights(nn.Module):
    def __init__(self, width, bottleneck=8, inner_lr=.01, momentum=.95,
                 ns_steps=5, ns_eps=1e-7, ns_bf16=False, learnable_lr=True):
        super().__init__()
        if min(width, bottleneck, inner_lr, ns_eps) <= 0 or not 0 <= momentum < 1 or ns_steps < 1:
            raise ValueError('Invalid FW dimensions or Inner Muon hyperparameters')
        self.width, self.bottleneck = width, bottleneck
        self.momentum, self.ns_steps, self.ns_eps = momentum, ns_steps, ns_eps
        self.ns_bf16 = ns_bf16
        self.qk = Linear(width, width, bias=False)
        self.gate = nn.Parameter(torch.empty(bottleneck, width))
        self.up = nn.Parameter(torch.empty(bottleneck, width))
        self.down = nn.Parameter(torch.empty(width, bottleneck))
        rates = torch.full((3,), math.log(inner_lr))
        if learnable_lr:
            self.log_learning_rates = nn.Parameter(rates)
        else:
            self.register_buffer('log_learning_rates', rates)
        for matrix, fan_in in ((self.qk.weight,width),(self.gate,width),(self.up,width),(self.down,bottleneck)):
            nn.init.normal_(matrix, std=1/math.sqrt(fan_in))

    def key(self, h):
        # Shared Q/K; native-style unit RMS, explicitly FP32 (or FP64 oracle).
        dtype = self.gate.dtype
        projected = self.qk(h).to(dtype)
        return F.rms_norm(projected, (self.width,), eps=1e-5)

    def initialize(self, h):
        """Bind h0 into W_down with a differentiable rank-one interpolation.

There is no hidden bypass. Reading the resulting FW at k0 reproduces h0 up
to denominator epsilon; h0 lives in the runtime matrix. This initialization
is not an auxiliary training loss or an extra learned V projection.
"""
        shape = h.shape[:-1]
        theta = tuple(w.expand(*shape, *w.shape) for w in (self.gate,self.up,self.down))
        key = self.key(h)
        old, (_, _, feature) = swiglu(theta, key)
        correction = (h.to(old.dtype)-old).unsqueeze(-1) * feature.unsqueeze(-2)
        denominator = feature.square().sum(-1,keepdim=True).unsqueeze(-1) + 1e-6
        theta = (theta[0],theta[1],theta[2]+correction/denominator)
        momentum = tuple(torch.zeros_like(w) for w in theta)
        output, _ = swiglu(theta, key)
        return output.to(h.dtype), theta, momentum

    def update_read(self, h, delta, theta, momentum):
        key = self.key(h)
        gradients, _, _ = binding_gradient(theta, key, delta.to(theta[0].dtype))
        theta, momentum = inner_muon(theta, momentum, gradients, self.log_learning_rates.exp(),
            beta=self.momentum, ns_steps=self.ns_steps, ns_eps=self.ns_eps, ns_bf16=self.ns_bf16)
        output, _ = swiglu(theta, key)
        return output.to(h.dtype), theta, momentum
