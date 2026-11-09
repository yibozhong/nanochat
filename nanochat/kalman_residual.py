"""Learned diagonal Kalman-inspired residuals; not a full BFT reproduction.

Each token has a positive variance carried across attention/MLP sublayers.
Identity dynamics: prior=P+Q, K=prior/(prior+R), posterior=K*R.
The observation innovation is 2*branch, so P0=Q=.5 and R=1 give
K=.5 and the original residual coefficient 1 at initialization. Q and R
are learned variance proxies, not statistically calibrated uncertainties.
No attention reweighting, REML moments, Jacobian propagation or detach.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import norm

VARIANCE_FLOOR = 1e-4
INITIAL_VARIANCE = .5
INNOVATION_SCALE = 2.


def kalman_update(x, variance, branch, condition, raw_q, bias, slope):
    """Pure tensor implementation, fused by torch.compile with its block.

    Variance arithmetic remains FP32 in BF16 training. FP64 is retained for
    independent mathematical checks. posterior=K*R avoids 1-K cancellation.
    """
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    prior = variance.to(dtype) + F.softplus(raw_q.to(dtype)) + VARIANCE_FLOOR
    observation = F.softplus(bias.to(dtype) + slope.to(dtype)*condition.to(dtype)) + VARIANCE_FLOOR
    gain = prior / (prior + observation)
    output = (x.to(dtype) + INNOVATION_SCALE*gain*branch.to(dtype)).to(x.dtype)
    return output, gain*observation


class DiagonalNoise(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.raw_q = nn.Parameter(torch.full((width,), math.log(math.expm1(.5-VARIANCE_FLOOR))))
        self.bias = nn.Parameter(torch.full((width,), math.log(math.expm1(1.-VARIANCE_FLOOR))))
        self.slope = nn.Parameter(torch.zeros(width))

    def forward(self, x, variance, branch, condition):
        return kalman_update(x, variance, branch, condition, self.raw_q, self.bias, self.slope)


class KalmanBlock(nn.Module):
    def __init__(self, original, width):
        super().__init__()
        self.attn = original.attn
        self.mlp = original.mlp
        self.attn_connection = self.mlp_connection = None
        self.attn_noise = DiagonalNoise(width)
        self.mlp_noise = DiagonalNoise(width)

    def forward(self, x, variance, cos, sin, kv_cache=None):
        condition = norm(x)
        branch = self.attn(condition, None, (cos, sin), (-1, 0), kv_cache)
        x, variance = self.attn_noise(x, variance, branch, condition)
        condition = norm(x)
        branch = self.mlp(condition)
        return self.mlp_noise(x, variance, branch, condition)


class KalmanResidualModel(ConnectionModel):
    def __init__(self, config):
        from dataclasses import replace
        assert config.variant == 'kalman_residual'
        # Construct and initialize exactly the original backbone first. Noise
        # parameters consume no RNG and preserve all backbone parameter names.
        super().__init__(replace(config, variant='baseline'))
        self.config = config
        self.blocks = nn.ModuleList([KalmanBlock(b, config.width) for b in self.blocks])

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks:
            self.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True) for b in self.blocks]
        return self

    def hidden(self, idx, kv_cache=None):
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + idx.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        if kv_cache is not None and self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
            raise ValueError('Activation checkpointing cannot mutate a KV cache')
        x = norm(self.wte(idx))
        # Only depth carries state. Independent samples/forwards start fresh;
        # cached decoding needs no additional temporal state beyond original KV.
        variance = torch.full_like(x, INITIAL_VARIANCE, dtype=torch.float32)
        cos, sin = self.cos[:, start:start+idx.shape[1]], self.sin[:, start:start+idx.shape[1]]
        for i,b in enumerate(self.blocks):
            fn = self.compiled_steps[i] if self.compiled_steps is not None else b.forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                x,variance = checkpoint(fn,x,variance,cos,sin,use_reentrant=False)
            else:
                x,variance = fn(x,variance,cos,sin,kv_cache)
        return norm(x)
