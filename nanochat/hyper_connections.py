"""Per-token HC and mHC residual routing, independent of the branch computation.

HC: Zhu et al., arXiv:2409.19606v3, equations 10--14 and Appendix J.
mHC: DeepSeek official V4/DeepGEMM PyTorch reference (pinned in the protocol).
Sinkhorn adaptation copyright (c) 2025 DeepSeek, MIT; see
research/connections/references/DeepSeek-LICENSE.
See research/connections/README.md for initialization and precision choices.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def routing_dtype(x):
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def sinkhorn(logits, iterations=20, eps=1e-6):
    """Official DeepGEMM softmax/epsilon/row-column order, with autograd.

    Finite iterations and epsilon give approximate, not exact, double stochasticity.
    """
    if iterations < 1:
        raise ValueError('Sinkhorn needs at least one iteration')
    value = logits.to(routing_dtype(logits)).softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


class HyperConnection(nn.Module):
    """Dynamic HC: per-stream LayerNorm, tanh dynamics, unconstrained mixing."""
    def __init__(self, width, streams, branch_index):
        super().__init__()
        self.width, self.streams, self.branch_index = width, streams, branch_index
        self.route_norm = nn.LayerNorm(width, eps=1e-5)
        self.alpha = nn.Parameter(torch.empty(streams, streams + 1))
        self.beta = nn.Parameter(torch.empty(streams))
        self.phi_alpha = nn.Parameter(torch.empty(width, streams + 1))
        self.phi_beta = nn.Parameter(torch.empty(width))
        self.scale_alpha = nn.Parameter(torch.empty(()))
        self.scale_beta = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.alpha.zero_()
        self.alpha[:, 1:].copy_(torch.eye(self.streams, device=self.alpha.device))
        self.alpha[self.branch_index % self.streams, 0] = 1
        self.beta.fill_(1)
        self.phi_alpha.zero_(); self.phi_beta.zero_()
        self.scale_alpha.fill_(.01); self.scale_beta.fill_(.01)
        self.route_norm.reset_parameters()

    def coefficients(self, x):
        xf = x.to(routing_dtype(x))
        z = self.route_norm(xf)
        alpha = self.alpha + self.scale_alpha * torch.tanh(z @ self.phi_alpha)
        post = self.beta + self.scale_beta * torch.tanh(z @ self.phi_beta)
        return alpha[..., 0], post, alpha[..., 1:].transpose(-1, -2)

    def read(self, x):
        pre, post, residual = self.coefficients(x)
        xf = x.to(pre.dtype)
        branch = (pre.unsqueeze(-2) @ xf).squeeze(-2).to(x.dtype)
        skip = residual @ xf
        return branch, skip, post

    @staticmethod
    def write(output, skip, post):
        return (skip + post.unsqueeze(-1) * output.to(post.dtype).unsqueeze(-2)).to(output.dtype)


class ManifoldHyperConnection(HyperConnection):
    """mHC with flattened-stream RMS input and 20-step Sinkhorn projection."""
    def __init__(self, width, streams, branch_index, iterations=20):
        nn.Module.__init__(self)
        self.width, self.streams = width, streams
        self.iterations = iterations
        self.phi = nn.Parameter(torch.empty(2 * streams + streams * streams, streams * width))
        self.bias = nn.Parameter(torch.empty(2 * streams + streams * streams))
        self.scales = nn.Parameter(torch.empty(3))
        # Initialize later, after common backbone weights, for identical seeds.

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1 / math.sqrt(self.streams * self.width))
        # Paper fixes the gates and projections, but does not specify these biases.
        # Mean-like pre read, unit post write, softly identity-biased residual.
        n = self.streams
        self.bias[:n].fill_(-math.log(max(n - 1, 1)))
        self.bias[n:2*n].zero_()
        self.bias[2*n:].view(n,n).copy_(torch.eye(n, device=self.phi.device))
        self.scales.fill_(.01)

    def coefficients(self, x):
        n = self.streams
        xf = x.to(routing_dtype(x)).flatten(-2)
        # Official code applies RMS factor AFTER the projection.
        raw = F.linear(xf, self.phi) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        pre = torch.sigmoid(self.bias[:n] + self.scales[0] * raw[..., :n]) + 1e-6
        post = 2 * torch.sigmoid(self.bias[n:2*n] + self.scales[1] * raw[..., n:2*n])
        logits = (self.bias[2*n:] + self.scales[2] * raw[..., 2*n:]).unflatten(-1, (n,n))
        # Official comb is [source,destination]. read() expects [destination,source].
        return pre, post, sinkhorn(logits, self.iterations).transpose(-1,-2)


class ManifoldHyperHead(nn.Module):
    """DeepSeek V4's official learned final stream reduction, before LM norm/head."""
    def __init__(self, width, streams):
        super().__init__()
        self.width, self.streams = width, streams
        self.phi = nn.Parameter(torch.empty(streams, streams*width))
        self.scale = nn.Parameter(torch.empty(1))
        self.bias = nn.Parameter(torch.empty(streams))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.phi, std=1/math.sqrt(self.streams*self.width))
        self.scale.fill_(.01)
        self.bias.fill_(-math.log(max(self.streams-1, 1)))

    def forward(self, x):
        xf = x.to(routing_dtype(x))
        flat = xf.flatten(-2)
        raw = F.linear(flat, self.phi) * torch.rsqrt(flat.square().mean(-1, keepdim=True)+1e-6)
        pre = torch.sigmoid(raw*self.scale+self.bias)+1e-6
        return (pre.unsqueeze(-1)*xf).sum(-2).to(x.dtype)
