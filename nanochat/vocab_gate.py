"""Full dense, context-dependent vocabulary gate with ordinary CE gradients.

z = W h; g = G h; logits = softcap(z * (1 + alpha*tanh(g))).
W and G are independent V x D matrices, packed into one (2V) x D GEMM.
There is no low-rank factor, auxiliary objective, or backward-only estimator.
"""
import torch
from torch.nn import functional as F


def install_gate(model):
    from nanochat.gpt import Linear
    old = model.lm_head
    # Preserve the entire baseline initialization and the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        packed = Linear(old.in_features, 2 * old.out_features, bias=False)
        with torch.no_grad():
            packed.weight[:old.out_features].copy_(old.weight)
            packed.weight[old.out_features:].zero_()
    model.lm_head = packed


def gate_logits(h, weight, alpha=.5, softcap=15.):
    z, g = F.linear(h, weight.to(h.dtype)).float().chunk(2, dim=-1)
    logits = z * (1 + alpha * torch.tanh(g))
    return softcap * torch.tanh(logits / softcap)


def gate_loss(h, weight, targets, alpha=.5, backend='triton', chunk_size=2048):
    if backend == 'reference':
        return F.cross_entropy(gate_logits(h, weight, alpha).reshape(-1, weight.shape[0] // 2), targets.reshape(-1))
    if backend != 'triton':
        raise ValueError('Unknown vocabulary gate backend')
    from nanochat.vocab_gate_triton import linear_gate_cross_entropy
    return linear_gate_cross_entropy(h, weight, targets, alpha=alpha, chunk_size=chunk_size)
