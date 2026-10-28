"""Single-device, full-accumulation auxiliary/NTP parameter-gradient balancing.

The scalar is measured on the union of native block parameters, then applied to
ALL auxiliary parameter gradients. This preserves a detached scalar-weighted
joint objective, including paths into earlier blocks and embeddings. Hidden
step scalars and the LM head are excluded from the norm measurement.
"""
import math

import torch

from gradformer.core import prediction_loss


class NormBalancedBackward:
    def __init__(self, parameters, block_groups):
        self.parameters = list(parameters)
        index = {id(p): i for i, p in enumerate(self.parameters)}
        self.groups = [[index[id(p)] for p in group] for group in block_groups]
        self.block_indices = [i for group in self.groups for i in group]
        if not self.block_indices or len(set(self.block_indices)) != len(self.block_indices):
            raise ValueError("Norm measurement requires nonempty, disjoint block groups")
        self.reset()

    def reset(self):
        self.ntp_gradients = [None] * len(self.parameters)
        self.aux_gradients = [None] * len(self.parameters)
        self.microbatches = 0
        self.finished = False

    def backward(self, ntp, inputs, predictions, *, alpha=1., beta=1., accumulation=1):
        if self.finished or not inputs or accumulation < 1:
            raise ValueError("Reset the balancer before a new accumulated supervised step")
        targets = [None] * len(inputs)
        handles = []
        for i, h in enumerate(inputs):
            def capture(grad, index=i):
                targets[index] = grad.detach() * accumulation
            handles.append(h.register_hook(capture))
        for p, g in zip(self.parameters, self.ntp_gradients):
            p.grad = g
        try:
            (ntp / accumulation).backward(retain_graph=True)
        finally:
            for handle in handles:
                handle.remove()
        self.ntp_gradients = [p.grad for p in self.parameters]
        if any(g is None for g in targets):
            raise RuntimeError("A layer input is disconnected from NTP")
        for p, g in zip(self.parameters, self.aux_gradients):
            p.grad = g
        aux, metrics = prediction_loss(predictions, targets, alpha=alpha, beta=beta)
        (aux / accumulation).backward()
        self.aux_gradients = [p.grad for p in self.parameters]
        self.microbatches += 1
        return aux.detach(), metrics.detach()

    def norm(self, gradients, indices):
        zero = self.parameters[0].new_zeros(())
        return sum((gradients[i].norm().square() for i in indices
                    if gradients[i] is not None), zero).sqrt()

    def dot(self, indices):
        zero = self.parameters[0].new_zeros(())
        return sum((torch.dot(self.ntp_gradients[i].reshape(-1), self.aux_gradients[i].reshape(-1))
                    for i in indices if self.ntp_gradients[i] is not None
                    and self.aux_gradients[i] is not None), zero)

    @torch.no_grad()
    def finish(self, *, ratio=1., accumulation=1, diagnostics=True, eps=1e-12):
        if self.finished or self.microbatches != accumulation:
            raise ValueError("Balance exactly once after all microbatches")
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Target gradient norm ratio must be finite and positive")
        ntp_norm = self.norm(self.ntp_gradients, self.block_indices)
        aux_norm = self.norm(self.aux_gradients, self.block_indices)
        if not torch.isfinite(torch.stack((ntp_norm, aux_norm))).all().item():
            raise RuntimeError("Nonfinite parameter-gradient norm before balancing")
        active = (ntp_norm > eps) & (aux_norm > eps)
        scale = torch.where(active, ratio * ntp_norm / aux_norm.clamp_min(eps), torch.zeros_like(ntp_norm))
        if not torch.isfinite(scale).item():
            raise RuntimeError("Nonfinite dynamic auxiliary coefficient")
        # No EMA or cap: each nondegenerate macrostep matches the requested norm.
        for gradient in self.aux_gradients:
            if gradient is not None:
                gradient.mul_(scale)
        weighted_norm = self.norm(self.aux_gradients, self.block_indices)
        actual_ratio = weighted_norm / ntp_norm.clamp_min(eps)
        if active.item() and not torch.isclose(actual_ratio, actual_ratio.new_tensor(ratio), rtol=2e-5, atol=2e-6).item():
            raise RuntimeError("Actual weighted auxiliary gradient does not match requested ratio")
        result = {}
        if diagnostics:
            result = {"balance/weight": scale.item(), "balance/active": int(active.item()),
                      "balance/ntp_block_norm": ntp_norm.item(),
                      "balance/raw_aux_block_norm": aux_norm.item(),
                      "balance/weighted_aux_block_norm": weighted_norm.item(),
                      "balance/aux_to_ntp_ratio": actual_ratio.item(),
                      "balance/cosine_with_ntp": (self.dot(self.block_indices)
                                                   / (ntp_norm * weighted_norm).clamp_min(eps)).item()}
            for layer, indices in enumerate(self.groups):
                n, a = self.norm(self.ntp_gradients, indices), self.norm(self.aux_gradients, indices)
                result[f"balance/layer_{layer}/aux_to_ntp_ratio"] = (a / n.clamp_min(eps)).item()
                result[f"balance/layer_{layer}/cosine_with_ntp"] = (self.dot(indices) / (n * a).clamp_min(eps)).item()
        for p, n, a in zip(self.parameters, self.ntp_gradients, self.aux_gradients):
            p.grad = a if a is not None else n
            if a is not None and n is not None:
                p.grad.add_(n)
        self.finished = True
        # p.grad retains the merged buffers; release unneeded separate NTP ones.
        self.ntp_gradients = self.aux_gradients = []
        return result


def for_model(model, backend):
    blocks = model.transformer.h if backend == "nanochat" else model.layers
    return NormBalancedBackward(model.parameters(), [list(block.parameters()) for block in blocks])
