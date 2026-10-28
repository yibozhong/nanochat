"""Mathematical core shared verbatim by the nanochat and llama2.c forks."""
import math
import torch
import torch.nn.functional as F


def initialize_steps(module, config, layers):
    if config.hidden_update != "baseline":
        if config.hidden_step_size <= 0:
            raise ValueError("hidden_step_size must be positive")
        rho = math.log(math.expm1(config.hidden_step_size))
        module.hidden_step_raw = torch.nn.Parameter(torch.full((layers,), rho))


def reset_steps(module, config):
    if config.hidden_update != "baseline":
        torch.nn.init.constant_(module.hidden_step_raw, math.log(math.expm1(config.hidden_step_size)))


def hidden_step(h, predicted, raw_step, state, layer, config):
    """A differentiable forward operator; no optimizer.step(), labels or true gradients.

    Adam moments belong to each sample/token and accumulate along depth only.
    At v=0 we choose the finite zero derivative of sqrt wrt v. This avoids
    sqrt'(0)*0 NaNs; the composite first-step map g/(abs(g)+eps) has derivative
    1/eps at g=0. Positive normal v values use the exact Adam denominator.
    """
    eta = F.softplus(raw_step)
    if config.hidden_update == "sgd":
        return h - eta * predicted, state
    if config.hidden_update != "adam":
        raise ValueError(config.hidden_update)
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    g = predicted.to(dtype)
    m, v = state if state is not None else (torch.zeros_like(g), torch.zeros_like(g))
    b1, b2 = config.hidden_beta1, config.hidden_beta2
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g.square()
    m_hat, v_hat = m / (1.0 - b1 ** (layer + 1)), v / (1.0 - b2 ** (layer + 1))
    root = torch.where(v_hat > 0, v_hat.clamp_min(torch.finfo(dtype).tiny).sqrt(), torch.zeros_like(v_hat))
    update = eta * m_hat / (root + config.hidden_eps)
    return h - update.to(h.dtype), (m, v)


def prediction_loss(predictions, targets, mask=None, alpha=1.0, beta=0.1, eps=1e-8):
    """Mean over hidden channels, valid tokens, then layers; raw NTP-mean targets.

    No target normalization. Target is detached even if callers forget.
    Zero target directions are excluded from cosine, but remain in the MSE.
    """
    terms, metrics = [], []
    for pred, target in zip(predictions, targets):
        dtype = torch.float64 if pred.dtype == torch.float64 else torch.float32
        p, g = pred.to(dtype), target.detach().to(dtype)
        valid = torch.ones_like(p[..., 0], dtype=torch.bool) if mask is None else mask
        weight = valid.to(dtype)
        count = weight.sum().clamp_min(1)
        mse = ((p - g).square().mean(dim=-1) * weight).sum() / count
        pn, gn = p.norm(dim=-1), g.norm(dim=-1)
        directional = valid & (gn > eps)
        cosine = (p * g).sum(dim=-1) / (pn.clamp_min(eps) * gn.clamp_min(eps))
        n_directional = directional.sum().clamp_min(1)
        cos_loss = ((1 - cosine) * directional).sum() / n_directional
        terms.append(alpha * mse + beta * cos_loss)
        metrics.append(torch.stack((mse.detach(), (cosine * directional).sum().detach() / n_directional,
                                    (pn * weight).sum().detach() / count,
                                    (gn * weight).sum().detach() / count,
                                    directional.sum().detach() / count)))
    return torch.stack(terms).mean(), torch.stack(metrics)


def joint_backward(ntp, inputs, predictions, mask=None, *, weight=0.1, alpha=1.0,
                   beta=0.1, accumulation=1, implementation="reuse", eps=1e-8):
    """Exactly NTP + weight*aux with stop-gradient targets, first derivatives only.

    Targets always differentiate the current microbatch's unscaled mean CE.
    Reuse computes NTP parameter gradients once and saves pure NTP cotangents
    via temporary hooks. Auxiliary gradients are then accumulated; hooks have
    already been removed. No optimizer update occurs between traversals.
    Distributed synchronization is performed by the caller after accumulation.
    """
    if not inputs or weight == 0:
        (ntp / accumulation).backward()
        return ntp.new_zeros(()), None
    if implementation == "reference":
        targets = torch.autograd.grad(ntp, inputs, retain_graph=True, create_graph=False)
        aux, metrics = prediction_loss(predictions, targets, mask, alpha, beta, eps)
        ((ntp + weight * aux) / accumulation).backward()
    elif implementation == "reuse":
        targets = [None] * len(inputs)
        handles = []
        for i, h in enumerate(inputs):
            def capture(grad, index=i):
                targets[index] = grad.detach() * accumulation
            handles.append(h.register_hook(capture))
        try:
            (ntp / accumulation).backward(retain_graph=True)
        finally:
            for handle in handles:
                handle.remove()
        if any(g is None for g in targets):
            raise RuntimeError("A layer input is disconnected from the final NTP loss")
        aux, metrics = prediction_loss(predictions, targets, mask, alpha, beta, eps)
        (weight * aux / accumulation).backward()
    else:
        raise ValueError(implementation)
    return aux.detach(), metrics.detach()
