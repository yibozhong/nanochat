"""Narrow vocabulary readout from the final FFN's existing expanded features.

The down projection and vocabulary projection each remain ONE matrix multiply.
Only the main residual determines RMS scaling, preserving the baseline function
when the new down-projection rows are zero. No auxiliary loss/custom gradient.
"""
import math

import torch
from torch.nn import functional as F


def readout_reference(residual, projected):
    """Independent eager oracle, including the BF16 residual-add boundary."""
    d = residual.shape[-1]
    main = residual + projected[..., :d]
    scale = torch.rsqrt(main.float().square().mean(-1, keepdim=True) + 2**-23)
    return torch.cat((main.float() * scale, projected[..., d:].float() * scale), -1).to(residual.dtype)


def readout(residual, projected, backend='triton'):
    if backend == 'reference':
        return readout_reference(residual, projected)
    from nanochat.ffn_readout_triton import fused_readout
    return fused_readout(residual, projected)


def install_readout(model):
    """Extend packed matrices after baseline initialization, without changing RNG."""
    from nanochat.gpt import Linear
    d, r = model.config.width, model.config.ffn_readout_width
    block = model.blocks[-1]
    with torch.random.fork_rng(devices=[]):
        down = Linear(4*d, d+r, bias=False)
        head = Linear(d+r, model.config.vocab_size, bias=False)
        with torch.no_grad():
            down.weight.zero_()
            down.weight[:d].copy_(block.mlp.c_proj.weight)
            head.weight[:, :d].copy_(model.lm_head.weight)
            torch.nn.init.normal_(head.weight[:, d:], std=.001)
        block.mlp.c_proj = down
        model.lm_head = head
    block.ffn_readout_width = r
    block.ffn_readout_backend = model.config.ffn_readout_backend


@torch.no_grad()
def initialize_gradmax(model, idx, targets, max_tokens=256, seed=42):
    """One bounded calibration; randomized SVD of missing-error/FFN covariance.

LM-specific adaptation of GradMax, NOT the original paper's algorithm:
      C = (I - Q_W Q_W.T) G.T S / N
    G is the actual pre-softcap logit gradient; S is the expanded FFN feature
    divided by the main residual RMS. We factor C implicitly, never allocate
    the V x 4D covariance or a full-sequence vocabulary gradient. Calibration
    only changes new head columns; zero new down rows preserve model outputs.
    Call before creating an optimizer; never call when restoring a checkpoint.
    """
    d, r = model.config.width, model.config.ffn_readout_width
    if not r or model.config.vocab_size <= d+r or max_tokens < r:
        raise ValueError('GradMax needs a branch, V > D+R, and at least R calibration tokens')
    if torch.count_nonzero(model.blocks[-1].mlp.c_proj.weight[d:]):
        raise ValueError('GradMax initialization requires untouched zero branch rows')
    n = min(max_tokens, idx.numel())
    if n < r:
        raise ValueError('Insufficient calibration tokens')
    # Ordinary model forward captures the exact feature/dtype boundaries.
    captured = {}
    block_hook = model.blocks[-1].register_forward_pre_hook(
        lambda module, args: captured.update(residual_input=args[0].detach()))
    hook = model.blocks[-1].mlp.c_proj.register_forward_pre_hook(
        lambda module, args: captured.update(features=args[0].detach()))
    compiled = model.compiled_steps
    model.compiled_steps = None
    try:
        h = model.hidden(idx)
    finally:
        hook.remove()
        block_hook.remove()
        model.compiled_steps = compiled
    # Deterministic evenly spaced positions; no change to training data/RNG.
    pos = torch.linspace(0, idx.numel()-1, n, device=idx.device).long()
    features = captured['features'].reshape(-1, 4*d)[pos].float()
    # Recompute only the final attention to recover the unnormalized residual.
    block = model.blocks[-1]
    x = captured['residual_input']
    cos, sin = model.cos[:, :idx.shape[1]], model.sin[:, :idx.shape[1]]
    x = x + block.attention_branch(x, cos, sin)
    main = x + block.mlp.c_proj(captured['features'])[..., :d]
    inv = torch.rsqrt(main.reshape(-1, d)[pos].float().square().mean(-1, keepdim=True) + 2**-23)
    features *= inv
    dtype = h.dtype
    effective_w = model.lm_head.weight.to(dtype)
    logits = F.linear(h.reshape(-1, d+r)[pos], effective_w).float()
    capped = 15 * torch.tanh(logits / 15)
    errors = capped.softmax(-1)
    errors[torch.arange(n, device=idx.device), targets.flatten()[pos]] -= 1
    errors *= 1 - torch.tanh(logits / 15).square()
    # Project out the existing readout directions in FP32.
    q_w = torch.linalg.qr(effective_w[:, :d].float(), mode='reduced').Q
    def project(x):
        return x - q_w @ (q_w.T @ x)
    k = min(r+16, n, 4*d, model.config.vocab_size-d)
    devices = [idx.device.index if idx.device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        omega = torch.randn(4*d, k, device=idx.device)
        q = torch.linalg.qr(project(errors.T @ (features @ omega)), mode='reduced').Q
        # One power iteration, with reorthogonalization.
        q = torch.linalg.qr(project(errors.T @ (features @ (features.T @ (errors @ q)))), mode='reduced').Q
        small = (errors @ q).T @ features / n
        left, singular, _ = torch.linalg.svd(small, full_matrices=False)
        u = q @ left[:, :r]
    u *= .001 * math.sqrt(model.config.vocab_size)
    model.lm_head.weight[:, d:].copy_(u)
    return dict(method='projected_GradMax_inspired',tokens=n,rank=r,
                top_singular_value=singular[0].item(),last_singular_value=singular[r-1].item(),
                relative_overlap=(q_w.T @ u).norm().div(u.norm()).item(),
                new_down_rows_zero=True)
