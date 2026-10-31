"""New-width gates; existing kernel/model tests are already archived and passed."""

import pytest
import torch
from torch.nn import functional as F

from test_mhc_nvidia_full import test_nvidia_full_pipeline_fp64 as pipeline_check
from test_mhc_nvidia_full import relative, record
from nanochat.nvidia_connections import NvidiaHyperHead
from nanochat.connection_model import ConnectionConfig, ConnectionModel
from scripts.connection_scale_profile import configure_attention


@pytest.mark.parametrize('width', [2048, 2560, 3072])
def test_new_width_pipeline(width):
    pipeline_check(width, False)


@pytest.mark.parametrize('width', [2048, 2560, 3072])
def test_new_width_head(width):
    torch.manual_seed(84)
    m = NvidiaHyperHead(width, 4, use_tf32=True).cuda(); m.reset_parameters()
    x = torch.randn(1, 128, width, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    out = m(x); probe = torch.randn_like(out).float() / out.numel()
    actual = torch.autograd.grad((out.float() * probe).sum(), (x, *m.parameters()))
    xr = x.detach().double().requires_grad_()
    params = [v.detach().double().requires_grad_() for v in m.parameters()]
    phi, scale, bias = params; flat = xr.flatten(-2)
    raw = F.linear(flat / (flat.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps).sqrt(), phi)
    expected = (xr * (raw * scale + bias).sigmoid().unsqueeze(-2)).sum(-1)
    reference = torch.autograd.grad((expected * probe.double()).sum(), (xr, *params))
    errors = dict(output=relative(out, expected), gradients=[relative(a, b) for a, b in zip(actual, reference)])
    record(f'scale-head-d{width}', errors)
    assert errors['output'] < .005 and max(errors['gradients']) < .02, errors


@pytest.mark.parametrize('width', [2048, 2560, 3072])
def test_new_width_2k_causality_and_cce(width):
    configure_attention(); torch.manual_seed(42)
    model = ConnectionModel(ConnectionConfig(variant='mhc', width=width, heads=width//128,
        depth=2, vocab_size=256, connection_backend='nvidia', fused_connection_grad=False)).prepare()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=.01); block.mlp.c_proj.weight.normal_(std=.01)
    ids = torch.randint(0, 256, (1, 2048), device='cuda')
    changed = ids.clone(); changed[:, 1024:] = (changed[:, 1024:] + 17) % 256
    with torch.no_grad():
        a, b = model(ids), model(changed)
        torch.testing.assert_close(a[:, :1024], b[:, :1024], atol=0, rtol=0)
    x, y = ids[:, :128].contiguous(), ids[:, 1:129].contiguous()
    loss, dense = model(x, y), model(x, y, dense=True)
    torch.testing.assert_close(loss, dense, atol=2e-4, rtol=2e-4)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
