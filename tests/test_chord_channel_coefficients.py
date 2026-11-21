"""GPU regressions for the project-only Chord write with channel-wise coefficients."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nanochat.connection_model import ConnectionConfig
from nanochat.lookahead import LookaheadResidual
from test_chord_fused_branch import test_fused_model as check_compiled_model
from test_momentum import gpu, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            residual_projection_init='normal', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True,
                            channel_chord_coefficients=True)


def reference(h, v, force, raw_perp, raw_parallel, raw_beta):
    def normalize(x):
        return x * x.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
    delta = normalize(force) - h
    along = (delta*v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    parallel = along * v
    h_next = normalize(h + F.softplus(raw_perp)*(delta - parallel) + F.softplus(raw_parallel)*parallel)
    beta = raw_beta.sigmoid()
    mix = beta*v + (1-beta)*(h_next-h)
    return h_next, mix - (mix*h_next).mean(-1, keepdim=True)*h_next


@pytest.mark.parametrize('width,dtype', [(128, torch.float32), (512, torch.bfloat16)])
def test_fused_channel_write_and_vjp(setting, width, dtype):
    site = LookaheadResidual(replace(setting, width=width, heads=width//128)).cuda()
    with torch.no_grad():
        for param in (site.raw, site.raw_parallel, site.raw_beta):
            param.add_(.7 * torch.randn_like(param))
    h, v, force = torch.randn(3, 37, width, device='cuda').unbind()
    h = h * h.square().mean(-1, keepdim=True).rsqrt()
    v = v - (h*v).mean(-1, keepdim=True)*h
    v[0].zero_()
    leaves = tuple(x.to(dtype).requires_grad_() for x in (h, v, force)) + (site.raw, site.raw_parallel, site.raw_beta)
    actual = site(*leaves[:3])
    doubles = tuple(x.detach().double().requires_grad_() for x in leaves)
    expected = reference(*doubles)
    upstream = tuple(torch.randn_like(x) for x in actual)
    got = torch.autograd.grad(actual, leaves, upstream)
    want = torch.autograd.grad(expected, doubles, tuple(x.double() for x in upstream))
    tol = .008 if dtype == torch.bfloat16 else 1e-4
    for a, b in zip((*actual, *got), (*expected, *want)):
        assert a.shape == b.shape
        assert (a.double()-b).norm() <= tol*b.norm() + 1e-7
    assert len({p.data_ptr() for p in leaves[-3:]}) == 3


def test_equal_initial_coefficients_match_scalar(setting):
    scalar = LookaheadResidual(replace(setting, channel_chord_coefficients=False)).cuda()
    channel = LookaheadResidual(setting).cuda()
    h, v, force = torch.randn(3, 37, setting.width, device='cuda').unbind()
    outputs = [site(h, v, force) for site in (scalar, channel)]
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    upstream = tuple(torch.randn_like(x) for x in outputs[0])
    shared, independent = [torch.autograd.grad(out, (site.raw, site.raw_parallel, site.raw_beta), upstream)
                           for site, out in zip((scalar, channel), outputs)]
    for a, b in zip(shared, independent):
        assert b.shape == (setting.width,)
        torch.testing.assert_close(a.squeeze(), b.sum(), rtol=1e-4, atol=2e-5)


def test_compiled_cache_causality_and_gradients(setting):
    check_compiled_model(setting, False)
