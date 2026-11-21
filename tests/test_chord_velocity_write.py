"""Direct scalar velocity writes in the fused, project-only Chord residual."""
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
                            chord_velocity_write=True)


def unit(x):
    return x * x.square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()


def reference(h, v, f, raw_perp, raw_parallel, raw_beta, raw_rho):
    delta = unit(f) - h
    along = (delta*v).mean(-1, keepdim=True) / v.square().mean(-1, keepdim=True).clamp_min(1e-12)
    parallel = along * v
    step = F.softplus(raw_perp)*(delta - parallel) + F.softplus(raw_parallel)*parallel
    nh = unit(h + step + raw_rho.tanh()*v)
    mix = raw_beta.sigmoid()*v + (1-raw_beta.sigmoid())*(nh-h)
    return nh, mix - (nh*mix).mean(-1, keepdim=True)*nh


@pytest.mark.parametrize('width,dtype,raw_rho', [(128, torch.float32, -.45), (512, torch.bfloat16, .7)])
def test_write_and_gradients(setting, width, dtype, raw_rho):
    site = LookaheadResidual(replace(setting, width=width, heads=width//128)).cuda()
    with torch.no_grad():
        site.raw_parallel.fill_(-.9)
        site.raw_beta.fill_(.4)
        site.raw_rho.fill_(raw_rho)
    h = unit(torch.randn(37, width, device='cuda'))
    v = torch.randn_like(h)
    v -= (h*v).mean(-1, keepdim=True)*h
    v[0].zero_()
    args = tuple(x.to(dtype).requires_grad_() for x in (h, v, torch.randn_like(h)))
    args += (site.raw, site.raw_parallel, site.raw_beta, site.raw_rho)
    actual = site(*args[:3])
    doubles = tuple(x.detach().double().requires_grad_() for x in args)
    expected = reference(*doubles)
    dy = tuple(torch.randn_like(x) for x in actual)
    got = torch.autograd.grad(actual, args, dy)
    want = torch.autograd.grad(expected, doubles, tuple(x.double() for x in dy))
    tol = .008 if dtype == torch.bfloat16 else 1e-4
    for x, y in zip((*actual, *got), (*expected, *want)):
        assert (x.double()-y).norm() <= tol*y.norm()+1e-5


def test_zero_rho_and_velocity_only_motion(setting):
    site = LookaheadResidual(setting).cuda()
    control = LookaheadResidual(replace(setting, chord_velocity_write=False)).cuda()
    h = torch.ones(3, setting.width, device='cuda')
    v = torch.ones_like(h)
    v[:, ::2] = -1
    f = torch.randn_like(h)
    assert site.raw_rho.shape == (1,) and site.raw_rho.item() == 0
    for actual, base in zip(site(h,v,f), control(h,v,f)):
        torch.testing.assert_close(actual, base, atol=0, rtol=0)
    # With target == h, only rho*v can move the hidden, including at the final FFN.
    force = h
    at_rest, _ = site(h,v,force)
    torch.testing.assert_close(at_rest, h, atol=2e-7, rtol=0)
    derivative, = torch.autograd.grad((at_rest*v).sum(), (site.raw_rho,))
    torch.testing.assert_close(derivative, v.square().sum().reshape_as(derivative))
    for raw in (-.5, .5):
        with torch.no_grad():
            site.raw_rho.fill_(raw)
            nh, nv = site(h,v,force)
            expected = unit(h + site.raw_rho.tanh()*v)
            torch.testing.assert_close(nh, expected)
            torch.testing.assert_close((nh*nv).mean(-1), torch.zeros_like(nh[:,0]), atol=2e-6, rtol=0)


def test_compiled_cache_causality_and_gradients(setting):
    check_compiled_model(setting, False)
