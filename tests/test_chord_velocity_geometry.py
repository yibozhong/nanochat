"""GPU checks for removing transport and the endpoint tangency constraint."""
from dataclasses import asdict

import pytest
import torch

from nanochat.chord_triton import split_boundary
from nanochat.connection_model import ConnectionConfig
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, inputs, make


@pytest.fixture(params=('project_only', 'ambient'))
def setting(request):
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            velocity_init='token_position', residual_projection_init='normal',
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry=request.param)


def test_write_and_derivatives_against_fp64(setting):
    geometry = setting.chord_velocity_geometry
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw.fill_(-.8)
        site.raw_parallel.fill_(.2)
        site.raw_beta.fill_(.4)
    h = normalize_residual(torch.randn(19, 512, device='cuda'))
    v = tangent_project(h, torch.randn_like(h))
    if geometry == 'ambient':
        v += .45*h
    v[0].zero_()
    f = torch.randn_like(h)
    args = tuple(a.requires_grad_() for a in (h, v, f)) + (site.raw, site.raw_parallel, site.raw_beta)
    actual = site(h, v, f)
    hd, vd, fd, a, b, decay = (x.detach().double().requires_grad_() for x in args)
    def unit(x):
        return x / x.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    direction = unit(fd) - hd
    along = (vd * direction).sum(-1, keepdim=True) / vd.square().sum(-1, keepdim=True).clamp_min(512e-12) * vd
    nh = unit(hd + torch.nn.functional.softplus(a)*(direction-along) + torch.nn.functional.softplus(b)*along)
    velocity = decay.sigmoid()*vd + (1-decay.sigmoid())*(nh-hd)
    if geometry == 'project_only':
        velocity = velocity - (nh*velocity).mean(-1, keepdim=True)*nh
        assert (actual[0]*actual[1]).mean(-1).abs().max() < 1e-6
    else:
        assert (actual[0]*actual[1]).mean(-1).abs().max() > .05
    reference = (nh, velocity)
    for x, y in zip(actual, reference):
        torch.testing.assert_close(x.double(), y, rtol=2e-5, atol=2e-6)
    up = tuple(torch.randn_like(x) for x in actual)
    got = torch.autograd.grad(actual, args, up)
    want = torch.autograd.grad(reference, (hd, vd, fd, a, b, decay), tuple(t.double() for t in up))
    for x, y in zip(got, want):
        assert (x.double()-y).norm() <= 3e-4*y.norm()+1e-5


def test_initial_velocity_and_spherical_read(setting):
    model = make(setting)
    tokens, _ = inputs()
    h, v = model.initial_state(model.wte(tokens), tokens)
    raw = model.velocity_embedding(tokens).float() + model.velocity_position_embedding.weight[:tokens.size(1)].float()
    expected = tangent_project(h, raw) if setting.chord_velocity_geometry == 'project_only' else raw
    torch.testing.assert_close(v, expected, rtol=0, atol=0)
    if setting.chord_velocity_geometry == 'ambient':
        v = v + 1.3*h
    site = model.blocks[0].attn_residual
    q = site.query(h, v)
    torch.testing.assert_close(q, normalize_residual(h+site.mu*v), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(q.square().mean(-1), torch.ones_like(q[..., 0]))
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting
    old = asdict(setting)
    del old['chord_velocity_geometry']
    assert ConnectionConfig.from_checkpoint(old).chord_velocity_geometry == 'parallel_transport'


def test_fused_boundary_query_and_derivatives(setting):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.7)
    mu = torch.tensor([-.63], device='cuda', requires_grad=True)
    h = normalize_residual(torch.randn(23, 512, device='cuda'))
    v = tangent_project(h, torch.randn_like(h))
    if setting.chord_velocity_geometry == 'ambient':
        v += .7*h
    f = torch.randn_like(h).bfloat16()
    args = tuple(x.requires_grad_() for x in (h, v, f)) + (site.raw, site.raw_parallel, site.raw_beta, mu)
    nh, nv = site.reference_write(h, v, f)
    offset = mu * nv
    if setting.chord_velocity_geometry == 'ambient':
        q = normalize_residual(nh+offset)
    else:
        q = (nh+offset) * torch.rsqrt(1+offset.square().mean(-1, keepdim=True))
    reference = (nh, nv, q.bfloat16())
    actual = split_boundary(site, h, v, f, mu)
    for x, y in zip(actual, reference):
        torch.testing.assert_close(x, y, rtol=.008 if x.dtype==torch.bfloat16 else 2e-5, atol=2e-6)
    up = tuple(torch.randn_like(x) for x in actual)
    got, want = (torch.autograd.grad(out, args, up) for out in (actual, reference))
    for x, y in zip(got, want):
        tol = .008 if x.dtype==torch.bfloat16 else 4e-4
        assert (x.float()-y.float()).norm() <= tol*y.float().norm()+2e-5
