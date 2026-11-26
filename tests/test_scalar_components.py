"""Equations and first backward for the BF16 Scalar Chord study from rest."""
from dataclasses import asdict

import torch

from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.packed_training import seed_all
from test_momentum import gpu


def small(**options):
    config = ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                              streams=1, chord_state_dtype='bfloat16', residual_projection_init='zero', **options)
    assert ConnectionConfig.from_checkpoint(asdict(config)) == config
    seed_all(42)
    return build_connection_model(config).prepare()


def test_from_rest():
    model = small(residual_dynamics='ngpt_lookahead', velocity_init='zero', chord_soft_force_tau=.05,
                  ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
                  fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    ids = torch.arange(16, device='cuda')[None]
    h, v = model.initial_state(model.wte(ids), ids)
    assert model.velocity_embedding is model.velocity_position_embedding is model.velocity_projection is None
    assert h.dtype == v.dtype == torch.bfloat16 and v.count_nonzero() == 0
    site = model.blocks[0].attn_residual
    assert set(dict(site.named_parameters())) == {'raw', 'raw_parallel', 'raw_beta', 'mu'}
    assert torch.equal(site.query(h, v), h)
    force = torch.randn_like(h, requires_grad=True)
    n, velocity = site(h, v, force)
    assert velocity.norm() > 0 and (n.float()*velocity.float()).mean(-1).abs().max() < 1e-3
    torch.testing.assert_close(n.float().square().mean(-1), torch.ones_like(n[..., 0]).float(), atol=.01, rtol=0)
    n.float().sum().backward()
    assert torch.isfinite(force.grad).all() and force.grad.norm() > 0


def test_sphere_only():
    model = small(residual_dynamics='ngpt', chord_soft_force_tau=.05)
    assert not any('velocity' in name for name, _ in model.named_parameters())
    site = model.blocks[0].attn_residual
    assert set(dict(site.named_parameters())) == {'raw'}
    ids = torch.arange(16, device='cuda')[None]
    h = torch.ones(2, 128, dtype=torch.bfloat16, device='cuda')
    force = torch.zeros_like(h, requires_grad=True)
    new = site(h, force)
    assert new.dtype == h.dtype and torch.equal(new, h)
    probe = torch.ones_like(h)
    probe[:, ::2] = -1
    grad, = torch.autograd.grad(new, force, probe)
    torch.testing.assert_close(grad, (probe.float()/.95).to(h.dtype))
    force = torch.randn_like(h)
    target = force.float() / (force.float().square().mean(-1, True) + .05**2).sqrt()
    x = h.float() + .05 * (target - h.float())
    torch.testing.assert_close(site(h, force), (x / x.square().mean(-1, True).sqrt()).to(h.dtype))
    with torch.no_grad():
        _, layers = model.backbone(h[None], return_layers=True)
    assert layers.dtype == torch.bfloat16
    assert torch.isfinite(model(ids, ids))


def test_lookahead_only():
    model = small(residual_dynamics='lookahead_only', ngpt_velocity_mode='ambient_learned_beta',
                  chord_velocity_geometry='ambient', velocity_init='zero')
    ids = torch.arange(16, device='cuda')[None]
    h, v = model.initial_state(model.wte(ids), ids)
    assert h.dtype == v.dtype == torch.bfloat16 and model.velocity_position_embedding is None
    assert model.velocity_embedding is model.velocity_projection is None and v.count_nonzero() == 0
    site = model.blocks[0].attn_residual
    assert set(dict(site.named_parameters())) == {'mu', 'raw_beta'}
    assert site.mu.item() == 0 and site.raw_beta.sigmoid().item() == .5
    assert model.blocks[-1].mlp_residual.raw_beta is None
    with torch.no_grad():
        site.mu.fill_(.25)
    # Radial velocity must survive the Euclidean recurrence.
    v = 2*h
    torch.testing.assert_close(site.query(h, v), h.float() + .25*v.float())
    force = h.clone()
    new, updated = site(h, v, force)
    assert torch.equal(new, h + force) and updated.dtype == v.dtype
    expected = .5*v.float() + .5*(new.float()-h.float())
    torch.testing.assert_close(updated, expected.to(v.dtype))
    updated.float().square().mean().backward()
    assert torch.isfinite(site.raw_beta.grad).all() and site.raw_beta.grad.abs().item() > 0
