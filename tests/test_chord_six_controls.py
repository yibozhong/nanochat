"""Check the active Chord controllers, initial velocities and optimizer groups."""
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.chord_token_controls import reference_write
from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.momentum import residual_update
from test_chord_soft_scalar import scalar_config
from test_momentum import gpu, inputs, make


@pytest.mark.parametrize('controls', ['static', 'learned_dynamic'])
@pytest.mark.parametrize('init', ['zero', 'token', 'projected_h0'])
def test_six_controls_and_initial_velocity(controls, init):
    cfg = replace(scalar_config(), chord_six_controls=controls, velocity_init=init,
                  chord_update_based=True, residual_projection_init='zero')
    assert ConnectionConfig.from_checkpoint(asdict(cfg)) == cfg
    model = make(cfg, perturb=False)
    ids, targets = inputs()
    x = model.wte(ids)
    h, v = model.initial_state(x, ids)
    assert h.dtype == v.dtype == torch.bfloat16
    assert (model.velocity_embedding is not None) == (init == 'token')
    assert (model.velocity_projection is not None) == (init == 'projected_h0')
    assert model.velocity_position_embedding is None
    opt, _ = optimizer_for(model)
    groups = {id(p): g for g in opt.param_groups for p in g['params']}
    if init == 'zero':
        assert torch.count_nonzero(v) == 0
    else:
        unit = torch.nn.functional.rms_norm(x.float(), (cfg.width,), eps=0.)
        if init == 'token':
            raw = model.velocity_embedding(ids).float()
            assert groups[id(model.velocity_embedding.weight)] is groups[id(model.wte.weight)]
        else:
            raw = model.velocity_projection(unit.to(x.dtype)).float()
            assert groups[id(model.velocity_projection.weight)] is groups[id(model.blocks[0].attn.c_q.weight)]
        expected = raw - unit * (unit*raw).mean(-1, True)
        torch.testing.assert_close(v.float(), expected, atol=.0003, rtol=.008)
    sites = [s for b in model.blocks for s in (b.attn_residual, b.mlp_residual)]
    routing = groups[id(sites[0].mu)]
    assert routing['kind'] == 'adamw' and routing['weight_decay'] == 0
    assert routing['initial_lr'] == pytest.approx(.008 * .25 * (cfg.width/768)**-.5)
    assert all(groups[id(p)] is routing for s in sites for p in s.parameters())
    for s in sites[:-1]:
        assert torch.equal(s.writes, torch.full_like(s.writes, .5))
        assert s.writes.numel() == 2 and s.raw_beta is not None
    assert sites[-1].writes is sites[-1].raw_beta is None
    loss = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(b.c_proj.weight.grad.norm() > 0 for block in model.blocks for b in (block.attn, block.mlp))


def test_static_fused_write_matches_token_equation_and_gradients():
    cfg = replace(scalar_config(), chord_six_controls='static', chord_update_based=True)
    site = residual_update(cfg).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.7)
        site.writes.copy_(site.writes.new_tensor([.25, .8]))
    h = torch.nn.functional.normalize(torch.randn(12, 128, device='cuda')) * 128**.5
    v = torch.randn_like(h)
    v -= h * (h*v).mean(-1, True)
    v[0].zero_()
    f = torch.randn_like(h)
    f[1].zero_()
    f[2:5] *= .01
    h, v, f = (x.bfloat16().requires_grad_() for x in (h, v, f))
    params = (site.raw, site.raw_parallel, site.raw_beta, site.writes)
    coeff = torch.stack((site.mu[0], site.alpha(), site.step_size(site.raw_parallel), site.beta(), *site.writes))
    expected = reference_write(h.double(), v.double(), f.double(), coeff.double(), site.rms_floor, False, site.soft_tau)
    actual = site(h, v, f)
    probe = [torch.randn_like(n) for n in actual]
    inputs = (h, v, f, *params)
    got = torch.autograd.grad(actual, inputs, probe)
    ref = torch.autograd.grad(expected, inputs, [x.double() for x in probe])
    for x, y in zip((*actual, *got), (*expected, *ref)):
        error = (x.double()-y.double()).norm()
        assert error <= .015 * y.double().norm() + 2e-5
    assert torch.all(ref[-1].abs() > 0)
