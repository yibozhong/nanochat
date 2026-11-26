"""Independent equations and first NTP gradient for the paired zero-v0 ablations."""
import torch
import torch.nn.functional as F

from nanochat.connection_cost import model_cost
from nanochat.connection_model import optimizer_for
from test_momentum import gpu, inputs
from test_scalar_components import small


def check_model(model):
    ids, targets = inputs()
    opt, _ = optimizer_for(model)
    controls = [p for b in model.blocks for s in (b.attn_residual, b.mlp_residual) for p in s.parameters()]
    group, = [g for g in opt.param_groups if any(p is controls[0] for p in g['params'])]
    assert {id(p) for p in group['params']} == {id(p) for p in controls}
    assert group['kind'] == 'adamw' and group['weight_decay'] == 0
    assert sum(p.numel() for p in controls) == model_cost(model.config)['additional_parameters']
    loss = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(branch.c_proj.weight.grad.norm() > 0 for b in model.blocks for branch in (b.attn, b.mlp))


def test_sphere_only():
    model = small(residual_dynamics='ngpt', chord_soft_force_tau=.05, ngpt_force_rms_floor=1e-6,
                  velocity_init='zero', chord_component_gates=True)
    check_model(model)
    assert not any('velocity' in name for name, _ in model.named_parameters())
    site = model.blocks[0].attn_residual
    with torch.no_grad():
        site.token_weight.normal_(std=.3)
        site.mix_scale.fill_(.5)
    h = F.normalize(torch.randn(8, 128, device='cuda'), dim=-1).mul(128**.5).bfloat16()
    f = torch.randn_like(h, requires_grad=True)
    hn = F.rms_norm(h.float(), (128,), eps=1e-6)
    a = F.softplus(site.raw + .5 * (hn*site.token_weight).sum(-1, True))
    expected = h.float() + a * f.float() / (f.float().square().mean(-1, True)+.05**2).sqrt()
    expected = F.normalize(expected, dim=-1) * 128**.5
    out = site(h, f)
    torch.testing.assert_close(out.float(), expected, atol=.016, rtol=.004)
    assert a.std() > 0 and out.dtype == torch.bfloat16
    assert torch.autograd.grad(out.float().sum(), site.token_weight)[0].norm() > 0


def test_lookahead_only():
    model = small(residual_dynamics='lookahead_only', ngpt_velocity_mode='ambient_learned_beta',
                  chord_velocity_geometry='ambient', velocity_init='zero', chord_component_gates=True)
    check_model(model)
    ids, _ = inputs()
    h, rest = model.initial_state(model.wte(ids), ids)
    assert rest.count_nonzero() == 0 and model.velocity_embedding is model.velocity_projection is None
    site = model.blocks[0].attn_residual
    with torch.no_grad():
        site.token_weight.normal_(std=.1)
        site.mix_scales.copy_(site.mix_scales.new_tensor([.3, .2]))
    v, force = torch.randn_like(h), torch.randn_like(h)
    v[0, 0].zero_()
    q, z = site.read(h, v)
    new, vel = site.write(h, v, force, z)
    hn = F.rms_norm(h.float(), (h.size(-1),), eps=1e-6)
    zref = F.linear(hn, site.token_weight) * h.new_tensor([.3,.2,.2,.2], dtype=torch.float32)
    qref = (h.float() + (site.mu + zref[..., :1])*v.float()).bfloat16()
    torch.testing.assert_close(q, F.rms_norm(qref, (h.size(-1),)), atol=.02, rtol=.01)
    torch.testing.assert_close(new, h + force, atol=0, rtol=0)
    d = new.float() - h.float()
    axis = v.float()
    along = axis * ((d*axis).sum(-1, True)/axis.square().sum(-1, True).clamp_min(1e-12))
    beta = (site.raw_beta + zref[..., 1:2]).sigmoid()
    gp, gt = (site.writes + zref[..., 2:]).split(1, -1)
    expected = beta*axis + gp*along + gt*(d-along)
    torch.testing.assert_close(vel.float(), expected, atol=.016, rtol=.006)
    assert torch.autograd.grad(vel.float().sum(), site.token_weight)[0].norm() > 0
    end = model.blocks[-1].mlp_residual
    assert end.token_weight.size(0) == 1 and end.writes is None
