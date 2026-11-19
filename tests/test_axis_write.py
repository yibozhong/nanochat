"""Projection, initialization and training checks for the Arc Residual write."""
from dataclasses import replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.engine import KVCache
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import seed_all
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_axis_write', ngpt_force_rms_floor=1e-6)


def test_zero_kappa_recovers_arc_and_receives_gradient(setting):
    baseline = make(replace(setting, ngpt_velocity_mode='lookahead_transport'))
    seed_all(42)
    model = make(setting)
    x, y = inputs()
    assert torch.equal(model(x), baseline(x))
    model(x, y, dense=True).backward()
    baseline(x, y, dense=True).backward()
    params = dict(model.named_parameters())
    for name, p in baseline.named_parameters():
        torch.testing.assert_close(params[name], p, rtol=0, atol=0)
        diff = (params[name].grad.float() - p.grad.float()).norm()
        assert diff <= .01 * p.grad.float().norm() + 1e-10, name
    opt, _ = optimizer_for(model)
    kappas = [p for n, p in params.items() if n.endswith('.kappa')]
    assert len(kappas) == 2*setting.depth
    for k in kappas:
        assert k.item() == 0 and torch.isfinite(k.grad).all() and k.grad.abs().item() > 0
        group, = [g for g in opt.param_groups if any(p is k for p in g['params'])]
        assert group['kind'] == 'adamw' and group['weight_decay'] == 0
    opt.step()
    assert all(k.item() != 0 for k in kappas)


@pytest.mark.parametrize('kappa', [-1., .7])
def test_projection_transport_and_derivatives(setting, kappa):
    site = LookaheadResidual(setting).cuda().double()
    h, v, force = torch.randn(3, 2, 7, device='cuda', dtype=torch.double)
    h = normalize_residual(h)
    v = tangent_project(h, v)
    gain = h.new_tensor([kappa], requires_grad=True)

    def run(h, v, f, gain):
        return torch.func.functional_call(site, {'kappa': gain}, (h, v, f))

    actual, velocity = run(h, v, force, gain)
    delta = normalize_residual(force) - h
    axis = torch.nn.functional.normalize(v, dim=-1)
    parallel = (delta * axis).sum(-1, keepdim=True) * axis
    expected = normalize_residual(h + site.alpha() * (delta + kappa * parallel))
    torch.testing.assert_close(actual, expected)
    # Independent short-arc transport formula for equal-radius endpoints.
    transported = v - ((v*expected).sum(-1, keepdim=True) /
                       (h.shape[-1] + (h*expected).sum(-1, keepdim=True))) * (h+expected)
    wanted = tangent_project(expected, site.decay*transported + (1-site.decay)*(expected-h))
    torch.testing.assert_close(velocity, wanted)
    args = tuple(t.requires_grad_() for t in (h, v, force, gain))
    assert torch.autograd.gradcheck(run, args)
    for magnitude in (0., 1e-12):
        tiny = (magnitude * v.detach()).requires_grad_()
        a, b = run(h, tiny, force, gain)
        grads = torch.autograd.grad(a.sum()+b.sum(), (h, tiny, force, gain))
        assert all(torch.isfinite(g).all() for g in grads)
        if magnitude == 0:
            torch.testing.assert_close(a, normalize_residual(h + site.alpha()*delta))


def test_active_write_cache_and_compiled_training(setting):
    eager, compiled = make(setting), make(setting, compiled=True)
    with torch.no_grad():
        for b in eager.blocks:
            if b.attn_residual.raw_parallel is None:
                b.attn_residual.kappa.fill_(.7)
                b.mlp_residual.kappa.fill_(-.4)
            else:
                b.attn_residual.raw_parallel.add_(.7)
                b.mlp_residual.raw_parallel.sub_(.4)
    compiled.load_state_dict(eager.state_dict())
    x, y = inputs()
    with torch.no_grad():
        logits = eager(x)
        cache = KVCache(2,1,32,128,2,'cuda',torch.bfloat16)
        chunked = torch.cat([eager(x[:,:7],kv_cache=cache),eager(x[:,7:],kv_cache=cache)], dim=1)
        assert (logits-chunked).norm() < .025*logits.norm()
        other = x.clone()
        other[:,7:] = (other[:,7:]+1) % setting.vocab_size
        assert torch.equal(logits[:,:7], eager(other)[:,:7])
    a, b = eager(x,y), compiled(x,y)
    torch.testing.assert_close(a,b, rtol=2e-4, atol=2e-4)
    a.backward()
    b.backward()
    for name, p in compiled.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # Compare the whole vector of gate derivatives to avoid a relative error
    # dominated by a single nearly orthogonal sublayer.
    def write_grads(model):
        names = ('.raw_parallel','.raw') if setting.ngpt_velocity_mode == 'lookahead_split_write' else ('.kappa',)
        return torch.cat([p.grad for n,p in model.named_parameters() if n.endswith(names)])
    expected = write_grads(eager)
    assert (write_grads(compiled)-expected).norm() <= .04*expected.norm()+1e-8
