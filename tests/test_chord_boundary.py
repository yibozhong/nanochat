from dataclasses import replace

import pytest
import torch

from nanochat.chord_triton import split_boundary
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.engine import KVCache
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from test_momentum import gpu, inputs

CONFIG = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead',
                          ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6)


@pytest.mark.parametrize('terminal,mu_init', [(False, 0.), (False, -.41), (True, .3)])
@pytest.mark.parametrize('width', [128, 512])
def test_boundary_gradients(terminal, mu_init, width):
    site = LookaheadResidual(CONFIG, velocity_consumed=not terminal).cuda()
    mu = torch.tensor([mu_init], device='cuda', requires_grad=True)
    with torch.no_grad():
        site.raw_parallel.add_(.73)
    h = normalize_residual(torch.randn(33, width, device='cuda'))
    v = tangent_project(h, torch.randn_like(h))
    force = torch.randn_like(h).bfloat16()
    v[0].zero_()
    force[1].zero_()
    h[2].fill_(1.)
    force[2].fill_(-1.)
    v[2] -= v[2].mean()
    h.requires_grad_(); v.requires_grad_(); force.requires_grad_()
    want_h, want_v = site(h, v, force)
    if terminal:
        ref, actual = (want_h,), (split_boundary(site, h, v, force, None),)
        args = (h, v, force, site.raw, site.raw_parallel)
    else:
        offset = mu * want_v
        query = (want_h + offset) * torch.rsqrt(1. + offset.square().mean(-1, keepdim=True))
        ref = (want_h, want_v, query.bfloat16())
        actual = split_boundary(site, h, v, force, mu)
        args = (h, v, force, site.raw, site.raw_parallel, site.raw_beta, mu)
    for x, y in zip(actual, ref):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=.008 if x.dtype == torch.bfloat16 else 1e-6)
    up = tuple(torch.randn_like(x) for x in ref)
    want = torch.autograd.grad(ref, args, up)
    got = torch.autograd.grad(actual, args, up)
    for i, (x, y) in enumerate(zip(got, want)):
        chunks = [(x[j], y[j]) for j in range(3)] + [(x[3:], y[3:])] if i < 3 else [(x,y)]
        for a,b in chunks:
            atol = .008 if a.dtype == torch.bfloat16 else .0004
            assert (a.float()-b.float()).norm() <= atol*b.float().norm()+2e-5, (i,a,b)


@pytest.mark.parametrize('mode', ['terminal', 'boundary', 'both'])
def test_boundary_model(mode):
    torch._dynamo.reset()
    c = replace(CONFIG, depth=3, width=128, heads=1, vocab_size=256, sequence_length=32)
    ref = build_connection_model(c).prepare(compile_blocks=True, chord_kernel='current')
    model = build_connection_model(c).prepare(compile_blocks=True, chord_kernel=mode)
    with torch.no_grad():
        for i, block in enumerate(ref.blocks):
            for j, site in enumerate((block.attn_residual, block.mlp_residual)):
                site.mu.fill_((1+i*2+j)*.06-.2)
                site.raw_parallel.add_(.5)
    model.load_state_dict(ref.state_dict(), strict=True)
    x, y = inputs()
    with torch.no_grad():
        logits = ref(x)
        actual = model(x)
        assert (actual - logits).norm() <= .015*logits.norm()
        changed = x.clone()
        changed[:,7:] = (changed[:,7:]+1) % c.vocab_size
        assert torch.equal(actual[:,:7], model(changed)[:,:7])
        cache = KVCache(2,1,32,128,c.depth,'cuda',torch.bfloat16)
        pieces = [model(x[:,:7], kv_cache=cache), model(x[:,7:], kv_cache=cache)]
        assert (torch.cat(pieces,dim=1)-actual).norm() <= .025*actual.norm()
    loss, candidate_loss = ref(x,y), model(x,y)
    torch.testing.assert_close(candidate_loss, loss, rtol=2e-4, atol=2e-4)
    loss.backward(); candidate_loss.backward()
    for (name,p), (other,q) in zip(ref.named_parameters(), model.named_parameters()):
        assert name == other and torch.isfinite(q.grad).all(), name
        assert (p.grad.float()-q.grad.float()).norm() <= .04*p.grad.float().norm()+1e-6, name
