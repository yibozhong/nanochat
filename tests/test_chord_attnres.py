"""GPU regression tests for spherical endpoint depth attention."""
from dataclasses import replace

import pytest
import torch

from nanochat.attention_residuals import AttnResMix, reference_mix
from nanochat.chord_attnres_triton import mix_endpoints
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for
from nanochat.engine import KVCache
from nanochat.gpt import norm
from nanochat.packed_training import seed_all
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update as check_resume


@pytest.fixture
def setting():
    return ConnectionConfig(depth=3, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
                            fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6, lookahead_mu_init=.35,
                            chord_attnres=True)


@pytest.mark.parametrize('d,k', [(128, 1), (512, 2), (512, 7), (128, 16)])
def test_mix_reference_and_each_gradient(d, k):
    reader = AttnResMix(d).cuda()
    xs = [torch.randn(1, 19, d, device='cuda', dtype=torch.bfloat16).requires_grad_() for _ in range(k+1)]
    with torch.no_grad():
        xs[0][:, 0].zero_()
    bank = torch.stack([x.flatten(0,1) for x in xs[:-1]], dim=1)
    ref = reference_mix(xs[-1].flatten(0,1), bank, reader.norm_weight, reader.proj_weight).view_as(xs[-1])
    actual = mix_endpoints(xs[-1], tuple(xs[:-1]), reader)
    torch.testing.assert_close(actual, ref, atol=.008, rtol=.008)
    grad = torch.randn_like(actual)
    targets = (*xs, reader.norm_weight, reader.proj_weight)
    for expected, got in zip(torch.autograd.grad(ref, targets, grad), torch.autograd.grad(actual, targets, grad)):
        assert torch.isfinite(got).all()
        assert (expected.float()-got.float()).norm() <= .009*expected.float().norm()+1e-6


def reference_hidden(model, ids):
    # Explicit sublayer recurrence. Store only the post-MLP endpoint at boundaries.
    h, v = model.initial_state(model.wte(ids), ids)
    hist = []
    cos, sin = model.cos[:,:ids.size(1)], model.sin[:,:ids.size(1)]
    for i, b in enumerate(model.blocks):
        for j, site in enumerate((b.attn_residual, b.mlp_residual)):
            q = site.query(h, v)
            if hist:
                r = model.depth_reads[i-model.first_mix][j]
                bank = torch.stack([a.flatten(0,1) for a in hist], dim=1)
                q = reference_mix(q.flatten(0,1), bank, r.norm_weight, r.proj_weight).reshape_as(h)
            force = b.attention_branch(q, cos, sin) if j == 0 else b.mlp_branch(q)
            h, v = site(h, v, force)
        if (i+1) % model.config.attnres_block_layers == 0:
            hist.append(h)
    return norm(h)


@pytest.mark.parametrize('backend,size', [('direct',2), ('packed',2), ('direct',4)])
def test_wiring_gradients_and_cache(setting, backend, size):
    c = replace(setting, attnres_block_size=size, chord_attnres_backend=backend)
    seed_all(42)
    control = build_connection_model(replace(c, chord_attnres=False, chord_attnres_backend='direct'))
    seed_all(42)
    candidate = build_connection_model(c)
    for key, value in control.state_dict().items():
        assert torch.equal(candidate.state_dict()[key], value), key
    model = candidate.prepare(compile_blocks=True)
    seed_all(42)
    reference = make(c)
    with torch.no_grad():
        for b in model.blocks:
            b.attn_residual.raw_parallel.add_(.65)
            b.mlp_residual.raw_parallel.sub_(.5)
        reference.load_state_dict(model.state_dict())
    ids, labels = inputs()
    expected = reference.readout(reference_hidden(reference, ids), labels)
    loss = model(ids, labels)
    torch.testing.assert_close(loss, expected, atol=.001, rtol=2e-4)
    expected.backward()
    loss.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name.startswith('depth_reads.'):
            assert p.grad.norm() > 0, name
    grads = [torch.cat([p.grad.flatten().float() for p in m.parameters()]) for m in (model, reference)]
    assert (grads[0]-grads[1]).norm() < .04*grads[1].norm()
    optimizer_for(model)
    with torch.no_grad():
        logits = model(ids)
        cache = KVCache(2,1,32,128,c.depth,'cuda',torch.bfloat16)
        parts = torch.cat([model(ids[:,:7],kv_cache=cache),model(ids[:,7:],kv_cache=cache)], dim=1)
        assert (parts-logits).norm() < .025*logits.norm()
        changed = ids.clone()
        changed[:,7:] = (changed[:,7:]+1) % c.vocab_size
        assert torch.equal(logits[:,:7], model(changed)[:,:7])


def test_first_block_is_plain_chord(setting):
    # With no completed earlier block the entire model must reduce to Chord.
    c = replace(setting, depth=1)
    seed_all(42)
    model = make(c)
    seed_all(42)
    base = make(replace(c, chord_attnres=False))
    ids, _ = inputs()
    assert not list(model.depth_reads)
    assert torch.equal(model(ids), base(ids))


@pytest.mark.parametrize('backend', ['direct', 'packed'])
def test_resume(setting, tmp_path, backend):
    check_resume(replace(setting, chord_attnres_backend=backend), tmp_path)
