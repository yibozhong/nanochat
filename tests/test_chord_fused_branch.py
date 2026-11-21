"""GPU checks for Chord read + branch RMSNorm and output-projection recomputation."""
from dataclasses import replace
import os

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.engine import KVCache
from nanochat.gpt import Linear, norm
from nanochat.lookahead import LookaheadResidual
from test_chord_token_velocity import test_long_sequence_state_storage
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead',
                            velocity_init=os.environ.get('CHORD_TEST_VELOCITY_INIT', 'token'),
                            residual_projection_init='normal', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry='project_only', fuse_chord_read_norm=True)


@pytest.mark.parametrize('width,scale,mu', [(128, .001, -.3), (512, 1., 0.), (512, 1., .4)])
def test_read_norm(setting, width, scale, mu):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.mu.fill_(mu)
    h, v = [(scale*torch.randn(37, width, device='cuda')).bfloat16().requires_grad_() for _ in range(2)]
    expected = norm(site.query(h, v))
    got = site.branch_input(h, v, h.dtype, True)
    assert [x.data_ptr() for x in got.grad_fn.saved_tensors] == [x.data_ptr() for x in (h, v, site.mu)]
    torch.testing.assert_close(expected, got, rtol=.008, atol=1e-5)
    dy = torch.randn(width, 37, device='cuda', dtype=h.dtype).T
    wanted, actual = [torch.autograd.grad(y, (h, v, site.mu), dy) for y in (expected, got)]
    for a, b in zip(actual, wanted):
        assert (a.float()-b.float()).norm() <= .015*b.float().norm()+2e-4


@pytest.mark.parametrize('expansion', [1, 4])
def test_force_recomputation(setting, expansion):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.7)
    layer = Linear(expansion*128, 128, bias=False).cuda()
    h, v = [torch.randn(2, 17, 128, device='cuda', dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
    x = torch.randn(2, 17, expansion*128, device='cuda', dtype=h.dtype, requires_grad=True)
    ref = site.projected_write(h, v, x, layer, False)
    out = site.projected_write(h, v, x, layer, True)
    saved = out[0].grad_fn.saved_tensors
    assert len(saved) == 5 and saved[-2].data_ptr() == x.data_ptr()
    assert saved[-1].dtype == x.dtype and saved[-1].shape == layer.weight.shape
    grads = tuple(torch.randn_like(y) for y in ref)
    leaves = (h, v, x, layer.weight, site.raw, site.raw_parallel, site.raw_beta)
    for a,b in zip(ref,out):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    want, got = [torch.autograd.grad(y, leaves, grads) for y in (ref, out)]
    for a,b in zip(want,got):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('remat', [False, True])
def test_fused_model(setting, remat):
    ref = make(replace(setting, fuse_chord_read_norm=False))
    target = make(replace(setting, recompute_chord_force=remat), compiled=True)
    with torch.no_grad():
        for block in ref.blocks:
            block.attn_residual.raw_parallel.add_(.65)
            block.mlp_residual.raw_parallel.sub_(.5)
    target.load_state_dict(ref.state_dict(), strict=True)
    ids, labels = inputs()
    h, v = target.initial_state(target.wte(ids), ids)
    assert h.dtype == v.dtype == torch.bfloat16
    assert (target.velocity_embedding is not None) == (setting.velocity_init in ('token', 'token_position'))
    assert (target.velocity_position_embedding is not None) == (setting.velocity_init == 'token_position')
    with torch.no_grad():
        logits = target(ids)
        assert (logits-ref(ids)).norm() < .025*logits.norm()
        cache = KVCache(2,1,32,128,2,'cuda',torch.bfloat16)
        chunked = torch.cat([target(ids[:,:7],kv_cache=cache), target(ids[:,7:],kv_cache=cache)],dim=1)
        assert (chunked-logits).norm() < .025*logits.norm()
        changed = ids.clone()
        changed[:,7:] = (changed[:,7:]+1) % setting.vocab_size
        assert torch.equal(logits[:,:7], target(changed)[:,:7])
    a,b = ref(ids,labels),target(ids,labels)
    torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    a.backward(); b.backward()
    expected, actual = [], []
    for (_, p), (_, q) in zip(ref.named_parameters(), target.named_parameters()):
        assert q.grad is not None and torch.isfinite(q.grad).all()
        expected.append(p.grad.float().flatten())
        actual.append(q.grad.float().flatten())
    left,right = torch.cat(expected),torch.cat(actual)
    assert (left-right).norm() < .03*left.norm()
    if remat:
        # Count the real dispatcher calls after compilation, separately by pass.
        from torch.profiler import ProfilerActivity, profile
        target.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU]) as forward:
            loss = target(ids,labels)
        with profile(activities=[ProfilerActivity.CPU]) as backward:
            loss.backward()
        def calls(trace):
            return sum(e.count for e in trace.key_averages() if e.key=='nanochat::rebuild_chord_force')
        assert calls(forward)==0 and calls(backward)==2*setting.depth
