"""GPU checks for Chord with an independent velocity token embedding."""
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from nanochat.connection_model import ConnectionConfig, optimizer_for
from nanochat.chord_triton import split_boundary
from nanochat.lookahead import LookaheadResidual
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import seed_all
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, make, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            residual_projection_init='normal', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_split_learned_beta', ngpt_force_rms_floor=1e-6,
                            chord_velocity_geometry='project_only')


@torch.no_grad()
def test_only_position_table_removed(setting):
    model = make(setting)
    seed_all(42)
    control = make(replace(setting, velocity_init='token_position'))
    weights, original = dict(model.named_parameters()), dict(control.named_parameters())
    assert set(original) == set(weights) | {'velocity_position_embedding.weight'}
    assert model.velocity_position_embedding is None
    for name, param in weights.items():
        assert torch.equal(param, original[name]), name
    token_weight = model.velocity_embedding.weight
    assert token_weight.data_ptr() != model.wte.weight.data_ptr()
    optimizer, _ = optimizer_for(model)
    groups = [group for group in optimizer.param_groups if any(p is token_weight for p in group['params'])]
    assert len(groups) == 1 and groups[0]['kind'] == 'adamw'

    ids = torch.tensor([[7, 11, 7]], device='cuda')
    h, v = model.initial_state(model.wte(ids), ids)
    full_h = normalize_residual(model.wte(ids))
    assert h.dtype == v.dtype == torch.bfloat16
    torch.testing.assert_close(h, full_h.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(v, tangent_project(full_h, model.velocity_embedding(ids).float()).bfloat16(), rtol=0, atol=0)
    old = asdict(setting)
    del old['chord_state_dtype']
    assert ConnectionConfig.from_checkpoint(old).chord_state_dtype == 'float32'
    assert torch.equal(v[:, 0], v[:, 2])
    _, shifted = model.initial_state(model.wte(ids), ids, start=9)
    assert torch.equal(v, shifted)
    assert ConnectionConfig.from_checkpoint(asdict(setting)) == setting


@pytest.mark.parametrize('boundary', [False, True])
def test_bf16_write_query_gradients(setting, boundary):
    site = LookaheadResidual(setting).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.7)
    mu = site.mu
    h = normalize_residual(torch.randn(37, 512, device='cuda')).bfloat16().requires_grad_()
    v = torch.randn_like(h).requires_grad_()
    f = torch.randn_like(h).requires_grad_()
    nh, nv = site.reference_write(h, v, f)
    offset = mu[0] * nv.float()
    q = ((nh.float() + offset) * torch.rsqrt(1 + offset.square().mean(-1, keepdim=True))).bfloat16()
    if boundary:
        got = split_boundary(site, h, v, f, mu)
    else:
        gh, gv = site(h, v, f)
        got = (gh, gv, site.query(gh, gv))
    ref = (nh, nv, q)
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype == torch.bfloat16
        assert (a.float()-b.float()).norm() <= .002*b.float().norm()
    upstream = tuple(torch.randn_like(a) for a in ref)
    inputs = (h, v, f, site.raw, site.raw_parallel, site.raw_beta, mu)
    left = torch.autograd.grad(got, inputs, upstream)
    right = torch.autograd.grad(ref, inputs, upstream)
    for a, b in zip(left, right):
        assert torch.isfinite(a).all()
        assert (a.float()-b.float()).norm() <= .02*b.float().norm()+2e-4


def test_long_sequence_state_storage(setting):
    # Measure only the residual operations here. The CLI profile separately
    # measures whole-model training at the actual 2048-token context.
    site = LookaheadResidual(setting).cuda()
    shape = (1, 16384, 512)
    base = [torch.randn(shape, dtype=torch.bfloat16, device='cuda') for _ in range(3)]
    peaks = {}
    for dtype in (torch.float32, torch.bfloat16):
        h, v = [x.to(dtype).detach().requires_grad_() for x in base[:2]]
        f = base[2].detach().requires_grad_()
        state_bytes = 2*h.numel()*h.element_size()
        def pack(tensor):
            if dtype == torch.bfloat16 and tensor.numel() == h.numel():
                assert tensor.dtype == torch.bfloat16, tensor.dtype
            return tensor
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            for _ in range(4):
                h, v = site(h, v, f)
                q = site.query(h, v).bfloat16()
                f = q
            torch.autograd.backward((h, v, q), tuple(torch.ones_like(x) for x in (h, v, q)))
        torch.cuda.synchronize()
        peaks[str(dtype)] = dict(state_bytes=state_bytes, extra_peak_bytes=torch.cuda.max_memory_allocated()-start)
        del h, v, q, f
        site.zero_grad(set_to_none=True)
    fp32, bf16 = peaks['torch.float32'], peaks['torch.bfloat16']
    assert bf16['state_bytes']*2 == fp32['state_bytes']
    assert bf16['extra_peak_bytes'] < fp32['extra_peak_bytes']
    if report := os.environ.get('CHORD_STORAGE_REPORT'):
        Path(report).write_text(json.dumps(dict(sequence_length=shape[1], width=shape[2], writes=4, measured=peaks)))
