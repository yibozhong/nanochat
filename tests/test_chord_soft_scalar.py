"""Scalar target-based Chord with a smooth force denominator."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.packed_training import seed_all
from nanochat.lookahead import LookaheadResidual
from test_momentum import gpu, inputs, make


def scalar_config():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                            ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
                            chord_state_dtype='bfloat16', fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6,
                            chord_soft_force_tau=.05)


def test_zero_force_gain_and_checkpoint_default():
    cfg = scalar_config()
    assert ConnectionConfig.from_checkpoint(asdict(cfg)) == cfg
    old = asdict(cfg)
    del old['chord_soft_force_tau']
    assert ConnectionConfig.from_checkpoint(old) == replace(cfg, chord_soft_force_tau=0.)
    site = LookaheadResidual(cfg).cuda()
    h = torch.ones(2, cfg.width, device='cuda')
    v = torch.zeros_like(h)
    probe = torch.ones_like(h)
    probe[:, ::2] = -1  # Tangent to h, exactly representable.
    for write in (site, site.reference_write):
        f = torch.zeros_like(h, requires_grad=True)
        n, _ = write(h, v, f)
        torch.testing.assert_close(n, h)
        df, = torch.autograd.grad(n, f, probe)
        gain = .05 / ((1 - .05) * cfg.chord_soft_force_tau)
        torch.testing.assert_close(df, probe * gain)


def test_fused_scalar_write_and_vjp():
    site = LookaheadResidual(scalar_config()).cuda()
    with torch.no_grad():
        site.raw_parallel.add_(.7)
        site.raw_beta.add_(.4)
    h = torch.nn.functional.normalize(torch.randn(8, 512, device='cuda')) * 512**.5
    v = torch.randn_like(h)
    v -= h*(h*v).mean(-1, True)
    v[0].zero_()
    amplitudes = h.new_tensor([0., 1e-8, .01, .049, .05, .051, .2, 1.])
    f = torch.randn_like(h) * amplitudes[:, None]
    actual_inputs = [x.bfloat16().requires_grad_() for x in (h, v, f)]
    oracle = deepcopy(site).double()
    ref_inputs = [x.detach().double().requires_grad_() for x in actual_inputs]
    actual = site(*actual_inputs)
    reference = oracle.reference_write(*ref_inputs)
    probes = [torch.randn_like(x) for x in actual]
    actual_grads = torch.autograd.grad(actual, (*actual_inputs, site.raw, site.raw_parallel, site.raw_beta), probes)
    ref_grads = torch.autograd.grad(reference, (*ref_inputs, oracle.raw, oracle.raw_parallel, oracle.raw_beta),
                                    [g.double() for g in probes])
    for x, y in zip((*actual, *actual_grads), (*reference, *ref_grads)):
        error = (x.double() - y).norm(dim=-1)
        assert torch.all(error < .012 * y.norm(dim=-1) + 2e-5)


def test_zero_output_init_first_ntp_backward():
    cfg = replace(scalar_config(), residual_projection_init='zero')
    ids, labels = inputs()
    output_grads = []
    for tau in (0., cfg.chord_soft_force_tau):
        seed_all(42)
        model = make(replace(cfg, chord_soft_force_tau=tau), perturb=False)
        weights = [branch.c_proj.weight for b in model.blocks for branch in (b.attn, b.mlp)]
        assert all(w.count_nonzero() == 0 for w in weights)
        ce = model(ids, labels)
        ce.backward()
        assert torch.isfinite(ce)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        norms = torch.stack([w.grad.float().norm() for w in weights])
        assert torch.all(norms > 0)
        output_grads.append(norms)
    ratio = output_grads[0] / output_grads[1]
    torch.testing.assert_close(ratio, torch.full_like(ratio, cfg.chord_soft_force_tau / cfg.ngpt_force_rms_floor),
                               rtol=.02, atol=0)
    print('First NTP backward, hard/soft output-projection gradient norm ratios:', ratio.tolist())
