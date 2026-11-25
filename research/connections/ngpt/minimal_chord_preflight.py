"""Hopper preflight for no-alpha Chord with raw or normalized branch outputs."""
import argparse
from dataclasses import replace
from pathlib import Path
import gc
import json
import math

import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig
from nanochat.engine import KVCache
from nanochat.gpt import norm
from nanochat.lookahead import MinimalChordResidual
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.native.chord.token_control_checks import check_vjp
from research.connections.ngpt.single_step_preflight import benchmark, make


def unit(x):
    x = x.float()
    return math.sqrt(x.size(-1)) * x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def geometry(c):
    site = MinimalChordResidual(c, True).cuda()
    width = c.width
    h = unit(torch.randn(2, 12, width, device='cuda'))
    velocity = torch.randn_like(h)
    v = (velocity - (velocity*h).mean(-1, keepdim=True)*h).bfloat16().requires_grad_()
    h = h.bfloat16().requires_grad_()
    f = torch.randn_like(h).requires_grad_()
    with torch.no_grad():
        site.mu.fill_(-.3)
        site.raw_beta.fill_(.7)
    q = site.branch_input(h, v, h.dtype, True)
    n, nv = site(h, v, f)
    if c.normalize_minimal_force:
        offset = site.mu * v.float()
        preview = (h.float() + offset) / (1 + offset.square().mean(-1, keepdim=True)).sqrt()
        q_ref = norm(preview.bfloat16())
    else:
        q_ref = unit(h.float() + site.mu*v.float())
    n_ref = unit(h.float() + (unit(f) if c.normalize_minimal_force else f.float()))
    beta = site.raw_beta.sigmoid()
    mix = beta*v.float() + (1-beta)*(n_ref-h.float())
    nv_ref = mix - n_ref*(n_ref*mix).mean(-1, keepdim=True)
    check_vjp((q_ref.bfloat16(), n_ref.bfloat16(), nv_ref.bfloat16()), (q,n,nv),
              (h,v,f,site.mu,site.raw_beta), bounded_read=True)
    with torch.no_grad():
        assert (n.float().square().mean(-1)-1).abs().max() < .004
        assert (n.float()*nv.float()).mean(-1).abs().max() < .004
        scaled, _ = site(h, v, 2*f)
        if c.normalize_minimal_force:
            torch.testing.assert_close(scaled, n, atol=0, rtol=0)
        else:
            assert (scaled.float()-n.float()).norm() > .1 * n.float().norm()
        # An exactly unit BF16 vector and its antipode cancel in both variants.
        pole = torch.ones_like(h)
        assert torch.equal(site(pole, v, -pole)[0], pole)
        torch.testing.assert_close(site(h,v,torch.zeros_like(f))[0].float(),
                                   unit(h).bfloat16().float(), atol=.016, rtol=.003)


def model_checks(c, control, data):
    m, other = make(c, False), make(control, False)
    for name, p in m.named_parameters():
        if '_residual.' not in name:
            assert torch.equal(p, other.get_parameter(name)), name
    ids, labels = (t[:, :32].contiguous() for t in data.batch('dev', 0, 1))
    m.eval()
    with torch.no_grad():
        sites = [s for b in m.blocks for s in (b.attn_residual, b.mlp_residual)]
        assert sum(p.numel() for s in sites for p in s.parameters()) == 2*len(sites)-1
        assert all(s.mu.item()==0 for s in sites)
        assert all(s.raw_beta.sigmoid().item()==.5 for s in sites[:-1]) and sites[-1].raw_beta is None
        # Independent forward oracle at the initial mu, including both norms on the normalized-force route.
        expected = m.initial_state(m.wte(ids), ids)[0]
        cos_sin = (m.cos[:, :ids.size(1)], m.sin[:, :ids.size(1)])
        for b in m.blocks:
            q = norm(expected) if c.normalize_minimal_force else unit(expected).bfloat16()
            a = b.attn(q, None, cos_sin, (-1, 0), None)
            expected = unit(expected.float() + (unit(a) if c.normalize_minimal_force else a.float())).bfloat16()
            q = norm(expected) if c.normalize_minimal_force else unit(expected).bfloat16()
            f = b.mlp(q)
            expected = unit(expected.float() + (unit(f) if c.normalize_minimal_force else f.float())).bfloat16()
        ref = norm(expected)
        actual = m.hidden(ids)
        assert (actual.float()-ref.float()).norm() < .015 * ref.float().norm()
        for site in sites:
            site.mu.fill_(.4)
        h = m.hidden(ids)
        changed = ids.clone()
        changed[:, 16:] = changed[:, 16:].roll(1, 1)
        torch.testing.assert_close(h[:, :16], m.hidden(changed)[:, :16], rtol=0, atol=0)
        cache = KVCache(1,c.heads,ids.size(1),c.width//c.heads,c.depth,'cuda',torch.bfloat16)
        prefix = m.hidden(ids[:, :-1].contiguous(), cache)
        last = m.hidden(ids[:, -1:], cache)
        assert cache.get_pos() == ids.size(1)
        torch.testing.assert_close(torch.cat((prefix,last),1), h, atol=.04, rtol=.025)
    init = 'normal' if c.normalize_minimal_force else 'zero'
    fresh = make(replace(c, residual_projection_init=init), False)
    assert fresh.velocity_embedding.weight.count_nonzero() > 0
    loss = fresh(ids, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for b in fresh.blocks:
        for proj in (b.attn.c_proj, b.mlp.c_proj):
            assert bool(proj.weight.count_nonzero()) == (init == 'normal')
            assert torch.isfinite(proj.weight.grad).all() and proj.weight.grad.count_nonzero() > 0
        if c.normalize_minimal_force:
            for weight in (b.attn.c_v.weight, b.mlp.c_fc.weight):
                assert torch.isfinite(weight.grad).all() and weight.grad.count_nonzero() > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize-force', action='store_true')
    normalized = parser.parse_args().normalize_force
    configure_attention('fa2')
    seed_all(42)
    c = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', ngpt_velocity_mode='lookahead_minimal',
                         chord_velocity_geometry='project_only',velocity_init='token',fuse_chord_read_norm=True)
    control = replace(c,ngpt_velocity_mode='lookahead_split_learned_beta',ngpt_force_rms_floor=1e-6)
    if normalized:
        c = replace(c, normalize_minimal_force=True, ngpt_force_rms_floor=control.ngpt_force_rms_floor)
    data = PackedData('../data')
    geometry(c)
    # Exercise causal and cached inference with active branches, then test the formal initialization.
    model_checks(c, control, data)
    c = replace(c, residual_projection_init='normal' if normalized else 'zero')
    if Path('source.json').exists():
        source = json.loads(Path('source.json').read_text())
        assert source['residual_projection_init'] == c.residual_projection_init
        assert '--residual-projection-init='+c.residual_projection_init in source['model_args']
        assert ('--normalize-minimal-force' in source['model_args']) == normalized
    result = dict(gpu=torch.cuda.get_device_name(), geometry_vjp=True, first_branch_gradients=True,
                  zero_output_init=not normalized, normalized_branch=normalized, causal_and_cache=True,
                  minimal_residual_projection_init=c.residual_projection_init)
    for name, cfg in [('minimal', c), ('default_chord', control)]:
        gc.collect()
        torch.cuda.empty_cache()
        result[name] = benchmark(cfg, data, name=='minimal')
    result['step_time_ratio'] = result['minimal']['median_step_s']/result['default_chord']['median_step_s']
    report = 'vanilla-norm-force-no-alpha-sm90.json' if normalized else 'vanilla-minimal-sm90.json'
    json_write(Path('results')/report, result)
    print('VANILLA_MINIMAL_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
