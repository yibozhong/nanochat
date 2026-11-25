"""Hopper preflight for raw-force Chord with scalar lookahead and retention."""
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
    q_ref = unit(h.float() + site.mu*v.float())
    n_ref = unit(h.float() + f.float())
    beta = site.raw_beta.sigmoid()
    mix = beta*v.float() + (1-beta)*(n_ref-h.float())
    nv_ref = mix - n_ref*(n_ref*mix).mean(-1, keepdim=True)
    check_vjp((q_ref.bfloat16(), n_ref.bfloat16(), nv_ref.bfloat16()), (q,n,nv),
              (h,v,f,site.mu,site.raw_beta), bounded_read=True)
    with torch.no_grad():
        assert (n.float().square().mean(-1)-1).abs().max() < .004
        assert (n.float()*nv.float()).mean(-1).abs().max() < .004
        assert (site(h,v,2*f)[0].float()-n.float()).norm() > .1 * n.float().norm()
        # Keep an existing position when the sum has no direction.
        assert torch.equal(site(h,v,-h)[0], h)
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
        # At mu=0 all position updates are ordinary projected residual additions.
        expected = m.initial_state(m.wte(ids), ids)[0]
        cos_sin = (m.cos[:, :ids.size(1)], m.sin[:, :ids.size(1)])
        for b in m.blocks:
            a = b.attn(unit(expected).bfloat16(), None, cos_sin, (-1, 0), None)
            expected = unit(expected.float() + a.float()).bfloat16()
            f = b.mlp(unit(expected).bfloat16())
            expected = unit(expected.float() + f.float()).bfloat16()
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
    zero = make(replace(c,residual_projection_init='zero'),False)
    assert zero.velocity_embedding.weight.count_nonzero() > 0
    loss = zero(ids, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for b in zero.blocks:
        for proj in (b.attn.c_proj, b.mlp.c_proj):
            assert proj.weight.count_nonzero()==0
            assert torch.isfinite(proj.weight.grad).all() and proj.weight.grad.count_nonzero()>0


def main():
    configure_attention('fa2')
    seed_all(42)
    c = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', ngpt_velocity_mode='lookahead_minimal',
                         chord_velocity_geometry='project_only',velocity_init='token',fuse_chord_read_norm=True)
    control = replace(c,ngpt_velocity_mode='lookahead_split_learned_beta',ngpt_force_rms_floor=1e-6)
    data = PackedData('../data')
    geometry(c)
    # Exercise causality and KV with active (nonzero) branches before checking zero-init gradients.
    model_checks(c, control, data)
    c = replace(c, residual_projection_init='zero')
    if Path('source.json').exists():
        source = json.loads(Path('source.json').read_text())
        assert source['residual_projection_init'] == c.residual_projection_init
        assert '--residual-projection-init=zero' in source['model_args']
    result = dict(gpu=torch.cuda.get_device_name(), geometry_vjp=True, zero_output_init=True, causal_and_cache=True,
                  minimal_residual_projection_init=c.residual_projection_init)
    for name, cfg in [('minimal', c), ('default_chord', control)]:
        gc.collect()
        torch.cuda.empty_cache()
        result[name] = benchmark(cfg, data, name=='minimal')
    result['step_time_ratio'] = result['minimal']['median_step_s']/result['default_chord']['median_step_s']
    json_write(Path('results/vanilla-minimal-sm90.json'), result)
    print('VANILLA_MINIMAL_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
