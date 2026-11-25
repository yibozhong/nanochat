"""Check update-based Chord against an independent derivative oracle and its target-based control."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig
from nanochat.engine import KVCache
from nanochat.lookahead import LookaheadResidual
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.native.chord.preflight import assert_exact_state
from research.connections.ngpt.single_step_preflight import benchmark, make


def unit(x):
    return x / x.square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()


def check_write(cfg):
    site = LookaheadResidual(cfg).cuda()
    # Different step sizes exercise both axis derivatives, including the h-v path.
    with torch.no_grad():
        site.raw.fill_(-1.6)
        site.raw_parallel.fill_(.2)
        site.raw_beta.fill_(.7)
    h = unit(torch.randn(25, cfg.width, device='cuda'))
    v, force = torch.randn_like(h), torch.randn_like(h)
    v[0].zero_()
    force[1].zero_()
    inputs = tuple(x.requires_grad_() for x in (h, v, force)) + (site.raw, site.raw_parallel, site.raw_beta)
    leaves = [x.detach().double().requires_grad_() for x in inputs]
    hd, vd, fd, a, b, beta_logit = leaves
    update = unit(fd)
    parallel = vd * (vd * update).mean(-1, keepdim=True) / vd.square().mean(-1, keepdim=True).clamp_min(1e-12)
    pos = unit(hd + torch.nn.functional.softplus(a)*(update-parallel) + torch.nn.functional.softplus(b)*parallel)
    beta = beta_logit.sigmoid()
    mixed = beta*vd + (1-beta)*(pos-hd)
    expected = (pos, mixed - pos*(mixed*pos).mean(-1, keepdim=True))
    probes = tuple(torch.randn_like(h) for _ in expected)
    grads = torch.autograd.grad(expected, leaves, tuple(t.double() for t in probes))
    for fn in (site, site.reference_write):
        out = fn(h, v, force)
        for result, ref in zip(out, expected):
            torch.testing.assert_close(result.double(), ref, atol=3e-6, rtol=3e-5)
        got = torch.autograd.grad(out, inputs, probes)
        for result, ref in zip(got, grads):
            # Per-row relative error also tests the zero-force row, where the norm Jacobian is 1/floor.
            error = (result.double()-ref).norm(dim=-1)
            assert torch.all(error <= 3e-4*ref.norm(dim=-1) + 1e-5)


@torch.no_grad()
def check_initialization_and_cache(cfg, control, data):
    model = make(cfg, False).eval()
    baseline = make(control, False).eval()
    assert_exact_state(model.state_dict(), baseline.state_dict())
    ids = data.batch('dev', 0, 1)[0][:, :32].contiguous()
    for block in model.blocks:
        for site in (block.attn_residual, block.mlp_residual):
            assert site.update_based
            site.mu.fill_(.4)
            site.raw_parallel.add_(.5)
    hidden = model.hidden(ids)
    changed = ids.clone()
    changed[:, 16:] = ids[:, 16:].roll(1, 1)
    torch.testing.assert_close(model.hidden(changed)[:, :16], hidden[:, :16], atol=0, rtol=0)
    cache = KVCache(1, cfg.heads, 32, cfg.width//cfg.heads, cfg.depth, 'cuda', torch.bfloat16)
    prefix = model.hidden(ids[:, :-1].contiguous(), cache)
    last = model.hidden(ids[:, -1:].contiguous(), cache)
    assert cache.get_pos() == ids.size(1)
    torch.testing.assert_close(torch.cat((prefix, last), dim=1), hidden, atol=.04, rtol=.025)


def main():
    source = json.loads(Path('source.json').read_text())
    flags = Path('research/connections/ngpt/chord.args').read_text().splitlines()
    assert source['model_args'] == flags + ['--chord-update-based']
    configure_attention('fa2')
    seed_all(42)
    control = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', velocity_init='token',
                               ngpt_velocity_mode='lookahead_split_learned_beta', chord_velocity_geometry='project_only',
                               fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    cfg = replace(control, chord_update_based=True)
    data = PackedData('../data')
    check_write(cfg)
    check_initialization_and_cache(cfg, control, data)
    result = dict(gpu=torch.cuda.get_device_name(), write_vjp=True, paired_weights=True, causal_and_cache=True)
    for name, setting in [('update', cfg), ('target', control)]:
        gc.collect()
        torch.cuda.empty_cache()
        result[name] = benchmark(setting, data, verify_resume=name=='update')
    result['step_time_ratio'] = result['update']['median_step_s']/result['target']['median_step_s']
    json_write(Path('results/vanilla-split-update-sm90.json'), result)
    print('VANILLA_SPLIT_UPDATE_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
