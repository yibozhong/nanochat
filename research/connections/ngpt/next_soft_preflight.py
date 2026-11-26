"""Hopper checks for the Next soft-force, h-conditioned Chord port."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from nanochat.chord_token_controls import TokenControlledSite, reference_write
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig
from nanochat.engine import KVCache
from nanochat.packed_training import PackedData, json_write, seed_all
from research.connections.native.chord.submit import chord_args
from research.connections.ngpt.chord_checks import benchmark, check_read_vjp, make


def check_read(site, h, v):
    with torch.no_grad():
        site.token_weight.normal_(std=.025)
        site.mix_scales.copy_(site.mix_scales.new_tensor([-.12, .07, .19][:len(site.mix_scales)]))
    q, controls = site.read(h, v)
    normalized = F.rms_norm(h.float(), (h.size(-1),), eps=1e-6)
    z = F.linear(normalized, site.token_weight[:, 0])
    s = site.mix_scales
    m = site.mu + z[..., :1]*s[0]
    alpha = F.softplus(torch.cat((site.raw, site.raw_parallel)) + z[..., 1:3]*s[1])
    if site.terminal:
        beta, gamma = torch.full_like(m, .5), torch.full_like(alpha, .5)
    else:
        beta = (site.raw_beta + z[..., 3:4]*s[2]).sigmoid()
        gamma = site.writes + z[..., 4:6]*s[2]
    offset = m * v.float()
    preview = ((h.float()+offset) * (1 + offset.square().mean(-1, keepdim=True)).rsqrt()).to(h.dtype)
    oracle = (F.rms_norm(preview, (h.size(-1),), eps=torch.finfo(torch.float32).eps),
              torch.cat((m, alpha, beta, gamma), dim=-1))
    check_read_vjp(oracle, (q, controls), (h, v, *site.parameters()))


def check_write(site, h, v, f, controls):
    # Independent FP64 decomposition in the endpoint tangent space.
    hd, vd, fd, c = leaves = [x.detach().double().requires_grad_() for x in (h, v, f, controls)]
    def along(x, axis):
        return axis * ((axis*x).mean(-1, True) / axis.square().mean(-1, True).clamp_min(1e-12))
    def tangent(x, n):
        return x - n*(n*x).mean(-1, True)
    u = fd / (fd.square().mean(-1, True) + .05**2).sqrt()
    _, ap, al, beta, gp, gt = c.unsqueeze(-1).unbind(-2)
    parallel = along(u, vd)
    mixed = hd + al*parallel + ap*(u-parallel)
    n = mixed / mixed.square().mean(-1, True).clamp_min(1e-12).sqrt()
    carried, chord = tangent(vd, n), tangent(n-hd, n)
    par_d = along(chord, carried)
    expected = (n,) if site.terminal else (n, beta*carried + gp*par_d + gt*(chord-par_d))
    probes = tuple(torch.randn_like(h) for _ in expected)
    grads = torch.autograd.grad(expected, leaves, tuple(x.double() for x in probes))
    for outputs in (site.write(h, v, f, controls), reference_write(h, v, f, controls, 1e-6, site.terminal, .05)):
        outputs = tuple(x for x in outputs if x is not None)
        got = torch.autograd.grad(outputs, (h, v, f, controls), probes)
        for actual, oracle in zip((*outputs, *got), (*expected, *grads)):
            err = (actual.double()-oracle).norm(dim=-1)
            assert torch.all(err <= .012*oracle.norm(dim=-1) + 2e-5), err.max().item()


def geometry(c):
    seed_all(42)
    for terminal in (False, True):
        site = TokenControlledSite(c, velocity_consumed=not terminal).cuda()
        h = F.normalize(torch.randn(12, c.width, device='cuda'), dim=-1) * c.width**.5
        v = torch.randn_like(h)
        v[0].zero_()
        h, v = (t.bfloat16().requires_grad_() for t in (h, v))
        check_read(site, h, v)
        amps = [0., 1e-8, .001, .01, .03, .049, .05, .051, .1, .25, 1., 4.]
        f = (torch.randn_like(h.float()) * h.new_tensor(amps).float()[:, None]).bfloat16().requires_grad_()
        controls = (torch.randn(12, 6, device='cuda')*.12 + h.new_tensor([0., .17, .09, .6, .4, .72]).float()).requires_grad_()
        check_write(site, h, v, f, controls)
    h = torch.ones_like(h[:1])
    f = torch.zeros_like(h, requires_grad=True)
    c0 = torch.tensor([[0., .05, .05, .5, .5, .5]], device='cuda')
    n, _ = site.write(h, torch.zeros_like(h), f, c0)
    probe = torch.randn_like(h)
    df, = torch.autograd.grad(n, f, probe)
    torch.testing.assert_close(df, (probe.float()-probe.float().mean(-1, True)).bfloat16(), atol=1e-5, rtol=.004)


def check_projected_h0(model, ids):
    weight = model.velocity_projection.weight
    assert model.velocity_projection.bias is None and model.velocity_embedding is model.velocity_position_embedding is None
    assert weight.shape == (model.config.width, model.config.width)
    x = model.wte(ids).detach().requires_grad_()
    h, v = model.initial_state(x)
    # Different positive scales per token expose an accidental W @ raw_embedding.
    scales = x.new_tensor([.25, 1., 2., 8.]).repeat(ids.size(1)//4).view(1, -1, 1)
    with torch.no_grad():
        for actual, scaled in zip((h, v), model.initial_state(x * scales)):
            torch.testing.assert_close(actual, scaled, rtol=0, atol=0)
    xd, wd = (a.detach().double().requires_grad_() for a in (x, weight))
    hd = xd / xd.square().mean(-1, True).sqrt()
    raw = F.linear(hd, wd)
    vd = raw - hd * (hd * raw).mean(-1, True)
    probe = torch.randn_like(v)
    got = torch.autograd.grad(v, (x, weight), probe)
    expected = torch.autograd.grad(vd, (xd, wd), probe.double())
    for a, b in zip((h, v, *got), (hd, vd, *expected)):
        assert (a.double()-b).norm() <= .012*b.norm()
    assert (h.float()*v.float()).mean(-1).abs().max() < 1e-4
    assert abs(v.float().square().mean().sqrt().item()/model.config.velocity_embedding_std - 1) < .1


def check_model(c, data):
    model = make(c, False)
    ids, targets = (t[:, :32].contiguous() for t in data.batch('dev', 0, 1))
    loss = model(ids, targets)
    loss.backward()
    assert torch.isfinite(loss)
    if c.velocity_init == 'projected_h0':
        check_projected_h0(model, ids)
    grad_norm = torch.stack([p.grad.float().norm() for p in model.parameters() if p.grad is not None]).norm()
    assert torch.isfinite(grad_norm)
    for block in model.blocks:
        for branch, site in ((block.attn.c_proj, block.attn_residual), (block.mlp.c_proj, block.mlp_residual)):
            assert branch.weight.count_nonzero() == 0 and branch.weight.grad.norm() > 0
            assert site.token_weight.count_nonzero() == 0 and (site.mix_scales == .01).all()
            # Exercise inference after the initially neutral branches and token gates have become active.
            with torch.no_grad():
                branch.weight.normal_(std=.005)
                site.mu.fill_(.4)
                site.token_weight.normal_(std=.02)
    model.zero_grad(set_to_none=True)
    model(ids, targets).backward()
    velocity_module = model.velocity_projection if c.velocity_init == 'projected_h0' else model.velocity_embedding
    if velocity_module is not None:
        grad = velocity_module.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    model.eval()
    with torch.no_grad():
        hidden = model.hidden(ids)
        changed = ids.clone()
        changed[:, 16:] = ids[:, 16:].roll(1, 1)
        torch.testing.assert_close(hidden[:, :16], model.hidden(changed)[:, :16], atol=0, rtol=0)
        cache = KVCache(1, c.heads, 32, c.width//c.heads, c.depth, 'cuda', torch.bfloat16)
        prefix = model.hidden(ids[:, :-1].contiguous(), cache)
        last = model.hidden(ids[:, -1:].contiguous(), cache)
        assert cache.get_pos() == ids.size(1)
        torch.testing.assert_close(torch.cat((prefix,last), 1), hidden, atol=.04, rtol=.025)
    return grad_norm.item()


def main():
    configure_attention('fa2')
    torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads(Path('source.json').read_text())
    assert source['recipe'] == 'chord_next_bf16'
    assert source['model_args'] == chord_args(source['recipe'], source['velocity_init'])
    control = ConnectionConfig(streams=1, residual_dynamics='ngpt_lookahead', velocity_init=source['velocity_init'],
                               chord_soft_force_tau=.05, chord_six_controls='static', chord_update_based=True,
                               ngpt_velocity_mode='lookahead_split_learned_beta', residual_projection_init='zero',
                               chord_velocity_geometry='project_only', fuse_chord_read_norm=True, ngpt_force_rms_floor=1e-6)
    c = replace(control, chord_six_controls='learned_dynamic')
    run, reference = source['recipe'], 'chord_scalar_bf16'
    data = PackedData('../data')
    geometry(c)
    result = dict(gpu=torch.cuda.get_device_name(), short_batch_init_grad_norm=check_model(c, data),
                  checked_lookahead_and_force_gradients=True, checked_causal_and_cache=True)
    for name, cfg in [(run, c), (reference, control)]:
        gc.collect()
        torch.cuda.empty_cache()
        result[name] = benchmark(cfg, data, verify_resume=name==run, schedule_steps=source['schedule_steps'])
    result['step_time_ratio'] = result[run]['median_step_s']/result[reference]['median_step_s']
    json_write(Path(f"results/vanilla-next-soft-{c.velocity_init}-sm90.json"), result)
    print('NEXT_SOFT_PREFLIGHT_PASS', result, flush=True)


if __name__ == '__main__':
    main()
