"""Bounded residual correctness/stability/throughput check; never creates W&B."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from nanochat.connection_cost import model_cost
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
from nanochat.packed_training import PackedData, evaluate, json_write, seed_all, train_step
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.gpt import norm


@torch.no_grad()
def residual_audit(model, x, require_unit_target=False):
    pure = model.config.residual_mode == 'ngpt'
    vanilla = model.config.residual_mode == 'vanilla'
    transported = model.config.residual_mode in ('ngpt_heavy_ball', 'ngpt_hbformer') and model.config.ngpt_velocity_mode == 'transport_rms'
    if vanilla:
        h, v = norm(model.wte(x)), None
    elif pure:
        h, v = normalize_residual(model.wte(x)), None
    else:
        h, v = model.initial_state(model.wte(x), x)
    cos, sin = model.cos[:, :x.shape[1]], model.sin[:, :x.shape[1]]
    rows = []
    for b in model.blocks:
        for name, branch, update in (
            ('attention', lambda z: b.attention_branch(z, cos, sin), getattr(b, 'attn_residual', None)),
            ('ffn', b.mlp_branch, getattr(b, 'mlp_residual', None))):
            force = branch(h.to(cos.dtype))
            force_rms = force.float().square().mean(-1).sqrt()
            row = dict(sublayer=len(rows), branch=name,
                       force_rms=force.float().square().mean().sqrt().item(),
                       min_token_force_rms=force_rms.min().item())
            if model.config.residual_mode in ('ngpt', 'ngpt_heavy_ball', 'ngpt_hbformer'):
                row['force_floor_fraction'] = (force_rms < model.config.ngpt_force_rms_floor).float().mean().item()
                z = normalize_residual(force, model.config.ngpt_force_rms_floor)
                row['target_rms_max_error'] = (z.square().mean(-1)-1).abs().max().item()
                if require_unit_target:
                    assert row['force_floor_fraction'] == 0 and row['target_rms_max_error'] < 2e-6, row
            if vanilla:
                h = h + force
            elif pure:
                h = update(h, force)
            else:
                if transported:
                    beta, gamma, eta = update.coefficients()
                    raw = beta*v + gamma*z
                    tangent = tangent_project(h, raw)
                    raw_rms2 = raw.square().mean(-1)
                    row['incoming_velocity_rms_max'] = v.square().mean(-1).sqrt().max().item()
                    row['incoming_tangent_error_max'] = (h*v).mean(-1).abs().max().item()
                    row['discarded_radial_energy_fraction'] = ((h*raw).mean(-1).square()/raw_rms2.clamp_min(1e-12)).mean().item()
                    row['velocity_floor_fraction'] = (tangent.square().mean(-1) < model.config.momentum_velocity_rms_floor**2).float().mean().item()
                h, v = update(h, v, force)
            if model.config.residual_mode in ('ngpt_heavy_ball', 'ngpt_hbformer', 'ngpt'):
                row['hidden_rms_max_error'] = (h.square().mean(-1)-1).abs().max().item()
                assert row['hidden_rms_max_error'] < 2e-6
            assert torch.isfinite(h).all()
            row.update(hidden_rms=h.float().square().mean().sqrt().item(),
                       max_token_hidden_rms=h.float().square().mean(-1).sqrt().max().item())
            if v is not None:
                assert torch.isfinite(v).all()
                row.update(velocity_rms=v.square().mean().sqrt().item(),
                           observed_hv_moment=(h*v).mean().item())
                if model.config.residual_mode == 'heavy_ball_rms':
                    row['velocity_rms_max'] = v.square().mean(-1).sqrt().max().item()
                    assert row['velocity_rms_max'] < 1+2e-6, row
                if transported:
                    row['velocity_rms_max'] = v.square().mean(-1).sqrt().max().item()
                    row['tangent_error_max'] = (h*v).mean(-1).abs().max().item()
                    assert row['velocity_rms_max'] < 1+2e-6 and row['tangent_error_max'] < 2e-6, row
            # Engineering guard for this bounded check, not a mathematical bound.
            assert row['max_token_hidden_rms'] < 10 and row.get('velocity_rms', 0) < 10, row
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--target-mode', choices=('ngpt_heavy_ball','ngpt','vanilla','heavy_ball_rms','ngpt_hbformer'), default='ngpt_heavy_ball')
    p.add_argument('--nonzero-init', action='store_true')
    p.add_argument('--transport-rms', action='store_true')
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    configure_attention('fa2'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = PackedData(a.data_dir)
    assert len(data.arrays['train']) >= 128*16
    result = dict(complete=False, formal_training=False, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, attention_backend='fa2',
                  seed=42, microbatch=16, batch_tokens=32768, models={})
    if a.target_mode in ('heavy_ball_rms', 'ngpt_hbformer'):
        assert a.nonzero_init and not a.transport_rms
        modes = [('vanilla', 32), ('ngpt_heavy_ball', 32), (a.target_mode, 128)]
    elif a.nonzero_init:
        modes = [(m, 128 if m == a.target_mode else 32) for m in ('vanilla','ngpt','ngpt_heavy_ball')]
    else:
        assert a.target_mode != 'vanilla'
        modes = [('vanilla', 32)]
        if a.target_mode == 'ngpt_heavy_ball':
            modes.append(('heavy_ball', 32))
        modes.append((a.target_mode, 128))
    if a.transport_rms:
        assert a.nonzero_init and a.target_mode == 'ngpt_heavy_ball'
        modes.insert(-1, ('ngpt_ambient', 32))
    for mode, steps in modes:
        seed_all(42)
        c = ConnectionConfig(streams=1, residual_dynamics='ngpt_heavy_ball' if mode == 'ngpt_ambient' else mode,
                             ngpt_velocity_mode='transport_rms' if (mode == 'ngpt_hbformer' or mode == 'ngpt_heavy_ball'
                                 and (a.transport_rms or a.target_mode in ('ngpt_hbformer', 'heavy_ball_rms'))) else 'ambient',
                             residual_projection_init='normal' if a.nonzero_init else 'zero',
                             ngpt_force_rms_floor=1e-6 if a.nonzero_init else .1)
        model = build_connection_model(c).prepare(compile_blocks=True)
        opt, wd = optimizer_for(model)
        initial_dev = evaluate(model, data)[0] if mode == a.target_mode else None
        initial_state = residual_audit(model, data.batch('dev', 0, 2)[0], require_unit_target=a.nonzero_init) if mode == a.target_mode or a.nonzero_init else None
        durations, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        with (out/(mode+'-steps.jsonl')).open('w', buffering=1) as log:
            for step in range(steps):
                schedule(opt, step, 305176, wd)
                torch.cuda.synchronize(); begin = time.perf_counter()
                loss = train_step(model, opt, data, step, microbatch=16)
                torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                row = dict(model=mode, step=step+1, loss=loss, seconds=seconds)
                log.write(json.dumps(row)+'\n'); losses.append(loss)
                if 16 <= step < 32:
                    durations.append(seconds)
                if step < 2 or (step+1) % 32 == 0:
                    print(json.dumps(row), flush=True)
        info = dict(config=asdict(c), parameters=sum(p.numel() for p in model.parameters()),
                    steps=steps, median_step_seconds=statistics.median(durations),
                    mean_step_seconds=statistics.mean(durations),
                    tokens_per_second=32768/statistics.mean(durations),
                    first_loss=losses[0], last_loss=losses[-1],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    initial_residual_state=initial_state,
                    theoretical_cost=model_cost(c, batch=16))
        if mode != 'vanilla':
            info['coefficients'] = model.residual_metrics()
        if mode == a.target_mode:
            info.update(initial_dev_ce=initial_dev, final_dev_ce=evaluate(model, data)[0],
                        final_residual_state=residual_audit(model, data.batch('dev', 0, 2)[0], require_unit_target=a.nonzero_init))
            assert statistics.mean(losses[-32:]) < statistics.mean(losses[:32]) and max(losses) < 20
            assert info['final_dev_ce'] < initial_dev
            assert info['peak_allocated_bytes'] < .9*torch.cuda.get_device_properties(0).total_memory
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], with_flops=True) as prof:
            train_step(model, opt, data, 0, microbatch=16)
            torch.cuda.synchronize()
        info['profiler_supported_flops'] = sum(e.flops for e in prof.key_averages() if e.flops)
        info['profiler_coverage'] = 'Partial compiled operator count; FA2/CCE/fused kernels not fully covered; not total hardware FLOPs.'
        result['models'][mode] = info
        json_write(out/'summary.json', result)
        del model, opt, prof
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    m = result['models']
    if 'heavy_ball' in m:
        result['time_ratio_to_hb'] = m[a.target_mode]['median_step_seconds']/m['heavy_ball']['median_step_seconds']
        assert result['time_ratio_to_hb'] < 1.75
    result['time_ratio_to_vanilla'] = m[a.target_mode]['median_step_seconds']/m['vanilla']['median_step_seconds']
    assert result['time_ratio_to_vanilla'] < 1.75
    if a.nonzero_init:
        result['time_ratios_to_vanilla'] = {name: info['median_step_seconds']/m['vanilla']['median_step_seconds'] for name, info in m.items()}
        assert all(ratio < 1.75 for ratio in result['time_ratios_to_vanilla'].values())
    if a.transport_rms:
        result['time_ratio_to_ambient'] = m[a.target_mode]['median_step_seconds']/m['ngpt_ambient']['median_step_seconds']
        assert result['time_ratio_to_ambient'] < 1.35
    result['diagnostic_training_tokens'] = sum(steps+1 for _, steps in modes)*32768
    result['complete'] = True
    json_write(out/'summary.json', result)
    print('MOMENTUM_PROFILE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
