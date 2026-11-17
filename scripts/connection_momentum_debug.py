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


@torch.no_grad()
def residual_audit(model, x):
    h, v = model.initial_state(model.wte(x), x)
    cos, sin = model.cos[:, :x.shape[1]], model.sin[:, :x.shape[1]]
    rows = []
    for b in model.blocks:
        for name, branch, update in (
            ('attention', lambda z: b.attention_branch(z, cos, sin), b.attn_residual),
            ('ffn', b.mlp_branch, b.mlp_residual)):
            force = branch(h.to(cos.dtype))
            row = dict(sublayer=len(rows), branch=name,
                       force_rms=force.float().square().mean().sqrt().item())
            h, v = update(h, v, force)
            if model.config.residual_mode == 'ngpt_heavy_ball':
                row['hidden_rms_max_error'] = (h.square().mean(-1)-1).abs().max().item()
                assert row['hidden_rms_max_error'] < 2e-6
            assert torch.isfinite(h).all() and torch.isfinite(v).all()
            row.update(hidden_rms=h.square().mean().sqrt().item(),
                       max_token_hidden_rms=h.square().mean(-1).sqrt().max().item(),
                       velocity_rms=v.square().mean().sqrt().item(),
                       observed_hv_moment=(h*v).mean().item())
            # Engineering guard for this bounded check, not a mathematical bound.
            assert row['max_token_hidden_rms'] < 10 and row['velocity_rms'] < 10, row
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    configure_attention('fa2'); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = PackedData(a.data_dir)
    assert len(data.arrays['train']) >= 128*16
    result = dict(complete=False, formal_training=False, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, attention_backend='fa2',
                  seed=42, microbatch=16, batch_tokens=32768, models={})
    for mode, steps in [('vanilla', 32), ('heavy_ball', 32), ('ngpt_heavy_ball', 128)]:
        seed_all(42)
        c = ConnectionConfig(streams=1, residual_dynamics=mode)
        model = build_connection_model(c).prepare(compile_blocks=True)
        opt, wd = optimizer_for(model)
        initial_dev = evaluate(model, data)[0] if mode == 'ngpt_heavy_ball' else None
        initial_state = residual_audit(model, data.batch('dev', 0, 2)[0]) if mode == 'ngpt_heavy_ball' else None
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
                    theoretical_cost=model_cost(c, batch=16))
        if mode != 'vanilla':
            info['coefficients'] = model.residual_metrics()
        if mode == 'ngpt_heavy_ball':
            info.update(initial_dev_ce=initial_dev, final_dev_ce=evaluate(model, data)[0],
                        initial_residual_state=initial_state,
                        final_residual_state=residual_audit(model, data.batch('dev', 0, 2)[0]))
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
    result['time_ratio_to_hb'] = m['ngpt_heavy_ball']['median_step_seconds']/m['heavy_ball']['median_step_seconds']
    result['time_ratio_to_vanilla'] = m['ngpt_heavy_ball']['median_step_seconds']/m['vanilla']['median_step_seconds']
    assert result['time_ratio_to_hb'] < 1.75
    assert result['time_ratio_to_vanilla'] < 1.75
    result['complete'] = True
    json_write(out/'summary.json', result)
    print('MOMENTUM_PROFILE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
