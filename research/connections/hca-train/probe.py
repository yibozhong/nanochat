"""Full 8-layer/2K training and exact next-update restore on a GPU worker.

No W&B; scratch checkpoints are removed in finally, never transferred.
Only CUDA OOM/capacity overflow allow the controller to try a smaller batch.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import time

from policy import CAPACITY_BYTES, STEPS, validate_probe


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return value


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('fixed_hca', 'dynamic_hca'), required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
    from nanochat.packed_training import PackedData, seed_all, train_step, snapshot, atomic_save, restore_rng, json_write
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads(Path('source.json').read_text())
    c = ConnectionConfig(variant='baseline', streams=1, attention_mode=a.mode,
                         hca_block_size=16, hca_ratio_loss_weight=1., hca_window=256,
                         hca_noise=True, hca_temperature=1.)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path('../hca-diagnostic-checkpoints') / (a.mode+'-'+str(a.microbatch))
    assert not scratch.exists(), 'Inspect an interrupted probe; do not overwrite it'
    scratch.mkdir(parents=True)
    result = dict(complete=False, status='running', formal_training=False, mode=a.mode,
                  microbatch=a.microbatch, commit=source['commit'], config=asdict(c),
                  gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
    json_write(out, result)
    try:
        seed_all(42)
        data = PackedData('../data')
        assert data.manifest['sequence_length'] == 2048
        model = build_connection_model(c).prepare(compile_blocks=False)
        opt, wd = optimizer_for(model)
        torch.cuda.reset_peak_memory_stats()
        losses, times = [], []
        for step in range(3):
            schedule(opt, step, STEPS, wd)
            if step == 2:
                atomic_save(snapshot(model, opt, step, config=asdict(c), next_data_row=step*16), scratch/'latest.pt')
            torch.cuda.synchronize(); begin = time.perf_counter()
            losses.append(train_step(model, opt, data, step, microbatch=a.microbatch))
            torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
            if torch.cuda.max_memory_reserved() > CAPACITY_BYTES:
                result.update(status='over_budget', peak_reserved_bytes=torch.cuda.max_memory_reserved())
                json_write(out, result); return
        # Inspect the real gradients retained from the third update.
        pool_nonzero = all(b.attn.pool_score.weight.grad is not None and bool(b.attn.pool_score.weight.grad.abs().max() > 0) for b in model.blocks)
        router_nonzero = (all(b.attn.boundary.weight.grad is not None and bool(b.attn.boundary.weight.grad.abs().max() > 0) for b in model.blocks)
                          if a.mode == 'dynamic_hca' else None)
        parameters = sum(p.numel() for p in model.parameters())
        metrics = model.last_train_metrics
        expected_model = cpu_state(model.state_dict())
        expected_opt = cpu_state(opt.state_dict())
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
        state = torch.load(scratch/'latest.pt', map_location='cpu', weights_only=False)
        assert state['step'] == 2 and state['next_data_row'] == 32 and state['config'] == asdict(c)
        model = build_connection_model(c).prepare(compile_blocks=False)
        opt, wd = optimizer_for(model)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng']); del state
        schedule(opt, 2, STEPS, wd)
        replay_loss = train_step(model, opt, data, 2, microbatch=a.microbatch)
        assert replay_loss == losses[2], (replay_loss, losses[2])
        model_equal, opt_equal = same(expected_model, model.state_dict()), same(expected_opt, opt.state_dict())
        result.update(complete=True, status='passed', parameters=parameters,
            steps=3, replayed_step=2, replay_model_equal=model_equal, replay_optimizer_equal=opt_equal,
            all_gradients_finite=True, losses=losses, metrics=metrics,
            pool_gradient_nonzero=pool_nonzero, router_gradient_nonzero=router_nonzero,
            steady_step_seconds=sum(times[1:])/2, step_seconds=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            estimated_train_only_hours=STEPS*sum(times[1:])/2/3600)
        validate_probe(result, a.mode, a.microbatch, source['commit'])
        json_write(out, result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except torch.cuda.OutOfMemoryError:
        result.update(status='oom', complete=False)
        json_write(out, result)
    finally:
        shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
