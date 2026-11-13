"""Production training, exact restore and controlled-backward replay on a GPU.

No W&B; scratch checkpoints are removed in finally, never transferred.
Only CUDA OOM/capacity overflow allow the controller to try a smaller batch.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import time

from policy import CAPACITY_BYTES, STEPS
from replay import cce_tape


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
    import numpy as np
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def differences(a, b, prefix=''):
    """Small numeric evidence, never tensor dumps or content hashes."""
    import torch
    rows = []
    if isinstance(a, torch.Tensor):
        b = b.detach().cpu()
        if a.dtype != b.dtype or not torch.equal(a, b):
            delta = a.double()-b.double()
            rows.append(dict(name=prefix, dtype=str(a.dtype), restored_dtype=str(b.dtype),
                unequal=int((a != b).sum()), elements=a.numel(), max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm()/a.double().norm().clamp_min(1e-30)).item()))
    elif isinstance(a, dict):
        for key in a:
            rows.extend(differences(a[key], b[key], prefix+'/'+str(key)))
    elif isinstance(a, (tuple, list)):
        for i, (x, y) in enumerate(zip(a, b)):
            rows.extend(differences(x, y, prefix+'/'+str(i)))
    elif a != b:
        rows.append(dict(name=prefix, scalar_mismatch=True, expected=a, actual=b))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('fixed_hca', 'dynamic_hca'), required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backend', choices=('reference','triton'), default='reference')
    p.add_argument('--block-size', type=int, default=16)
    a = p.parse_args()
    import torch
    from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule
    from nanochat.packed_training import PackedData, seed_all, train_step, snapshot, atomic_save, restore_rng, rng_state, json_write
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (9, 0)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads(Path('source.json').read_text())
    c = ConnectionConfig(variant='baseline', streams=1, attention_mode=a.mode,
                         hca_block_size=a.block_size, hca_ratio_loss_weight=1., hca_window=256,
                         hca_noise=True, hca_temperature=1., hca_backend=a.backend)
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
            with cce_tape() if step == 2 else nullcontext() as entries:
                losses.append(train_step(model, opt, data, step, microbatch=a.microbatch))
                if step == 2:
                    tape = entries
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
        expected_rng = cpu_state(rng_state())
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
        state = torch.load(scratch/'latest.pt', map_location='cpu', weights_only=False)
        assert state['step'] == 2 and state['next_data_row'] == 32 and state['config'] == asdict(c)
        model = build_connection_model(c).prepare(compile_blocks=False)
        opt, wd = optimizer_for(model)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        result['loaded_model_equal'] = same(state['model'], model.state_dict())
        result['loaded_optimizer_equal'] = same(state['optimizer'], opt.state_dict())
        if not (result['loaded_model_equal'] and result['loaded_optimizer_equal']):
            result.update(status='failed_load', model_differences=differences(state['model'], model.state_dict()),
                          optimizer_differences=differences(state['optimizer'], opt.state_dict()))
            json_write(out, result)
        assert result['loaded_model_equal'] and result['loaded_optimizer_equal'], 'Checkpoint load changed stored values'
        restore_rng(state['rng'])
        result['loaded_rng_equal'] = same(state['rng'], rng_state())
        result['loaded_data_row_equal'] = state['next_data_row'] == 32
        assert result['loaded_rng_equal'] and result['loaded_data_row_equal']
        del state
        schedule(opt, 2, STEPS, wd)
        replay_loss = train_step(model, opt, data, 2, microbatch=a.microbatch)
        assert replay_loss == losses[2], (replay_loss, losses[2])
        model_equal, opt_equal = same(expected_model, model.state_dict()), same(expected_opt, opt.state_dict())
        result.update(parameters=parameters, steps=3, replayed_step=2,
            actual_cce_replay_model_equal=model_equal, actual_cce_replay_optimizer_equal=opt_equal,
            actual_cce_replay_loss_equal=replay_loss == losses[2],
            all_gradients_finite=True, losses=losses, metrics=metrics,
            pool_gradient_nonzero=pool_nonzero, router_gradient_nonzero=router_nonzero,
            # Step2 records diagnostic tensors; use the warmed, uninstrumented
            # step1 for a preliminary estimate and retain all raw durations.
            steady_step_seconds=times[1], step_seconds=times,
            timing_note='Single warmed uninstrumented update; step2 additionally records CCE replay inputs/gradients',
            estimated_train_only_hours=STEPS*times[1]/3600)
        if not (model_equal and opt_equal):
            result['model_differences'] = differences(expected_model, model.state_dict())
            result['optimizer_differences'] = differences(expected_opt, opt.state_dict())
        # A fresh reload executes the full backward/update with the exact CCE
        # gradients from the original update. Checkpoints, RNG, token targets,
        # CCE inputs/arguments, loss scale, all weights and all optimizer state
        # must match exactly. No post-update tolerance is widened.
        del model, opt
        gc.collect(); torch.cuda.empty_cache()
        state = torch.load(scratch/'latest.pt', map_location='cpu', weights_only=False)
        model = build_connection_model(c).prepare(compile_blocks=False)
        opt, wd = optimizer_for(model)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        assert same(state['model'], model.state_dict()) and same(state['optimizer'], opt.state_dict())
        restore_rng(state['rng']); assert same(state['rng'], rng_state()); del state
        schedule(opt, 2, STEPS, wd)
        with cce_tape(tape):
            controlled_loss = train_step(model, opt, data, 2, microbatch=a.microbatch)
        controlled_model = same(expected_model, model.state_dict())
        controlled_opt = same(expected_opt, opt.state_dict())
        controlled_rng = same(expected_rng, rng_state())
        result.update(restore_contract='exact_state_rng_data_and_fixed_cce_gradient_replay',
            controlled_backward_loss_equal=controlled_loss == losses[2],
            controlled_backward_model_equal=controlled_model, controlled_backward_optimizer_equal=controlled_opt,
            controlled_backward_rng_equal=controlled_rng,
            cce_replay_microbatches=len(tape),
            complete=controlled_model and controlled_opt and controlled_rng and controlled_loss == losses[2],
            status='passed' if controlled_model and controlled_opt and controlled_rng and controlled_loss == losses[2] else 'failed_controlled_replay',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        json_write(out, result)
        print(json.dumps(result, allow_nan=False), flush=True)
        assert result['complete'], 'Controlled CCE-gradient replay still differs; inspect full backward/optimizer'
        assert result['loaded_model_equal'] and result['loaded_optimizer_equal']
        assert result['loaded_rng_equal'] and result['loaded_data_row_equal']
        assert result['cce_replay_microbatches'] == 16//a.microbatch
    except torch.cuda.OutOfMemoryError:
        result.update(status='oom', complete=False)
        json_write(out, result)
    finally:
        shutil.rmtree(scratch)


if __name__ == '__main__':
    main()
