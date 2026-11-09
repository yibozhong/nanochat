"""One fresh-process full-attention context-length timing or capacity probe."""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule, CONNECTION_VARIANTS
from nanochat.packed_training import PackedData, json_write, seed_all, train_step

BATCH_TOKENS = 32768
MIN_GPU_BYTES = 22000 * 1024**2


class TimingData:
    """Re-segment existing packed tokens for performance, not long-document quality.

    Original 2049-token rows are independent packed examples. Flattening them
    does not recover original document continuity. All variants use the same
    deterministic examples at a given context length; no new cache is saved.
    """
    def __init__(self, root, length):
        self.data = PackedData(root)
        self.flat = self.data.arrays['train'].reshape(-1)
        self.length = length

    def batch(self, split, start, n, device='cuda'):
        assert split == 'train'
        begin, end = start * (self.length + 1), (start + n) * (self.length + 1)
        assert end <= self.flat.size
        a = np.array(self.flat[begin:end].reshape(n, self.length + 1), dtype=np.int64, copy=True)
        rows = torch.from_numpy(a).to(device)
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=CONNECTION_VARIANTS, required=True)
    p.add_argument('--mlp-activation', choices=('relu2','pairwise'), default='relu2')
    p.add_argument('--pairwise-backend', choices=('triton','torch'), default='triton')
    p.add_argument('--attention-backend', choices=('sdpa_cudnn','fa3','fa2','hardware_auto'), default='sdpa_cudnn')
    p.add_argument('--sequence-length', type=int, choices=(2048, 4096, 8192, 16384), required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--repetition', type=int, default=0)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--compact-routing', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--data-dir', default='../data')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = BATCH_TOKENS // a.sequence_length
    assert 0 < a.microbatch <= rows and rows % a.microbatch == 0
    assert not os.environ.get('WANDB_API_KEY') and not Path('../.netrc').exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    actual_attention_backend = configure_attention(a.attention_backend)
    seed_all(42)
    props = torch.cuda.get_device_properties(0)
    cfg = ConnectionConfig(variant=a.variant, mlp_activation=a.mlp_activation, pairwise_backend=a.pairwise_backend, sequence_length=a.sequence_length, compact_routing=a.compact_routing,
                           streams=1 if a.variant in ('skip_scale','attnres_scale') or a.mlp_activation=='pairwise' else 4)
    record = dict(complete=False, variant=a.variant, sequence_length=a.sequence_length,
        microbatch=a.microbatch, grad_accum=rows//a.microbatch, batch_tokens=BATCH_TOKENS,
        repetition=a.repetition, probe=a.probe, model=asdict(cfg), logging_backend='local',
        gpu=props.name, gpu_uuid=str(props.uuid), device_total_bytes=props.total_memory,
        torch=torch.__version__, cuda=torch.version.cuda, attention_backend=a.attention_backend,
        actual_attention_backend=actual_attention_backend,
        minimum_gpu_bytes=MIN_GPU_BYTES, memory_limit_bytes=int(.9*MIN_GPU_BYTES),
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        source=json.loads(Path('source.json').read_text()), seed=42,
        data_note=TimingData.__doc__, timing_includes='CPU data copy/H2D, CCE, backward, finite checks, MuonAdamW',
        warmup_steps=4 if a.probe else 20, timed_steps=0 if a.probe else 40)
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = build_connection_model(cfg).prepare(compile_blocks=True)
        match_vanilla_initialization(model, 42)
        opt, wd = optimizer_for(model, BATCH_TOKENS)
        data = TimingData(a.data_dir, a.sequence_length)
        # Verify adjacent targets and exact deterministic slicing before timing.
        x, y = data.batch('train', 0, a.microbatch, device='cpu')
        assert x.shape == y.shape == (a.microbatch, a.sequence_length)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert 0 <= x.min() and x.max() < cfg.vocab_size
        record['input_selection'] = dict(data_dir=str(Path(a.data_dir).resolve()), start_row=0, rows=a.microbatch, sequence_length=a.sequence_length, seed=42)
        record['parameters'] = sum(p.numel() for p in model.parameters())
        del x, y
        torch.cuda.reset_peak_memory_stats()
        losses = []; times = []; first_step = None; started = time.perf_counter()
        for step in range(record['warmup_steps'] + record['timed_steps']):
            schedule(opt, step, 305176, wd)
            if step == record['warmup_steps']:
                gc.collect()
                torch.cuda.synchronize()
                record['setup_and_warmup_seconds'] = time.perf_counter()-started
            torch.cuda.synchronize(); tick = time.perf_counter()
            loss = train_step(model, opt, data, step, a.microbatch, BATCH_TOKENS)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
            losses.append(loss)
            if step == 0: first_step = elapsed
            if step >= record['warmup_steps']: times.append(elapsed)
            if step == 0 or (step+1) % 20 == 0:
                print(json.dumps(dict(step=step+1, loss=loss, seconds=elapsed)), flush=True)
        assert all(math.isfinite(x) for x in losses)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        record.update(first_step_including_compilation_seconds=first_step,
            losses=losses, optimizer_updates=len(losses), training_tokens=len(losses)*BATCH_TOKENS,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record['fits_minimum_gpu_budget'] = record['peak_reserved_bytes'] < record['memory_limit_bytes']
        if not a.probe:
            mean = statistics.mean(times)
            record.update(step_seconds=times, mean_seconds=mean, median_seconds=statistics.median(times),
                stdev_seconds=statistics.stdev(times), tokens_per_second=BATCH_TOKENS/mean,
                estimated_10b_training_hours=305176*mean/3600)
            # Same compiled shape as timing; trained branches are nonzero.
            # Run after saving memory stats, so correctness checks do not inflate them.
            model.eval()
            with torch.no_grad():
                x, _ = data.batch('train', 0, a.microbatch)
                changed = x.clone(); half = a.sequence_length//2
                changed[0, half:] = (changed[0, half:]+17) % cfg.vocab_size
                before = model.hidden(x); after = model.hidden(changed)
                delta = (before[0, :half]-after[0, :half]).abs().max().item()
                record['causal_prefix_max_abs'] = delta
                assert delta == 0, record
                if a.microbatch > 1:
                    assert torch.equal(before[1:], after[1:])
            record['causality_passed'] = True
        record.update(complete=True, status='fits' if record['fits_minimum_gpu_budget'] else 'over_budget')
    except torch.cuda.OutOfMemoryError as exc:
        record.update(status='oom', error=str(exc)[:2000], fits_minimum_gpu_budget=False)
    except Exception as exc:
        record.update(status='error', error=repr(exc)); json_write(output, record); raise
    json_write(output, record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('source','model','losses','step_seconds')}), flush=True)


if __name__ == '__main__':
    main()
