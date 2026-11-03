"""Same-head-budget stream-attention timing; synthetic data, no W&B."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from nanochat.connection_experiment import configure_attention, match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import json_write, seed_all, train_step


class SyntheticData:
    def __init__(self, vocab, length):
        generator = torch.Generator(device='cpu').manual_seed(901)
        self.rows = torch.randint(vocab, (16, length+1), generator=generator)

    def batch(self, split, start, n, device='cuda'):
        assert split == 'train'
        indices = torch.arange(start, start+n) % len(self.rows)
        rows = self.rows[indices].to(device)
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline', 'shared', 'per_stream'), required=True)
    parser.add_argument('--repetition', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert not os.environ.get('WANDB_API_KEY') and not Path('../.netrc').exists()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('sdpa_cudnn'); seed_all(42)
    config = ConnectionConfig(variant='baseline' if args.mode=='baseline' else 'mhc',
        attention_mode='per_stream' if args.mode=='per_stream' else 'shared')
    model = ConnectionModel(config).prepare(compile_blocks=True)
    match_vanilla_initialization(model, 42)
    opt, wd = optimizer_for(model)
    data = SyntheticData(config.vocab_size, config.sequence_length)
    props = torch.cuda.get_device_properties(0)
    record = dict(complete=False, mode=args.mode, repetition=args.repetition, model=asdict(config),
        parameters=sum(p.numel() for p in model.parameters()), gpu=props.name, gpu_uuid=str(props.uuid),
        logging_backend='local', data='fixed synthetic NTP sequences; timing/smoke only, not quality',
        microbatch=16, grad_accum=1, batch_tokens=32768, warmup_steps=20, timed_steps=30,
        attention_backend='sdpa_cudnn', source=json.loads(Path('source.json').read_text()))
    # Multiply-add = 2 FLOPs. Causal pair count T(T+1)/2, forward only.
    # Excludes normalization/routing, activations, CCE softmax, optimizer and
    # padding/recomputation; equal GEMM FLOPs do not imply equal wall time.
    n, d, t, layers, vocab = 32768, config.width, config.sequence_length, config.depth, config.vocab_size
    record['forward_matmul_flops'] = dict(attention_projection=layers*8*n*d*d,
        causal_attention=layers*2*n*d*(t+1), mlp=layers*16*n*d*d, lm_head=2*n*d*vocab)
    record['forward_matmul_flops_total'] = sum(record['forward_matmul_flops'].values())
    record['train_matmul_flops_estimate'] = 3*record['forward_matmul_flops_total']
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(50):
        schedule(opt, step, 305176, wd)
        torch.cuda.synchronize(); begin = time.perf_counter()
        loss = train_step(model, opt, data, step, microbatch=16)
        torch.cuda.synchronize(); seconds = time.perf_counter()-begin
        losses.append(loss)
        if step >= 20: times.append(seconds)
        if step == 0 or (step+1)%10 == 0:
            print(json.dumps(dict(mode=args.mode, step=step+1, loss=loss, seconds=seconds)), flush=True)
    assert all(math.isfinite(loss) for loss in losses)
    record.update(complete=True, losses=losses, step_seconds=times,
        mean_seconds=statistics.mean(times), median_seconds=statistics.median(times),
        tokens_per_second=32768/statistics.mean(times),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    record['fits_22gb_budget'] = record['peak_reserved_bytes'] < .9*22000*1024**2
    # Check real compiled 2K/micro16 causal/batch isolation after training,
    # outside the timing/memory measurement. Use hidden to avoid dense logits.
    with torch.no_grad():
        x, _ = data.batch('train', 0, 16)
        changed = x.clone(); changed[0, 1024:] = (changed[0, 1024:]+17)%vocab
        before, after = model.hidden(x), model.hidden(changed)
        record['causal_prefix_max_abs'] = (before[0, :1024]-after[0, :1024]).abs().max().item()
        record['batch_isolation_max_abs'] = (before[1:]-after[1:]).abs().max().item()
    assert record['causal_prefix_max_abs'] == record['batch_isolation_max_abs'] == 0, record
    json_write(args.output, record)


if __name__ == '__main__': main()
