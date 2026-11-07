"""Bounded full-step ABBA comparison for VO16; no formal training."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.hash_qkv_bench import write

def profile(args):
    import gc
    import math
    import statistics
    import numpy as np
    import torch
    from nanochat.connection_experiment import configure_attention
    from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
    from nanochat.hash_vo import HashVOModel
    from nanochat.packed_training import PackedData, seed_all
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa3'); seed_all(42)
    data = PackedData(args.data)
    config = ConnectionConfig()
    table = np.load(args.lookup)
    model = (HashVOModel(config, table) if args.variant == 'hash' else ConnectionModel(config)).prepare(compile_blocks=True)
    opt, wd = optimizer_for(model)
    props = torch.cuda.get_device_properties(0)
    report = dict(complete=False, formal_training=False, variant=args.variant, repetition=args.repetition,
        gpu=props.name, gpu_uuid=str(props.uuid), capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__, attention='pinned FA3', deepgemm='2.2.0',
        seed=42, parameters=sum(p.numel() for p in model.parameters()),
        buckets=16 if args.variant == 'hash' else 1, hash_method='frozen training-frequency-balanced',
        hash_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        width=512, depth=8, heads=4, sequence_length=2048, microbatch=16, global_tokens=32768,
        warmup_steps=20, timed_steps=40, schedule_steps=305176,
        timing_includes='CPU data/lookup/sort/plan/H2D, compiled model, CCE, backward, finite checks, original MuonAdamW',
        statistics_preparation_in_timing=False, batch_plan_preparation_in_timing=True,
        independent_V_O_Muon=True, source=json.loads(Path('source.json').read_text()))
    report['expected_parameters'] = 121634816 if args.variant == 'hash' else 58720256
    assert report['parameters'] == report['expected_parameters'], report
    losses, times, plan_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for step in range(60):
            schedule(opt, step, 305176, wd)
            if step == 20:
                gc.collect(); torch.cuda.synchronize()
                report['warmup_and_compile_seconds'] = time.perf_counter() - started
            torch.cuda.synchronize(); tick = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            x_cpu, y_cpu = data.batch('train', step * 16, 16, device='cpu')
            if step == 0:
                report['input_sha256'] = hashlib.sha256(x_cpu.numpy().tobytes()+y_cpu.numpy().tobytes()).hexdigest()
            plan_tick = time.perf_counter()
            routing = model.make_routing_plan(x_cpu) if args.variant == 'hash' else None
            plan_elapsed = time.perf_counter() - plan_tick
            x, y = x_cpu.cuda(), y_cpu.cuda()
            loss = model(x, y, routing=routing) if args.variant == 'hash' else model(x, y)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            if not bool(finite) or not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite loss/gradient before update {step+1}')
            opt.step()
            value = loss.item()
            torch.cuda.synchronize(); elapsed = time.perf_counter() - tick
            losses.append(value)
            if step >= 20:
                times.append(elapsed); plan_times.append(plan_elapsed)
            if step == 0 or (step + 1) % 20 == 0:
                print(json.dumps(dict(variant=args.variant, repetition=args.repetition,
                    step=step+1, ce=value, ms=1000*elapsed)), flush=True)
        assert all(math.isfinite(v) for v in losses)
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        report.update(optimizer_updates=60, training_tokens=60*32768, losses=losses,
            step_ms=[1000*s for s in times], mean_ms=1000*statistics.mean(times),
            median_ms=1000*statistics.median(times), stdev_ms=1000*statistics.stdev(times),
            plan_cpu_and_transfer_ms=1000*statistics.mean(plan_times),
            tokens_per_second=32768/statistics.mean(times),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        # Exercise real trained, nonzero branches at production length.
        model.eval()
        with torch.no_grad():
            x, _ = data.batch('train', 0, 2)
            changed = x.clone(); changed[0, 1024:] = (changed[0, 1024:] + 17) % 32768
            before, after = model.hidden(x), model.hidden(changed)
            assert torch.equal(before[0, :1024], after[0, :1024])
            assert torch.equal(before[1], after[1])
        report.update(complete=True, trained_causality_passed=True)
    except BaseException as error:
        report.update(error=repr(error), completed_updates=len(losses), losses=losses)
        write(args.output, report)
        raise
    write(args.output, report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='../data')
    p.add_argument('--lookup',default='research/connections/hash-qkv/lookup.npy')
    p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=('baseline','hash'),required=True)
    p.add_argument('--repetition',type=int,default=0)
    profile(p.parse_args())
