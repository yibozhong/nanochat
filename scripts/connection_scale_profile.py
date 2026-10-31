"""One fresh-process capacity probe or bounded 1B--2B training-step profile."""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import gc
import gzip
import json
import os
from pathlib import Path
import resource
import statistics
import time
import uuid

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from nanochat.experiment_logging import diagnostic_run

from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
from nanochat.packed_training import PackedData, seed_all, train_step, json_write
from scripts.connection_profile import summarize, trace_summary

GROUP = 'mhc-scale-1b-2b-20261030'
BATCH_TOKENS = 32768


def configure_attention():
    # Explicitly preserve the actual cuDNN SDPA implementation in the H200
    # small-model trace, rather than relying on the offline FA3 import failing.
    from nanochat import flash_attention
    flash_attention.USE_FA3 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(True)


def phased_step(model, opt, data, step, microbatch, *, events=False, labels=False):
    """Same operations as train_step, including accumulation and finite checks."""
    stamps = {}; cpu = {}
    @contextmanager
    def phase(name):
        if events:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
        tick = time.perf_counter()
        with record_function('phase/' + name) if labels else nullcontext():
            yield
        cpu[name] = cpu.get(name, 0.) + (time.perf_counter() - tick) * 1000
        if events:
            end.record(); stamps.setdefault(name, []).append((begin, end))
    rows = BATCH_TOKENS // model.config.sequence_length
    accum = rows // microbatch
    with phase('zero_grad'):
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device='cuda')
    for offset in range(0, rows, microbatch):
        with phase('data'):
            x, y = data.batch('train', step * rows + offset, microbatch)
        with phase('forward'):
            loss = model(x, y)
        with phase('backward'):
            (loss / accum).backward()
            total += loss.detach() / accum
    with phase('finite_check'):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite = torch.stack([torch.isfinite(g).all() for g in grads]).all()
        if not bool(finite) or not torch.isfinite(total):
            raise FloatingPointError('Nonfinite loss/gradient before scale-profile update')
    with phase('optimizer'):
        opt.step()
    with phase('loss_item'):
        value = total.item()
    if events:
        torch.cuda.synchronize()
        return value, dict(cpu_wall_ms=cpu, stream_elapsed_ms={
            k: sum(a.elapsed_time(b) for a, b in pairs) for k, pairs in stamps.items()})
    return value


def memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                device_total_bytes=total, device_free_after_step_bytes=free,
                cpu_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['baseline', 'mhc'], required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--microbatch', type=int, required=True)
    parser.add_argument('--repetition', type=int, default=0)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-dir', default='../data')
    args = parser.parse_args()
    assert args.width in (2048, 2560, 3072) and args.depth == 16
    assert args.microbatch in (16, 8, 4, 2, 1)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention(); seed_all(42)
    cfg = ConnectionConfig(variant=args.variant, depth=args.depth, width=args.width,
        heads=args.width // 128, connection_backend='nvidia',
        fused_connection_grad=False, activation_checkpoint=False)
    source = json.loads(Path('source.json').read_text())
    props = torch.cuda.get_device_properties(0)
    ident = uuid.uuid4().hex[:8]
    settings = dict(source=source, model=asdict(cfg), seed=42,
        microbatch=args.microbatch, grad_accum=16 // args.microbatch, batch_tokens=BATCH_TOKENS,
        repetition=args.repetition, probe=args.probe, gpu=props.name, gpu_uuid=str(props.uuid),
        gpu_capability=list(torch.cuda.get_device_capability()), sm_count=props.multi_processor_count,
        attention_backend='SDPA_CUDNN_ATTENTION_only', torch=torch.__version__, cuda=torch.version.cuda,
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        optimizer='native MuonAdamW with existing batch/width scaling',
        warmup_steps=0 if args.probe else 20, timed_steps=0 if args.probe else 60,
        event_steps=0 if args.probe else 4, profiler_steps=0 if args.probe else 2,
        note='Temporary throughput updates only; no checkpoint or quality comparison')
    kwargs = dict(project='nanochat', entity='yibozhong657-none', group=GROUP,
        id=ident, name=f'{"capacity" if args.probe else "profile"}-d{args.width}-{args.variant}-r{args.repetition}-mb{args.microbatch}',
        job_type='capacity-probe' if args.probe else 'profiling', config=settings)
    run=diagnostic_run(**kwargs)
    step = 0; tick = time.perf_counter()
    try:
        model = ConnectionModel(cfg).prepare(compile_blocks=True)
        opt, wd = optimizer_for(model, BATCH_TOKENS)
        settings['parameters'] = sum(p.numel() for p in model.parameters())
        settings['parameter_tensors'] = len(list(model.parameters()))
        run.config.update({k: settings[k] for k in ('parameters', 'parameter_tensors')})
        json_write(out / 'config.json', settings)
        data = PackedData(args.data_dir)
        assert data.manifest['sequence_length'] == cfg.sequence_length
        def ordinary():
            nonlocal step
            value = train_step(model, opt, data, step, args.microbatch, BATCH_TOKENS)
            step += 1
            return value
        if args.probe:
            for _ in range(4):
                schedule(opt, step, 30518, wd); loss = ordinary()
            torch.cuda.synchronize()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            mem = memory_snapshot()
            safe = (mem['peak_reserved_bytes'] <= .90 * mem['device_total_bytes']
                    and mem['device_free_after_step_bytes'] >= 4 * 1024**3)
            result = dict(complete=True, status='fits' if safe else 'insufficient_headroom',
                run_id=ident, source=source, config=settings, memory=mem,
                optimizer_updates=step, training_tokens=step * BATCH_TOKENS, last_train_ce=loss)
        else:
            print(f'Warmup D{args.width} {args.variant}', flush=True)
            for _ in range(20):
                schedule(opt, step, 30518, wd); loss = ordinary()
            gc.collect(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(60):
                schedule(opt, step, 30518, wd)
                torch.cuda.synchronize(); start = time.perf_counter()
                loss = ordinary(); torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
                if (i+1) % 20 == 0:
                    print(f'Timed {i+1}/60 mean={statistics.mean(times):.3f} ms', flush=True)
            mem = memory_snapshot()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            phases = []
            for _ in range(4):
                schedule(opt, step, 30518, wd)
                loss, measurements = phased_step(model, opt, data, step, args.microbatch, events=True)
                phases.append(measurements); step += 1
            phase_means = {axis: {name: statistics.mean(v[axis][name] for v in phases)
                                 for name in phases[0][axis]} for axis in phases[0]}
            print('CUDA trace', flush=True); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=False, with_stack=False, profile_memory=False) as prof:
                for _ in range(2):
                    schedule(opt, step, 30518, wd)
                    loss = phased_step(model, opt, data, step, args.microbatch, labels=True)
                    step += 1; torch.cuda.synchronize(); prof.step()
            trace = out / 'trace.json'; prof.export_chrome_trace(str(trace))
            attribution = trace_summary(trace, 2)
            with trace.open('rb') as src, gzip.open(out / 'trace.json.gz', 'wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            trace.unlink()
            assert step == 86
            result = dict(complete=True, status='complete', run_id=ident, source=source, config=settings,
                optimizer_updates=step, training_tokens=step * BATCH_TOKENS,
                normal_ms=summarize(times), normal_timing_memory=mem,
                phase_means=phase_means, phase_samples=phases, cuda_profile=attribution,
                last_train_ce=loss, elapsed_seconds=time.perf_counter() - tick, pretraining_launched=False)
        json_write(out / 'summary.json', result)
        run.summary.update(dict(complete=True, status=result['status'], optimizer_updates=step,
            training_tokens=step * BATCH_TOKENS, peak_reserved_bytes=result.get('memory', result.get('normal_timing_memory'))['peak_reserved_bytes']))
        if not args.probe:
            run.summary.update(dict(normal_step_ms=result['normal_ms']['mean'],
                                    tokens_per_second=BATCH_TOKENS * 1000 / result['normal_ms']['mean']))
        run.save(str(out / '*.json'), base_path=str(out.parent), policy='now'); run.finish()
        print(json.dumps({k: v for k, v in result.items() if k in ('status', 'run_id', 'optimizer_updates')}), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        if not args.probe:
            run.finish(exit_code=1); raise
        result = dict(complete=True, status='oom', run_id=ident, source=source, config=settings,
            optimizer_updates=step, training_tokens=step * BATCH_TOKENS, error=str(exc))
        json_write(out / 'summary.json', result)
        run.summary.update(dict(complete=True, status='oom', optimizer_updates=step, training_tokens=step * BATCH_TOKENS))
        run.finish(exit_code=1)
        print('Capacity probe OOM; parent will lower the common microbatch', flush=True)
    except BaseException:
        run.finish(exit_code=1); raise


if __name__ == '__main__':
    main()
