"""Bounded QKV hash validation and same-GPU speed comparison; no formal run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(args):
    import numpy as np
    from nanochat.hash_qkv import balanced_lookup
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    path = Path(args.data)/'train.bin'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['files']['train']['sha256']
    raw = np.memmap(path, dtype=np.uint16, mode='r',
        shape=(manifest['files']['train']['rows'], manifest['sequence_length'] + 1))
    counts = np.bincount(raw[:, :-1].reshape(-1), minlength=32768)
    table = balanced_lookup(counts, 16)
    np.save(Path(args.output)/'lookup.npy', table)
    np.save(Path(args.output)/'train-frequency.npy', counts)
    masses = np.bincount(table, weights=counts, minlength=16)
    random_table = np.random.default_rng(42).permutation(32768) % 16
    batches = []
    for step in range(60):
        ids = raw[step * 16:(step + 1) * 16, :-1].reshape(-1)
        row = dict(step=step)
        for name, mapping in [('balanced', table), ('random', random_table)]:
            sizes = np.bincount(mapping[ids], minlength=16)
            row[name] = dict(counts=sizes.tolist(), cv=float(sizes.std()/sizes.mean()),
                max_over_mean=float(sizes.max()/sizes.mean()), min=int(sizes.min()), max=int(sizes.max()))
        batches.append(row)
    result = dict(method='descending training-input-frequency, greedy least-loaded bucket; frozen',
        buckets=16, train_input_tokens=int(counts.sum()), training_file_sha256=digest,
        statistics_source='all 2048 rows of the existing 4.194304M-token TRAIN gate cache; excludes final labels',
        dev_or_test_used=False, vocabulary=32768, hash_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
        frequency_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        corpus_bucket_mass=masses.astype(np.int64).tolist(), corpus_cv=float(masses.std()/masses.mean()),
        batches=batches)
    write(Path(args.output)/'balanced-hash.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'batches'}), flush=True)


def profile(args):
    import gc
    import math
    import statistics
    import numpy as np
    import torch
    from nanochat.connection_experiment import configure_attention
    from nanochat.connection_model import ConnectionConfig, ConnectionModel, optimizer_for, schedule
    from nanochat.hash_qkv import HashQKVModel
    from nanochat.packed_training import PackedData, seed_all
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configure_attention('fa3'); seed_all(42)
    data = PackedData(args.data)
    config = ConnectionConfig()
    table = np.load(args.lookup)
    model = (HashQKVModel(config, table) if args.variant == 'hash' else ConnectionModel(config)).prepare(compile_blocks=True)
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
        independent_Q_K_V_Muon=True, source=json.loads(Path('source.json').read_text()))
    report['expected_parameters'] = 153092096 if args.variant == 'hash' else 58720256
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


def micro(args):
    import statistics
    import numpy as np
    import torch
    from nanochat.hash_qkv import HashQKVProjection, make_plan
    from nanochat.packed_training import PackedData, seed_all
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; seed_all(42)
    table = np.load(args.lookup)
    xids, _ = PackedData(args.data).batch('train', 0, 16, device='cpu')
    plan = make_plan(xids.numpy(), table)
    originals = [torch.randn(512, 512) / 512**.5 for _ in range(3)]
    model = HashQKVProjection(originals).cuda()
    forward = torch.compile(model.forward, dynamic=False, fullgraph=True)
    x = torch.randn(32768, 512, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(32768, 1536, device='cuda', dtype=torch.bfloat16)
    samples = []
    for it in range(28):
        model.zero_grad(set_to_none=True); x.grad = None
        torch.cuda.synchronize(); begin = time.perf_counter()
        forward(x, plan).backward(dy)
        torch.cuda.synchronize(); duration = time.perf_counter() - begin
        if it >= 8: samples.append(1000*duration)
    # Dense fused QKV has the same useful matrix FLOPs and width.
    dense = torch.nn.Parameter(torch.cat(originals).cuda())
    dense_fn = torch.compile(lambda a,w: torch.nn.functional.linear(a,w.to(a.dtype)), fullgraph=True)
    dense_samples = []
    for it in range(28):
        dense.grad = None; x.grad = None
        torch.cuda.synchronize(); begin = time.perf_counter()
        dense_fn(x, dense).backward(dy)
        torch.cuda.synchronize(); duration = time.perf_counter() - begin
        if it >= 8: dense_samples.append(1000*duration)
    write(args.output, dict(complete=True, gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        tokens=32768, width=512, output_width=1536, buckets=16,
        padded_tokens=plan.pack.numel(), padding_ratio=plan.pack.numel()/32768,
        hash_qkv_fwd_bwd_ms=statistics.mean(samples), dense_fused_qkv_fwd_bwd_ms=statistics.mean(dense_samples),
        hash_samples_ms=samples, dense_samples_ms=dense_samples,
        includes='weight packing/cast, activation pack/unpack, forward and all gradients',
        excludes='optimizer, per-batch routing-plan preparation (included in the full-model timing)',
        logical_forward_flops=2*32768*512*1536))


def run_child(args, root, name):
    print('BEGIN ' + name, flush=True)
    with (root/(name+'.log')).open('w') as log:
        process = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        code = process.wait()
    if code: raise RuntimeError(f'{name} exited {code}; report preserved, no automatic training')


def main(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    base = ['-m', 'scripts.hash_qkv_bench', '--data', args.data]
    report = dict(complete=False, formal_training=False, automatic_training=False,
        source=json.loads(Path('source.json').read_text()), profiles=[])
    try:
        run_child(base+['--mode','prepare','--output',str(root)], root, 'balanced-hash')
        lookup = str(root/'lookup.npy')
        run_child(['-m','pytest','-v','tests/test_hash_qkv.py','--junitxml='+str(root/'tests.xml')], root, 'checks')
        suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
        counts = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert counts == dict(tests=12, failures=0, errors=0, skipped=0), counts
        report['checks'] = counts
        for index, variant in enumerate(('baseline', 'hash', 'hash', 'baseline')):
            name = f'profile-{index}-{variant}'
            output = root/(name+'.json')
            run_child(base+['--mode','profile','--variant',variant,'--repetition',str(index),
                '--lookup',lookup,'--output',str(output)], root, name)
            value = json.loads(output.read_text()); assert value['complete']
            report['profiles'].append(value)
            write(root/'report.json', report)
        run_child(base+['--mode','micro','--lookup',lookup,'--output',str(root/'micro.json')], root, 'micro')
        assert len({r['gpu_uuid'] for r in report['profiles']}) == 1
        assert len({r['input_sha256'] for r in report['profiles']}) == 1
        report['balanced_hash'] = json.loads((root/'balanced-hash.json').read_text())
        report['micro'] = json.loads((root/'micro.json').read_text())
        b = sum(r['mean_ms'] for r in report['profiles'] if r['variant']=='baseline')/2
        h = sum(r['mean_ms'] for r in report['profiles'] if r['variant']=='hash')/2
        report.update(complete=True, baseline_ms=b, hash_qkv_ms=h, overhead_percent=100*(h/b-1),
            limitations='Short NTP validation and speed only; no quality or long-run stability claim; no checkpoint returned.')
    except BaseException as error:
        report['error'] = repr(error); write(root/'report.json', report); raise
    write(root/'report.json', report)
    print(json.dumps(dict(complete=True, checks=report['checks'], baseline_ms=b, hash_qkv_ms=h,
        overhead_percent=report['overhead_percent'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('all','prepare','profile','micro'), default='all')
    parser.add_argument('--data', default='../data')
    parser.add_argument('--output', default='results/hash-qkv')
    parser.add_argument('--lookup')
    parser.add_argument('--variant', choices=('baseline','hash'))
    parser.add_argument('--repetition', type=int, default=0)
    args = parser.parse_args()
    {'all':main, 'prepare':prepare, 'profile':profile, 'micro':micro}[args.mode](args)
