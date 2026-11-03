"""Bounded same-host packing comparison, including byte and source-state parity."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

sys.path.insert(0, str(Path('nanochat-src').resolve()))
os.environ['NANOCHAT_BASE_DIR'] = str(Path('cache').resolve())
import numpy as np
import torch
import nanochat.dataset as dataset
import nanochat.dataloader as native
from nanochat.tokenizer import get_tokenizer
import fast_pack


def check_selection():
    rng = random.Random(42)
    indexed = fast_pack.LengthBuffer()
    reference = []
    identity = 0
    for step in range(5000):
        if not reference or rng.random() < .55:
            # Repeated lengths exercise stable ties; large docs exercise cropping.
            tokens = [identity] * rng.choice((1, 2, 3, 7, 16, 63, 128, 2049, 4096))
            identity += 1
            reference.append(tokens)
            indexed.add(tokens)
        else:
            remaining = rng.randrange(1, 2050)
            fits = [i for i, doc in enumerate(reference) if len(doc) <= remaining]
            if fits:
                pick = max(fits, key=lambda i: len(reference[i]))
            else:
                pick = min(range(len(reference)), key=lambda i: len(reference[i]))
            assert indexed.take(remaining) is reference.pop(pick)
        assert indexed.count == len(reference)
    return 5000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('native', 'text_only', 'indexed4', 'indexed8'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=544)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    dataset.DATA_DIR = str(Path('raw').resolve())
    torch.set_num_threads(2)
    checks = check_selection()
    tokenizer = get_tokenizer()
    threads = 8 if a.mode == 'indexed8' else 4
    timers = dict(parquet_read_seconds=0., encode_seconds=0.)
    original_pf = native.pq.ParquetFile

    class TimedParquet:
        def __init__(self, *args, **kwargs):
            self.impl = original_pf(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.impl, name)

        def read_row_group(self, *args, **kwargs):
            if a.mode == 'text_only':
                kwargs['columns'] = ['text']
            start = time.perf_counter()
            result = self.impl.read_row_group(*args, **kwargs)
            timers['parquet_read_seconds'] += time.perf_counter() - start
            return result

    native.pq.ParquetFile = TimedParquet  # fast_pack shares this module in the process.
    encode = tokenizer.encode

    def timed_encode(*args, **kwargs):
        start = time.perf_counter()
        result = encode(*args, **kwargs)
        timers['encode_seconds'] += time.perf_counter() - start
        return result

    tokenizer.encode = timed_encode
    make = fast_pack.loader if a.mode.startswith('indexed') else native.tokenizing_distributed_data_loader_with_state_bos_bestfit
    gen = make(tokenizer, 16, 2048, 'train', device='cpu', tokenizer_threads=threads)
    digest = hashlib.sha256()
    states = hashlib.sha256()
    profiler = cProfile.Profile() if a.profile else None
    if profiler:
        profiler.enable()
    begin = time.perf_counter()
    warm = begin
    warm_timers = dict(timers)
    for i in range(a.steps):
        if i == a.warmup:
            warm = time.perf_counter()
            warm_timers = dict(timers)
        x, y, state = next(gen)
        rows = torch.cat((x, y[:, -1:]), dim=1).numpy().astype(np.uint16)
        assert np.array_equal(rows[:, 1:], y.numpy())
        digest.update(rows.tobytes())
        states.update(json.dumps(state, sort_keys=True).encode())
    end = time.perf_counter()
    if profiler:
        profiler.disable()
        import pstats
        with Path(a.output).with_suffix('.profile.txt').open('w') as output:
            pstats.Stats(profiler, stream=output).strip_dirs().sort_stats('cumtime').print_stats(30)
    elapsed = end - warm
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), 'unknown')
    result = dict(mode=a.mode, steps=a.steps, warmup=a.warmup,
                  total_seconds=end - begin, timed_seconds=elapsed,
                  tokens_per_second=(a.steps-a.warmup)*32768/elapsed,
                  train_prefix_sha256=digest.hexdigest(), source_states_sha256=states.hexdigest(),
                  final_source_state=state, tokenizer_threads=threads, selection_checks=checks,
                  measured_components={key: value-warm_timers[key] for key, value in timers.items()},
                  hostname=platform.node(), cpu_model=cpu, profiled=a.profile)
    if not a.profile:
        val = make(tokenizer, 16, 2048, 'val', device='cpu', tokenizer_threads=threads)
        for name, count, expected in (
            ('dev', 4, 'bd67b90b4386ba6428992864712f56eb10527678ed8b3664ed5f63c1643458a4'),
            ('test', 64, 'e7d24b6a89155ebc7123afe68bd2c58aeaf159b5087dbdf28e090ce09f293d09')):
            h = hashlib.sha256()
            for _ in range(count):
                x, y, _ = next(val)
                h.update(torch.cat((x, y[:, -1:]), dim=1).numpy().astype(np.uint16).tobytes())
            assert h.hexdigest() == expected, (name, h.hexdigest())
            result[name + '_sha256'] = h.hexdigest()
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
