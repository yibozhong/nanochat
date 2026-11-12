"""Bounded HCA checks/cache accounting; no formal run or W&B, no checkpoints out."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    out = Path('results/hca'); out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-v', 'tests/test_hca.py',
                           '--junitxml=' + str(out/'tests.xml')])
    suites = ET.parse(out/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    (out/'test-counts.json').write_text(json.dumps(counts, indent=2)+'\n')
    if proc.returncode or counts['failures'] or counts['errors'] or counts['skipped']:
        raise RuntimeError(f'HCA validation failed: {counts}')

    import torch
    from nanochat.connection_model import ConnectionConfig, build_connection_model
    from nanochat.packed_training import PackedData, seed_all
    from nanochat.connection_experiment import configure_attention
    configure_attention('fa2')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(complete=False, formal_training=False, tests=counts,
                  source=json.loads(Path('source.json').read_text()),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  comparisons={}, limitations='Unoptimized reference; bounded checks, not learned quality evidence. No formal training.')
    data = PackedData('../data')
    x, y = data.batch('train', 0, 1)
    # Full 2K and production width/head geometry; two layers limit reference
    # autograd overhead. All three modes see the same FineWeb tokens/seed.
    for mode in ('full', 'fixed_hca', 'dynamic_hca'):
        seed_all(42)
        c = ConnectionConfig(variant='baseline', streams=1, depth=2, width=512, heads=4,
                             sequence_length=2048, attention_mode=mode, hca_noise=False)
        m = build_connection_model(c).prepare(dtype=torch.bfloat16).eval()
        # Nonzero output branches exercise the attention beyond initialization.
        with torch.no_grad():
            for b in m.blocks:
                b.attn.c_proj.weight.normal_(std=.01)
                b.mlp.c_proj.weight.normal_(std=.005)
                if mode != 'full':
                    b.attn.pool_score.weight.normal_(std=.01)
                    if mode == 'dynamic_hca':
                        b.attn.boundary.weight.normal_(std=.05)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        before = time.perf_counter()
        with torch.no_grad():
            logits = m(x)
            torch.cuda.synchronize()
            dense_seconds = time.perf_counter() - before
            cache = m.make_kv_cache()
            # A split inside an incomplete B=16 chunk, then close the frame,
            # then decode several tokens and finish the same 2K sequence.
            pieces = []; start = 0
            torch.cuda.synchronize(); before = time.perf_counter()
            for n in (1003, 21, 1, 1, 1, 1021):
                pieces.append(m(x[:, start:start+n], kv_cache=cache)); start += n
            torch.cuda.synchronize(); cache_seconds = time.perf_counter() - before
            cached = torch.cat(pieces, 1)
            torch.testing.assert_close(cached, logits, rtol=.01, atol=.01)
            ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
        row = dict(untrained_dev_quality=False, diagnostic_train_ce=ce,
                   full_forward_seconds=dense_seconds, split_cached_seconds=cache_seconds,
                   max_logit_difference=(cached-logits).abs().max().item(),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   parameters=sum(p.numel() for p in m.parameters()))
        full_bytes = 2*c.depth*2048*c.width*2
        if mode == 'full':
            row['cache'] = dict(storage_bytes=cache.k_cache.untyped_storage().nbytes()+cache.v_cache.untyped_storage().nbytes()
                               + cache.cache_seqlens.untyped_storage().nbytes(), tokens=cache.get_pos())
        else:
            row['cache'] = cache.statistics()
            assert all(v == [128] for v in row['cache']['completed_chunks'])
            assert row['cache']['storage_bytes'] == row['cache']['tensor_bytes'], 'Hidden backing storage retained'
            assert row['cache']['storage_bytes'] < full_bytes
            row['chunk_lengths'] = [[end-start for start,end in zip([-1]+layer[0].ends[:-1], layer[0].ends)]
                                    for layer in cache.layers]
            if mode == 'dynamic_hca':
                assert any(len(set(lengths)) > 1 for lengths in row['chunk_lengths']), 'Content router not exercised'
            row['compressed_history_ratio'] = 16
        row['effective_cache_ratio'] = full_bytes / row['cache']['storage_bytes']
        report['comparisons'][mode] = row
        (out/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(mode=mode, **row), allow_nan=False), flush=True)
        del m, cache, logits, cached, pieces
        torch.cuda.empty_cache()
    report['complete'] = True
    (out/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('HCA_VALIDATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
