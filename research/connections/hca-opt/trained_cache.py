"""Production 2K trained-router cache audit; diagnostic weights only, no W&B."""
import argparse
import json
import os
from pathlib import Path
import torch
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.connection_experiment import configure_attention
from nanochat.packed_training import PackedData

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default='../diagnostic-state/latest.pt')
p.add_argument('--label', default='confidence-trained-cache')
a = p.parse_args()
assert os.environ['WANDB_MODE'] == 'disabled'
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
configure_attention('fa2')
state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
assert state['formal_training'] is False
c = ConnectionConfig(**state['config'])
model = build_connection_model(c).prepare(compile_blocks=False).eval()
model.load_state_dict(state['model'])
report = dict(complete=False, formal_training=False, step=state['step'],
              checkpoint_source=state['source'],
              source=json.loads(Path('source.json').read_text()),
              gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
del state
data = PackedData('../data')
ids, _ = data.batch('dev', 0, 1)
out = Path('results/optimization') / (a.label + '.json')
def write():
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
write()
routes = {}
for layer, block in enumerate(model.blocks):
    original = block.attn.accelerated
    def capture(*args, _original=original, _layer=layer, **kwargs):
        result = _original(*args, **kwargs)
        routes[_layer] = (result[1].detach().clone(), args[5].detach().clone())
        return result
    block.attn.accelerated = capture

with torch.no_grad():
    expected = model.hidden(ids)
    full_routes = dict(routes)
    altered = ids.clone(); altered[:, 1000:] = (altered[:, 1000:] + 1) % c.vocab_size
    changed = model.hidden(altered)
    report['causal_prefix_exact'] = torch.equal(changed[:, :1000], expected[:, :1000])
    cache = model.make_kv_cache(1)
    parts = []; start = 0
    splits = (137, 1, 511, 7, 1000, 392)
    assert sum(splits) == ids.shape[1]
    for n in splits[:-1]:
        parts.append(model.hidden(ids[:, start:start+n], kv_cache=cache)); start += n
    saved = cache.state_dict()
    restored = model.make_kv_cache(1); restored.load_state_dict(saved)
    suffix = model.hidden(ids[:, start:], kv_cache=cache)
    replay = model.hidden(ids[:, start:], kv_cache=restored)
    report['cache_replay_exact'] = torch.equal(suffix, replay)
    report['cache_replay_stats_equal'] = cache.statistics() == restored.statistics()
    report['cache_replay_tensors_exact'] = all(
        (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
        for la, lb in zip(cache.layers, restored.layers)
        for sa, sb in zip(la, lb)
        for x, y in zip((*sa.pool, *sa.pool_carry, sa.pool_tile_k, sa.pool_tile_v,
                        sa.pool_tile_score, sa.pool_tile_starts, sa.local_k, sa.local_v,
                        sa.summary_k, sa.summary_v, sa.open_k, sa.open_v),
                       (*sb.pool, *sb.pool_carry, sb.pool_tile_k, sb.pool_tile_v,
                        sb.pool_tile_score, sb.pool_tile_starts, sb.local_k, sb.local_v,
                        sb.summary_k, sb.summary_v, sb.open_k, sb.open_v)))
    parts.append(suffix)
    actual = torch.cat(parts, 1)
    delta = actual.float() - expected.float()
    report['hidden_relative_l2'] = (delta.norm()/expected.float().norm()).item()
    report['hidden_normalized_peak'] = (delta.abs().max()/expected.float().abs().max().clamp_min(1.)).item()
    report['hidden_finite'] = bool(torch.isfinite(actual).all())
    report['splits'] = splits
    report['cache'] = cache.statistics()
    full_bytes = c.depth * 2 * ids.numel() * c.width * model.wte.weight.element_size()
    report['baseline_kv_bytes'] = full_bytes
    report['cache_storage_compression'] = full_bytes / report['cache']['storage_bytes']
    report['layers'] = []
    for layer in range(c.depth):
        starts, content = full_routes[layer]
        positions = starts[0].nonzero().flatten().tolist()
        expected_ends = [i-1 for i in positions[1:]]
        stored = cache.layers[layer][0]
        mismatch = sorted(set(expected_ends).symmetric_difference(stored.ends))
        lengths = [j-i for i,j in zip(positions, positions[1:]+[ids.shape[1]])]
        report['layers'].append(dict(layer=layer, full_chunks=len(positions),
            cached_chunks=len(stored.ends)+1, boundary_mismatches=len(mismatch),
            first_mismatches=mismatch[:20], smallest_full_logit_margin=content[0,1:].abs().min().item(),
            mean_chunk_length=sum(lengths)/len(lengths), min_chunk_length=min(lengths),
            max_chunk_length=max(lengths), open_chunk_tokens=stored.age,
            full_open_chunk_tokens=lengths[-1]))
    write()
    assert report['causal_prefix_exact'] and report['cache_replay_exact']
    assert report['cache_replay_stats_equal'] and report['cache_replay_tensors_exact'] and report['hidden_finite']
    assert report['hidden_relative_l2'] <= .02 and report['hidden_normalized_peak'] <= .02
    assert all(row['boundary_mismatches'] == 0 for row in report['layers'])
    assert all(row['open_chunk_tokens'] == row['full_open_chunk_tokens'] for row in report['layers'])
report.update(complete=True, status='passed')
write()
print(json.dumps(report), flush=True)
