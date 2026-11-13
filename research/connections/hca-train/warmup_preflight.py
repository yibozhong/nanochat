"""One authorized from-zero warmup ablation; GPU gate shares formal allocation."""
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET

from policy import STEPS, TOKENS
from submit import write
from optimized_preflight import validate_probe, validate_cache, COMPRESSION_RANGE

PLAN = 'hca-warmup-b4-10b-20261113'
WARMUP = 2000
DIAGNOSTIC_STEPS = 3000


def validate_source(s):
    assert s['plan'] == PLAN and s['formal_training'] is True and s['attention_mode'] == 'dynamic_hca'
    assert s['steps'] == STEPS and s['training_tokens'] == TOKENS
    assert s['seed'] == 42 and s['microbatch'] == 16 and s['batch_tokens'] == 32768
    assert s['initialization'] == 'seed42_from_zero_no_validation_weights' and s['hca_backend'] == 'triton'
    assert s['hca'] == dict(target_ratio=4, window=256, ratio_loss_weight=1, temperature=1,
        noise=False, dynamic_gradient='confidence', router_init='uncompressed', ratio_warmup_steps=WARMUP)


def validate_ready(v, s):
    validate_source(s)
    assert v['complete'] is True and v['formal_training'] is False and v['source'] == s
    assert v['microbatch'] == 16 and v['capability'] == [9, 0]
    assert v['tests']['tests'] >= 39 and all(v['tests'][k] == 0 for k in ('failures','errors','skipped'))
    assert v['initial_hard_keep_rate'] == 1 and v['initial_ratio_loss_weight'] == 0
    assert v['diagnostic_steps'] == DIAGNOSTIC_STEPS and v['warmup_weights_exact'] is True
    assert v['replay_initial_steps'] == [1000, 3000] and v['cache_checks_passed'] == 2
    for name in ('train_last500_summary_compression','dev_summary_compression'):
        assert COMPRESSION_RANGE[0] <= v[name] <= COMPRESSION_RANGE[1]
    assert math.isfinite(v['steady_step_seconds']) and v['steady_step_seconds'] > 0


def validate_trajectory(first, second, rows, s):
    for v, start, stop in ((first, 0, 1000), (second, 1000, DIAGNOSTIC_STEPS)):
        assert v['complete'] is True and v['formal_training'] is False and v['status'] == 'finite_limit'
        assert v['source'] == s and v['steps'] == stop and v.get('resumed_step', 0) == start
        a = v['args']; c = v['config']
        assert a['seed'] == 42 and a['block_size'] == 4 and a['gradient'] == 'confidence'
        assert c['hca_router_init'] == 'uncompressed' and c['hca_ratio_warmup_steps'] == WARMUP
        assert c['hca_ratio_loss_weight'] == 1 and c['hca_noise'] is False
    assert first['args']['resume'] is None and first['initial_validation']['step'] == 0
    assert first['initial_validation']['hard_keep_rate'] == 1
    assert second['args']['resume'] == '../diagnostic-state/latest.pt'
    assert len(rows) == DIAGNOSTIC_STEPS
    for step, row in enumerate(rows, 1):
        assert row['step'] == step
        assert all(math.isfinite(row[k]) for k in ('loss','ce','objective','hard_keep_rate','seconds','ratio_loss_weight'))
        assert 0 < row['hard_keep_rate'] <= 1 and row['seconds'] > 0
        # Metrics are FP32 scalars; permit only conversion rounding, not a
        # shift in update indexing or a reset after the diagnostic checkpoint.
        assert abs(row['ratio_loss_weight'] - min((step-1)/WARMUP, 1.)) <= 3e-8
    assert rows[0]['hard_keep_rate'] == 1 and rows[0]['ratio_loss_weight'] == 0
    assert rows[0]['loss'] == rows[0]['ce']
    dev = second['validation'][-1]
    assert dev['step'] == DIAGNOSTIC_STEPS and dev['tokens'] == 131072 and math.isfinite(dev['ce'])
    train_ratio = 1/statistics.mean(r['hard_keep_rate'] for r in rows[-500:])
    dev_ratio = 1/dev['hard_keep_rate']
    for ratio in (train_ratio, dev_ratio):
        assert COMPRESSION_RANGE[0] <= ratio <= COMPRESSION_RANGE[1], ('Post-warmup B4 rate gate failed', ratio)
    return dict(initial_hard_keep_rate=1., initial_ratio_loss_weight=0., warmup_weights_exact=True,
                diagnostic_steps=DIAGNOSTIC_STEPS, train_last500_summary_compression=train_ratio,
                dev_summary_compression=dev_ratio, diagnostic_dev_ce=dev['ce'])


def main():
    assert os.environ['WANDB_MODE'] == 'disabled'
    s = json.loads(Path('source.json').read_text()); validate_source(s)
    import torch
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    assert any(n in torch.cuda.get_device_name() for n in ('H100','H200'))
    assert torch.cuda.get_device_properties(0).total_memory >= 78000*1024**2
    out = Path('results/hca-startup'); out.mkdir(parents=True, exist_ok=True)
    marker = out/'passed.json'
    if marker.exists():
        ready = json.loads(marker.read_text()); validate_ready(ready, s)
        print('OPTIMIZED_B4_STARTUP_PASSED '+json.dumps(ready, allow_nan=False), flush=True)
        return
    assert not (out/'started.json').exists(), 'Inspect interrupted validation before retrying'
    write(out/'started.json', dict(source=s))
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_hca_warmup.py',
        'tests/test_hca.py', 'tests/test_hca_confidence.py', 'tests/test_packed_training_checks.py',
        '--junitxml='+str(out/'warmup.xml')], check=True)
    suites = ET.parse(out/'warmup.xml').getroot().findall('testsuite')
    counts = {k:sum(int(v.get(k, 0)) for v in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['tests'] >= 39 and all(counts[k] == 0 for k in ('failures','errors','skipped'))
    def run(script, *args):
        subprocess.run([sys.executable, 'research/connections/'+script, *args], check=True)
    shared = ['--mode','dynamic_hca','--block-size','4','--ratio-weight','1',
              '--router-init','uncompressed','--ratio-warmup-steps',str(WARMUP)]
    opt = Path('results/optimization')
    phases = []
    for stop in (1000, DIAGNOSTIC_STEPS):
        label = 'warmup-stability-'+str(stop)
        extra = ['--record-initial'] if stop == 1000 else ['--resume','../diagnostic-state/latest.pt']
        run('hca-opt/stability.py', *shared, '--gradient','confidence','--steps',str(stop),
            '--seed','42','--label',label,'--save-state',*extra)
        phases.append(json.loads((opt/(label+'.json')).read_text()))
        probe = out/('warmup-replay-'+str(stop)+'.json')
        run('hca-train/replay_check.py', *shared, '--dynamic-gradient','confidence',
            '--microbatch','16','--backend','triton','--resume','../diagnostic-state/latest.pt','--output',str(probe))
        v = json.loads(probe.read_text()); validate_probe(v, s, initial_step=stop)
        assert v['initial_step'] == stop and v['config']['hca_ratio_warmup_steps'] == WARMUP
        assert v['config']['hca_router_init'] == 'uncompressed'
        assert abs(v['metrics']['ratio_loss_weight'] - min((stop+2)/WARMUP, 1.)) <= 3e-8
    rows = []
    for stop in (1000, DIAGNOSTIC_STEPS):
        rows += [json.loads(line) for line in (opt/f'warmup-stability-{stop}-steps.jsonl').read_text().splitlines()]
    trajectory = validate_trajectory(*phases, rows, s)
    for row, splits in ((0,'137,1,511,7,1000,392'), (1,'1,63,64,7,512,1,1400')):
        label = 'warmup-cache-'+str(row)
        run('hca-opt/trained_cache.py','--row',str(row),'--splits',splits,'--label',label)
        validate_cache(json.loads((opt/(label+'.json')).read_text()), s, diagnostic_steps=DIAGNOSTIC_STEPS)
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid); timings = {}
    for mode in ('full','fixed_hca','dynamic_hca'):
        label = 'warmup-speed-'+mode
        extra = ['--resume','../diagnostic-state/latest.pt'] if mode == 'dynamic_hca' else []
        run('hca-opt/benchmark.py','--mode',mode,'--block-size','4','--steps','40','--label',label,*extra)
        v = json.loads((opt/(label+'.json')).read_text())
        assert v['formal_training'] is False and v['source'] == s and v['gpu_uuid'] == gpu_uuid
        assert len(v['seconds']) == len(v['losses']) == 40
        assert all(math.isfinite(x) and x > 0 for x in v['seconds']) and all(math.isfinite(x) for x in v['losses'])
        timings[mode] = v['median_seconds']
    assert not any(Path('../checkpoints').glob('*.pt'))
    Path('../diagnostic-state/latest.pt').unlink()
    ready = dict(complete=True, formal_training=False, source=s, microbatch=16, capability=[9,0],
        tests=counts, gpu=torch.cuda.get_device_name(), gpu_uuid=gpu_uuid,
        replay_initial_steps=[1000,3000], cache_checks_passed=2, **trajectory,
        same_card_seconds=timings, steady_step_seconds=timings['dynamic_hca'],
        estimated_train_only_hours=STEPS*timings['dynamic_hca']/3600)
    validate_ready(ready,s);write(marker,ready)
    print('OPTIMIZED_B4_STARTUP_PASSED '+json.dumps(ready,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
