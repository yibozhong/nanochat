"""B4 pair authorization: reuse Fixed evidence; validate confidence B4 on worker.

Importing this module on the AP is stdlib-only. All tensor work is in child
processes on the allocated GPU, before the separate from-zero formal process.
"""
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from policy import CAPACITY_BYTES, STEPS, TOKENS
from submit import write

PLAN = 'hca-optimized-b4-10b-20261113'
VERIFIED = '21ef47d9bc6db859fcf4386205f2ed35cb5c4969'
MODES = ('fixed_hca', 'dynamic_hca')
DIAGNOSTIC_STEPS = 2000
# A soft mean constraint, not a quota on each sequence/chunk. Freeze before
# seeing B4 results: train last500 and final held-out mean must be within 25%.
COMPRESSION_RANGE = (3.2, 5.0)


def validate_source(s):
    assert s['plan'] == PLAN and s['formal_training'] is True
    assert s['attention_mode'] in MODES
    assert s['steps'] == STEPS and s['training_tokens'] == TOKENS
    assert s['seed'] == 42 and s['microbatch'] == 16 and s['batch_tokens'] == 32768
    assert s['initialization'] == 'seed42_from_zero_no_validation_weights'
    assert s['hca_backend'] == 'triton' and s['verified_runtime_commit'] == VERIFIED
    expected_gradient = 'confidence' if s['attention_mode'] == 'dynamic_hca' else 'dmc'
    assert s['hca'] == dict(target_ratio=4, window=256, ratio_loss_weight=1,
                           temperature=1, noise=False, dynamic_gradient=expected_gradient)


def validate_probe(v, s):
    assert v['complete'] is True and v['status'] == 'passed' and v['formal_training'] is False
    assert v['mode'] == s['attention_mode'] and v['commit'] == s['commit'] and v['microbatch'] == 16
    assert v['steps'] == 3 and v['replayed_step'] == 2 and v['cce_replay_microbatches'] == 1
    assert v['restore_contract'] == 'exact_state_rng_data_and_fixed_cce_gradient_replay'
    for key in ('loaded_model_equal', 'loaded_optimizer_equal', 'loaded_rng_equal',
                'loaded_data_row_equal', 'actual_cce_inputs_equal', 'controlled_backward_loss_equal',
                'controlled_backward_model_equal', 'controlled_backward_optimizer_equal',
                'controlled_backward_rng_equal', 'all_gradients_finite',
                'pool_gradient_nonzero', 'router_gradient_nonzero'):
        assert v[key] is True, key
    assert math.isfinite(v['actual_cce_loss_abs_difference']) and v['actual_cce_loss_abs_difference'] >= 0
    assert v['peak_reserved_bytes'] <= CAPACITY_BYTES and v['parameters'] == 60821512
    c = v['config']
    assert (c['architecture'], c['variant'], c['streams']) == ('vanilla', 'baseline', 1)
    assert (c['depth'], c['width'], c['heads'], c['sequence_length'], c['vocab_size']) == (8, 512, 4, 2048, 32768)
    assert c['attention_mode'] == 'dynamic_hca' and c['hca_backend'] == 'triton'
    assert c['hca_dynamic_gradient'] == 'confidence' and c['hca_noise'] is False
    assert c['hca_block_size'] == 4 and c['hca_window'] == 256
    assert c['hca_ratio_loss_weight'] == 1 and c['hca_temperature'] == 1 and not c['activation_checkpoint']
    assert len(v['losses']) == 3 and all(math.isfinite(x) for x in v['losses'])


def validate_stability(v, rows, s):
    assert v['complete'] is True and v['formal_training'] is False and v['status'] == 'finite_limit'
    assert v['source'] == s and v['steps'] == DIAGNOSTIC_STEPS
    a = v['args']
    assert (a['mode'], a['gradient'], a['block_size'], a['ratio_weight'], a['seed']) == ('dynamic_hca', 'confidence', 4, 1, 42)
    assert a['resume'] is None and a['save_state'] is True
    assert len(rows) == DIAGNOSTIC_STEPS and [r['step'] for r in rows] == list(range(1, DIAGNOSTIC_STEPS+1))
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ('loss', 'ce', 'objective', 'hard_keep_rate', 'soft_keep_rate', 'seconds'))
        assert row['seconds'] > 0 and 0 < row['hard_keep_rate'] <= 1
    train_ratio = 1 / statistics.mean(row['hard_keep_rate'] for row in rows[-500:])
    dev = v['validation'][-1]
    assert dev['step'] == DIAGNOSTIC_STEPS and dev['tokens'] == 131072 and math.isfinite(dev['ce'])
    dev_ratio = 1 / dev['hard_keep_rate']
    for ratio in (train_ratio, dev_ratio):
        assert COMPRESSION_RANGE[0] <= ratio <= COMPRESSION_RANGE[1], ('B4 mean rate outside startup gate', ratio)
    return dict(train_last500_summary_compression=train_ratio, dev_summary_compression=dev_ratio,
                diagnostic_dev_ce=dev['ce'], diagnostic_steps=DIAGNOSTIC_STEPS)


def validate_cache(v, s):
    assert v['complete'] is True and v['status'] == 'passed' and v['formal_training'] is False
    assert v['source'] == s and v['checkpoint_source'] == s and v['step'] == DIAGNOSTIC_STEPS
    assert v['transplanted_weights'] is False
    for key in ('causal_prefix_exact', 'cache_replay_exact', 'cache_replay_stats_equal',
                'cache_replay_tensors_exact', 'hidden_finite'):
        assert v[key] is True
    assert 0 <= v['hidden_relative_l2'] <= .02 and 0 <= v['hidden_normalized_peak'] <= .02
    assert len(v['layers']) == 8
    assert all(r['boundary_mismatches'] == 0 and r['open_chunk_tokens'] == r['full_open_chunk_tokens'] for r in v['layers'])


def validate_ready(v, s):
    validate_source(s)
    assert v['complete'] is True and v['formal_training'] is False and v['source'] == s
    assert v['microbatch'] == 16 and v['capability'] == [9, 0] and v['verified_runtime_commit'] == VERIFIED
    assert math.isfinite(v['steady_step_seconds']) and v['steady_step_seconds'] > 0
    if s['attention_mode'] == 'dynamic_hca':
        assert v['diagnostic_steps'] == DIAGNOSTIC_STEPS and v['replay_passed'] is True and v['cache_checks_passed'] == 2
        for key in ('train_last500_summary_compression', 'dev_summary_compression'):
            assert COMPRESSION_RANGE[0] <= v[key] <= COMPRESSION_RANGE[1]
    else:
        assert v['reused_fixed_b4_evidence'] is True


def main():
    assert os.environ['WANDB_MODE'] == 'disabled'
    s = json.loads(Path('source.json').read_text()); validate_source(s)
    import torch
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0)
    assert any(n in torch.cuda.get_device_name() for n in ('H100', 'H200'))
    assert torch.cuda.get_device_properties(0).total_memory >= 78000*1024**2
    marker = Path('results/hca-startup/passed.json'); marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        ready = json.loads(marker.read_text()); validate_ready(ready, s)
        print('OPTIMIZED_B4_STARTUP_PASSED '+json.dumps(ready, allow_nan=False), flush=True)
        return
    # A previous interrupted/failed diagnostic is not automatically rerun.
    intent = marker.parent/'started.json'
    assert not intent.exists(), 'Interrupted startup; inspect debug artifacts before retrying'
    write(intent, dict(source=s))
    evidence = json.loads(Path('research/connections/hca-opt/results.json').read_text())
    assert evidence['verification_commit'] == VERIFIED and evidence['formal_training'] is False
    assert evidence['tests']['runtime_and_reference_gpu'] == 111
    assert evidence['tests']['failed'] == evidence['tests']['skipped'] == 0
    ready = dict(complete=True, formal_training=False, source=s, microbatch=16, capability=[9, 0],
                 gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                 verified_runtime_commit=VERIFIED)
    if s['attention_mode'] == 'fixed_hca':
        assert evidence['stability']['fixed-b4-stability']['steps'] == 2000
        assert evidence['recovery']['fixed-b4-replay']['exact_checkpoint_inputs_and_controlled_update'] is True
        ready.update(reused_fixed_b4_evidence=True,
                     steady_step_seconds=evidence['performance']['final-fixed-b4']['median_ms']/1000,
                     timing_scope='Prior same-runtime H100 measurement, not current allocation timing')
    else:
        def run(script, *args):
            subprocess.run([sys.executable, 'research/connections/'+script, *args], check=True)
        probe = marker.parent/'confidence-b4-replay.json'
        run('hca-train/replay_check.py', '--mode', 'dynamic_hca', '--dynamic-gradient', 'confidence',
            '--ratio-weight', '1', '--microbatch', '16', '--backend', 'triton', '--block-size', '4', '--output', str(probe))
        validate_probe(json.loads(probe.read_text()), s)
        label = 'confidence-b4-startup'
        run('hca-opt/stability.py', '--mode', 'dynamic_hca', '--gradient', 'confidence', '--block-size', '4',
            '--ratio-weight', '1', '--steps', str(DIAGNOSTIC_STEPS), '--seed', '42', '--label', label, '--save-state')
        out = Path('results/optimization')
        stability = json.loads((out/(label+'.json')).read_text())
        rows = [json.loads(line) for line in (out/(label+'-steps.jsonl')).read_text().splitlines()]
        ready.update(validate_stability(stability, rows, s))
        for row, splits in ((0, '137,1,511,7,1000,392'), (1, '1,63,64,7,512,1,1400')):
            name = 'confidence-b4-cache-'+str(row)
            run('hca-opt/trained_cache.py', '--row', str(row), '--splits', splits, '--label', name)
            validate_cache(json.loads((out/(name+'.json')).read_text()), s)
        # Small same-card comparison; no diagnostic weights written back.
        timings = {}
        for mode in ('full', *MODES):
            name = 'b4-startup-speed-'+mode
            args = ['--mode', mode, '--block-size', '4', '--label', name, '--steps', '40']
            if mode == 'dynamic_hca':
                args += ['--resume', '../diagnostic-state/latest.pt']
            run('hca-opt/benchmark.py', *args)
            result = json.loads((out/(name+'.json')).read_text())
            assert result['formal_training'] is False and result['source'] == s
            assert result['gpu_uuid'] == ready['gpu_uuid']
            assert len(result['seconds']) == len(result['losses']) == 40
            assert all(math.isfinite(x) and x > 0 for x in result['seconds'])
            assert all(math.isfinite(x) for x in result['losses'])
            timings[mode] = result['median_seconds']
        ready.update(replay_passed=True, cache_checks_passed=2, same_card_seconds=timings,
                     steady_step_seconds=timings['dynamic_hca'], timing_scope='This GPU, 5 warmup +40 updates')
        # Only the disposable diagnostic file is removed; formal checkpoints are
        # a different directory and the new formal process reseeds everything.
        Path('../diagnostic-state/latest.pt').unlink()
    assert not any(Path('../checkpoints').glob('*.pt')), 'Fresh startup must not contain formal/diagnostic weights'
    ready['estimated_train_only_hours'] = STEPS*ready['steady_step_seconds']/3600
    validate_ready(ready, s); write(marker, ready)
    print('OPTIMIZED_B4_STARTUP_PASSED '+json.dumps(ready, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
