"""Pure validation of the evidence required before the one authorized10B run."""
import math


def validate_gate(report, state):
    assert report['complete'] and report['formal_training'] is False
    assert report['source'] == state['source']
    assert report['checks'] == dict(tests=26, failures=0, errors=0, skipped=0)
    assert report['microbatch'] in (16, 8, 4)
    profiles = report['profiles']
    assert set(profiles) == {'baseline', 'state_tokens', 'state_tokens_torch'}
    for name, p in profiles.items():
        assert p['complete'] and p['causality_passed'] and p['causal_prefix_max_abs'] == 0
        assert p['optimizer_updates'] == 60 and len(p['losses']) == 60
        assert p['training_tokens'] == 60*32768
        assert all(math.isfinite(x) for x in p['losses'])
        assert p['batch_tokens'] == 32768 and p['sequence_length'] == 2048 and p['seed'] == 42
        assert p['microbatch'] == report['microbatch']
        assert p['source'] == state['source']
        assert p['gpu_uuid'] == profiles['baseline']['gpu_uuid']
        assert p['input_sha256'] == profiles['baseline']['input_sha256']
        assert p['logging_backend'] == 'local'
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds'] > 0
        assert 0 < p['peak_reserved_bytes'] < .9*p['device_total_bytes']
        if name != 'baseline':
            assert p['model']['variant'] == 'state_tokens'
            assert p['model']['state_write_residual']
            assert p['model']['state_attention_backend'] == ('triton' if name == 'state_tokens' else 'auto')
    r = report['restart']
    assert r['complete'] and r['steps'] == 50 and r['training_tokens'] == 50*32768
    assert r['source'] == state['source'] and r['variant'] == 'state_tokens'
    assert all(math.isfinite(r[k]) for k in ('initial_dev_ce','final_dev_ce','best_dev_ce'))
    assert r['final_dev_ce'] < r['initial_dev_ce']
    assert r['peak_reserved_bytes'] > 0
    return report
