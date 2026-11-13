"""Frozen two-run authorization and strict, stdlib-only report validation."""
import math

MODES = ('fixed_hca', 'dynamic_hca')
STEPS = 305176
BATCH_TOKENS = 32768
TOKENS = STEPS * BATCH_TOKENS
CORE_SOURCE = '559227a58d354f553d23b6514ff8956609b2aa7b'
CORE_JOB = '8483245.0'
CAPACITY_BYTES = int(.9 * 78000 * 1024**2)
TEST_COUNTS = dict(tests=18, failures=0, errors=0, skipped=0)


def validate_probe(v, mode, micro, commit):
    assert v['complete'] is True and v['formal_training'] is False
    assert v['status'] == 'passed' and v['mode'] == mode and v['microbatch'] == micro
    assert v['commit'] == commit and v['steps'] == 3 and v['replayed_step'] == 2
    assert v['replay_model_equal'] is True and v['replay_optimizer_equal'] is True
    assert v['all_gradients_finite'] is True
    assert v['peak_reserved_bytes'] <= CAPACITY_BYTES
    assert v['parameters'] == (60817408 if mode == 'fixed_hca' else 60821512)
    c = v['config']
    assert c['attention_mode'] == mode and c['variant'] == 'baseline' and c['architecture'] == 'vanilla'
    assert (c['depth'], c['width'], c['heads'], c['sequence_length'], c['vocab_size'], c['streams']) == (8, 512, 4, 2048, 32768, 1)
    assert c['hca_block_size'] == 16 and c['hca_window'] == 256 and c['hca_ratio_loss_weight'] == 1
    assert c['hca_temperature'] == 1 and c['hca_noise'] is True and not c['activation_checkpoint']
    assert len(v['losses']) == 3 and all(math.isfinite(x) for x in v['losses'])
    assert math.isfinite(v['steady_step_seconds']) and v['steady_step_seconds'] > 0
    assert v['pool_gradient_nonzero'] is True
    if mode == 'dynamic_hca':
        assert v['router_gradient_nonzero'] is True


def validate_shared(v, commit):
    assert v['complete'] is True and v['formal_training'] is False
    assert v['commit'] == commit and v['capability'] == [9, 0]
    assert v['core_tests'] == TEST_COUNTS and v['core_source'] == CORE_SOURCE
    assert v['comparison_complete'] is True
    assert set(v['modes']) == set(MODES)
    micro = v['microbatch']
    assert micro in (16, 8, 4, 2, 1)
    for mode in MODES:
        validate_probe(v['modes'][mode], mode, micro, commit)
    assert len({v['modes'][mode]['gpu_uuid'] for mode in MODES}) == 1
    return micro


def first_formal_step(lines, mode, run_id):
    import json
    marker = 'FORMAL_TRAINING_BEGIN '+mode+' '+run_id
    formal = False
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN '):
            formal = line.strip() == marker
        if not formal or not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row.get('step'), int) and row['step'] >= 1 and isinstance(row.get('loss'), (int, float)) and math.isfinite(row['loss']):
            return row
    return None
