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


def validate_probe(v, mode, micro, commit, block_size=16):
    assert v['complete'] is True and v['formal_training'] is False
    assert v['status'] == 'passed' and v['mode'] == mode and v['microbatch'] == micro
    assert v['commit'] == commit and v['steps'] == 3 and v['replayed_step'] == 2
    # Same-input CCE backward nondeterminism was isolated in 9499a07 and
    # controlled full-update replay passed in 01ddae1. Require exact checkpoint
    # load, RNG/data, and full backward/update with fixed CCE gradients. New
    # probes additionally compare actual CCE inputs exactly: its forward also
    # has unordered logaddexp reductions. Old frozen reports only recorded loss.
    assert v['restore_contract'] == 'exact_state_rng_data_and_fixed_cce_gradient_replay'
    for key in ('loaded_model_equal', 'loaded_optimizer_equal', 'loaded_rng_equal',
                'loaded_data_row_equal',
                'controlled_backward_loss_equal', 'controlled_backward_model_equal',
                'controlled_backward_optimizer_equal', 'controlled_backward_rng_equal'):
        assert v[key] is True, key
    if 'actual_cce_inputs_equal' in v:
        assert v['actual_cce_inputs_equal'] is True
        assert math.isfinite(v['actual_cce_loss_abs_difference']) and v['actual_cce_loss_abs_difference'] >= 0
    else:
        assert v['actual_cce_replay_loss_equal'] is True
    assert v['cce_replay_microbatches'] == 16//micro
    assert v['all_gradients_finite'] is True
    assert v['peak_reserved_bytes'] <= CAPACITY_BYTES
    assert v['parameters'] == (60817408 if mode == 'fixed_hca' else 60821512)
    c = v['config']
    assert c['attention_mode'] == mode and c['variant'] == 'baseline' and c['architecture'] == 'vanilla'
    assert (c['depth'], c['width'], c['heads'], c['sequence_length'], c['vocab_size'], c['streams']) == (8, 512, 4, 2048, 32768, 1)
    assert c['hca_block_size'] == block_size and c['hca_window'] == 256 and c['hca_ratio_loss_weight'] == 1
    assert c['hca_temperature'] == 1 and c['hca_noise'] is True and not c['activation_checkpoint']
    assert c['hca_backend'] == 'triton'
    # This frozen authorization is for the original estimator. The separate
    # confidence diagnostics do not authorize replacing its formal run.
    assert c.get('hca_dynamic_gradient', 'dmc') == 'dmc'
    assert len(v['losses']) == 3 and all(math.isfinite(x) for x in v['losses'])
    assert math.isfinite(v['steady_step_seconds']) and v['steady_step_seconds'] > 0
    assert v['pool_gradient_nonzero'] is True
    if mode == 'dynamic_hca':
        assert v['router_gradient_nonzero'] is True


def validate_shared(v, commit):
    assert v['complete'] is True and v['formal_training'] is False
    assert v['commit'] == commit and v['capability'] == [9, 0]
    assert v['core_tests'] == TEST_COUNTS and v['core_source'] == commit
    assert v['kernel_tests'] == dict(tests=24, failures=0, errors=0, skipped=0)
    assert v['hca_backend'] == 'triton'
    assert v['speed']['complete'] and v['speed']['formal_training'] is False
    assert v['speed']['source']['commit'] == commit
    assert v['comparison_complete'] is True
    assert set(v['modes']) == set(MODES)
    micro = v['microbatch']
    assert micro in (16, 8, 4, 2, 1)
    for mode in MODES:
        validate_probe(v['modes'][mode], mode, micro, commit)
    assert len({v['modes'][mode]['gpu_uuid'] for mode in MODES}) == 1
    assert v['speed']['gpu_uuid'] == v['modes']['fixed_hca']['gpu_uuid']
    assert v['speed']['microbatch'] == micro
    for mode in MODES:
        assert 0 < v['speed']['modes'][mode]['relative_to_baseline'] < 10
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
