"""Stdlib-only policy for the one authorized FFN-readout 10B run."""
import math

PLAN = 'ffn-readout-r64-10b-20261113'
STEPS = 305176
BATCH_TOKENS = 32768
TOKENS = STEPS * BATCH_TOKENS
VERIFIED = '2c5d9d0205417f62c3d652f067c9fd67e9c329fa'
DATA_URI = 'osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-7685535.tar'
DATA_REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'


def validate_source(source, evidence):
    assert source['plan'] == PLAN and source['formal_training'] is True
    assert source['steps'] == STEPS and source['training_tokens'] == TOKENS
    assert source['batch_tokens'] == BATCH_TOKENS and source['microbatch'] == 16
    assert source['seed'] == 42 and source['attention_backend'] == 'fa2'
    assert source['initialization'] == 'seed42_from_zero_no_diagnostic_weights'
    assert source['data_uri'] == DATA_URI and source['dataset_revision'] == DATA_REVISION
    assert source['checkpoint_seconds'] == 1800 and source['evaluate_test'] is True
    assert evidence['complete'] and evidence['tests']['gpu_passed'] == 34
    assert evidence['tests']['failed'] == 0 and evidence['cli_resume']['passed']
    assert source['model'] == evidence['workload']
    c = source['model']
    assert (c['variant'],c['architecture'],c['streams']) == ('baseline','vanilla',1)
    assert (c['depth'],c['width'],c['heads'],c['sequence_length'],c['vocab_size']) == (8,512,4,2048,32768)
    assert (c['ffn_readout_width'],c['ffn_readout_init'],c['ffn_readout_backend']) == (64,'gradmax','triton')
    assert not c['activation_checkpoint'] and not c['fused_connection_grad']


def validate_complete(summary, source):
    assert summary['complete'] is True and summary['source'] == source
    assert summary['run_id'] == source['run_id'] and summary['variant'] == 'baseline'
    assert summary['steps'] == STEPS and summary['training_tokens'] == TOKENS
    assert summary['test_evaluated'] is True
    for name in ('final_dev_ce','final_test_ce','best_selected_test_ce'):
        assert math.isfinite(summary[name])
    assert set(summary['checkpoint_files']) == {'latest.pt','best.pt'}


def finite_formal_row(row):
    return (isinstance(row,dict) and type(row.get('step')) is int and
            0 < row['step'] <= STEPS and row.get('tokens') == row['step']*BATCH_TOKENS and
            isinstance(row.get('loss'),(int,float)) and math.isfinite(row['loss']))
