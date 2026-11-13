"""Stdlib policy for one replacement full-dense vocabulary-gate 10B run."""
import math
PLAN='vocab-gate-dense-10b-20261113'
STEPS=305176
BATCH_TOKENS=32768
TOKENS=STEPS*BATCH_TOKENS
MODEL=dict(variant='baseline',depth=8,width=512,heads=4,sequence_length=2048,
    vocab_size=32768,streams=1,sinkhorn_iterations=20,activation_checkpoint=False,
    connection_backend='nvidia',fused_connection_grad=False,architecture='vanilla',
    window_pattern='SSSL',attention_kernel='original',compact_routing=False,
    attnres_block_size=2,attnres_block_size_unit='sublayers',vocab_gate=True,
    vocab_gate_alpha=.5,vocab_gate_backend='triton',vocab_gate_chunk=2048)


def validate_source(s):
    assert s['plan']==PLAN and s['formal_training'] is True
    assert s['steps']==STEPS and s['training_tokens']==TOKENS
    assert (s['batch_tokens'],s['microbatch'],s['seed'])==(BATCH_TOKENS,16,42)
    assert s['model']==MODEL and s['attention_backend']=='fa2'
    assert s['initialization']=='seed42_from_zero_no_diagnostic_weights'
    assert s['checkpoint_seconds']==1800 and s['evaluate_test'] is True
    assert s['data_uri']=='osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-7685535.tar'
    assert s['dataset_revision']=='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'


def validate_gate(g,s):
    assert g['complete'] and g['source']==s
    assert g['tests_failed']==0 and g['tests_passed']>=48
    assert g['finite_training_steps']>=200 and g['cli_resume_passed']
    assert g['causal_cache_passed'] and g['optimizer_rng_resume_passed']
    assert g['gate_gradient_nonzero'] and g['trained_gate_nonzero']
    for key in ('baseline_step_ms','gate_step_ms'):
        assert math.isfinite(g[key]) and g[key]>0


def validate_complete(r,s):
    assert r['complete'] is True and r['source']==s and r['run_id']==s['run_id']
    assert r['variant']=='baseline' and r['steps']==STEPS and r['training_tokens']==TOKENS
    assert r['test_evaluated'] is True
    for key in ('final_dev_ce','final_test_ce','best_selected_test_ce'):assert math.isfinite(r[key])
    assert set(r['checkpoint_files'])=={'latest.pt','best.pt'}


def finite_formal_row(row):
    return (isinstance(row,dict) and type(row.get('step')) is int and 0<row['step']<=STEPS and
            row.get('tokens')==row['step']*BATCH_TOKENS and isinstance(row.get('loss'),(int,float)) and math.isfinite(row['loss']))
