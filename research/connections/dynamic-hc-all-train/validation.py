"""Fail-closed checks used by both publication and the unique10B submission."""
import math


def validate_gates(gates):
    assert len(gates)==98
    assert all(all(math.isfinite(z) for z in s.values()) and s['mean_abs_change']>1e-7 for s in gates.values())
    slopes={k:s for k,s in gates.items() if k.endswith('_a')}
    assert len(slopes)==49 and all(s['channel_std']>1e-7 for s in slopes.values())


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and not v['preflight']
    assert v['source']==source
    assert v['checks']==dict(tests=29,failures=0,errors=0,skipped=0)
    assert v['capability'] in ([8,0],[9,0])
    assert v['microbatch'] in (16,8,4)
    r=v['restart']
    assert r['complete'] and r['steps']==128 and r['variant']=='dynamic_hc_all' and r['source']==source
    assert r['training_tokens']==128*32768 and math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['peak_reserved_bytes']<.9*22000*1024**2
    validate_gates(v['trained_gate_statistics'])
    profiles=list(v['profiles'].values())
    assert sorted(p['variant'] for p in profiles)==['dynamic_hc_all','dynamic_hc_all','mhc_no_sinkhorn','mhc_no_sinkhorn','scaled_hc_all','scaled_hc_all']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source
        assert p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert p['causality_passed'] and p['microbatch']==v['microbatch'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['logging_backend']=='local' and p['model']['architecture']=='vanilla'
        assert p['parameters']==({'dynamic_hc_all':59716021,'scaled_hc_all':59615669,'mhc_no_sinkhorn':59515317}[p['variant']])
        assert p['model']['streams']==4
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    assert v['micro']['complete'] and v['micro']['shape']==[v['microbatch'],2048,512,4]
    assert v['micro']['gpu_uuid']==profiles[0]['gpu_uuid']
    assert set(v['micro']['results'])=={'read_static','read_dynamic','write_static','write_dynamic'}
