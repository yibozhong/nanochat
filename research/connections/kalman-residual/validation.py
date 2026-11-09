"""Fail-closed checks used by both publication and the unique10B submission."""
import math


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and not v['preflight']
    assert v['source']==source
    assert v['checks']==dict(tests=21,failures=0,errors=0,skipped=0)
    assert v['capability'] in ([8,0],[9,0],[12,0])
    assert v['microbatch'] in (16,8,4)
    r=v['restart']
    assert r['complete'] and r['steps']==128 and r['variant']=='kalman_residual' and r['source']==source
    assert r['training_tokens']==128*32768 and math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['peak_reserved_bytes']<.9*22000*1024**2
    scales=v['trained_noise_statistics']
    assert len(scales)==48
    assert all(all(math.isfinite(z) for z in s.values())  for s in scales.values())
    assert sum(k.endswith('.slope') and s['mean_abs_change']>1e-7 and s['channel_std']>1e-7 for k,s in scales.items())==16
    b=v['baseline_restart']
    assert b['complete'] and b['steps']==128 and b['variant']=='baseline' and b['source']==source
    assert b['training_tokens']==128*32768 and b['peak_reserved_bytes']<.9*22000*1024**2
    assert math.isfinite(b['final_dev_ce']) and b['final_dev_ce']<b['initial_dev_ce']
    profiles=list(v['profiles'].values())
    assert sorted(p['variant'] for p in profiles)==['baseline','baseline','kalman_residual','kalman_residual']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source
        assert p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert p['causality_passed'] and p['microbatch']==v['microbatch'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['logging_backend']=='local' and p['model']['architecture']=='vanilla'
        assert p['parameters']==(58744832 if p['variant']=='kalman_residual' else 58720256)
        if p['variant']=='kalman_residual':assert p['model']['streams']==1
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert p['peak_reserved_bytes']<.9*22000*1024**2
