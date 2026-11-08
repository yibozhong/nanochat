"""Fail-closed checks used by both publication and the unique10B submission."""
import math


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and not v['preflight']
    assert v['source']==source
    assert v['checks']==dict(tests=24,failures=0,errors=0,skipped=0)
    assert v['capability'] in ([8,0],[9,0],[12,0])
    assert v['microbatch'] in (16,8,4)
    r=v['restart']
    assert r['complete'] and r['steps']==128 and r['variant']=='attnres_scale' and r['source']==source
    assert r['training_tokens']==128*32768 and math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['peak_reserved_bytes']<.9*22000*1024**2
    scales=v['trained_scale_statistics']
    assert len(scales)==16
    assert all(all(math.isfinite(z) for z in s.values()) and s['mean_abs_change']>1e-7 and s['channel_std']>1e-7 for s in scales.values())
    profiles=list(v['profiles'].values())
    assert sorted(p['variant'] for p in profiles)==['attnres','attnres','attnres_scale','attnres_scale','baseline','baseline']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source
        assert p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert p['causality_passed'] and p['microbatch']==v['microbatch'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['logging_backend']=='local' and p['model']['architecture']=='vanilla'
        assert p['parameters']==({'attnres_scale':58782720,'attnres':58737664,'baseline':58720256}[p['variant']])
        if p['variant']=='attnres_scale':assert p['model']['streams']==1
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert p['peak_reserved_bytes']<.9*22000*1024**2
