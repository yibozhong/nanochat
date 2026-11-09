"""Validate the completed play-memory gate before one formal10B submission."""
import math


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and v['source']==source
    assert v['checks']==dict(tests=23,failures=0,errors=0,skipped=0)
    assert v['microbatch'] in (16,8,4)
    assert set(v['training'])=={'none','play'}
    for memory,r in v['training'].items():
        assert r['complete'] and r['steps']==128 and r['training_tokens']==128*32768
        assert r['variant']=='baseline' and r['source']==source
        assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
        assert r['peak_reserved_bytes']<.9*22000*1024**2
    profiles=list(v['profiles'].values())
    assert sorted(p['model']['mlp_memory'] for p in profiles)==['none','none','play','play']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source
        assert p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert p['causality_passed'] and p['microbatch']==v['microbatch']
        assert p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['logging_backend']=='local' and p['variant']=='baseline'
        c=p['model'];assert c['architecture']=='vanilla' and c['depth']==8 and c['width']==512 and c['heads']==4
        assert p['parameters']==(58753024 if c['mlp_memory']=='play' else 58720256)
        if c['mlp_memory']=='play':assert c['streams']==1 and c['play_radius_init']==.25
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    weights=v['memory_statistics']['weights'];radii=v['memory_statistics']['radii']
    assert len(weights)==len(radii)==8
    assert all(all(math.isfinite(x) for x in w.values()) and w['max_abs']>1e-7 for w in weights.values())
    assert all(all(math.isfinite(x) for x in r.values()) and r['min']>0 and r['mean_change']>1e-7 for r in radii.values())
