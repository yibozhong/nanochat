"""Validate bounded evidence. Speed target failure is a result, not a math pass."""
import math


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and v['source']==source
    assert v['checks']==dict(tests=24,failures=0,errors=0,skipped=0)
    assert v['capability']==[9,0] and v['microbatch'] in (16,8,4)
    profiles=list(v['profiles'].values())
    assert len(profiles)==4 and sorted(p['model']['mlp_activation'] for p in profiles)==['pairwise','pairwise','relu2','relu2']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source and p['causality_passed']
        assert p['variant']=='baseline' and p['model']['architecture']=='vanilla'
        assert p['model']['pairwise_backend']=='triton' and p['microbatch']==v['microbatch']
        assert p['parameters']==(58728448 if p['model']['mlp_activation']=='pairwise' else 58720256)
        assert p['batch_tokens']==32768 and p['sequence_length']==2048 and p['logging_backend']=='local'
        assert p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    for activation in ('relu2','pairwise'):
        t=v['training'][activation];r=t['summary'];c=t['config']
        assert r['complete'] and r['steps']==128 and r['training_tokens']==128*32768
        assert r['source']==source and r['variant']=='baseline'
        assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
        assert r['peak_reserved_bytes']<.9*22000*1024**2
        assert c['logging_backend']=='local' and c['model']['mlp_activation']==activation
        assert c['model']['pairwise_backend']=='triton' and c['model']['streams']==1
    w=v['trained_rotation_statistics'];assert len(w)==8
    assert all(all(math.isfinite(x) for x in s.values()) and s['mean_abs']>1e-7 and s['std']>1e-7 for s in w.values())
    means=v['mean_step_seconds']
    assert all(math.isfinite(x) and x>0 for x in means.values())
    overhead=100*(means['pairwise']/means['relu2']-1)
    assert abs(overhead-v['overhead_percent'])<1e-8
    assert v['speed_target_percent']==5 and v['speed_target_met']==(overhead<=5)
