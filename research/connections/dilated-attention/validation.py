"""Fail-closed collection of bounded GPU evidence; never starts formal training."""
import json
import math
import statistics


def validate(v,source):
    assert v['complete'] and not v['formal_training'] and v['source']==source
    assert v['checks']==dict(tests=30,failures=0,errors=0,skipped=0)
    assert v['capability']==[9,0] and v['microbatch'] in (16,8,4)
    profiles=list(v['profiles'].values());patterns=('full','sink_swa','progressive_dilated')
    assert len(profiles)==6 and len({p['gpu_uuid'] for p in profiles})==1
    assert len({json.dumps(p['input_selection'],sort_keys=True) for p in profiles})==1
    for pattern in patterns:
        selected=[p for p in profiles if p['model']['attention_pattern']==pattern]
        assert len(selected)==2
        assert abs(statistics.mean(p['mean_seconds'] for p in selected)-v['mean_step_seconds'][pattern])<1e-10
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['source']==source and p['causality_passed']
        assert p['variant']=='baseline' and p['model']['architecture']=='vanilla'
        assert p['model']['sparse_backend']=='triton' and p['microbatch']==v['microbatch']
        assert p['parameters']==58720256 and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['logging_backend']=='local' and p['optimizer_updates']==60 and p['warmup_steps']==20 and p['timed_steps']==40
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    for pattern in patterns:
        t=v['training'][pattern];r=t['summary'];c=t['config']
        assert r['complete'] and r['steps']==128 and r['training_tokens']==128*32768
        assert r['source']==source and r['variant']=='baseline'
        assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
        assert r['peak_reserved_bytes']<.9*22000*1024**2
        assert c['logging_backend']=='local' and c['model']['attention_pattern']==pattern
        assert c['model']['sparse_backend']=='triton' and c['model']['streams']==1 and c['parameters']==58720256
