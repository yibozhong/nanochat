"""Read-only validation of the historical GPU evidence; never submits jobs."""
import math

def validate(value,state):
    assert value['complete'] and not value['formal_training'] and not value['preflight']
    assert value['source']==state['source']
    assert value['checks']==dict(tests=25,failures=0,errors=0,skipped=0)
    assert value['restart']['complete'] and value['restart']['steps']==128
    assert all(math.isfinite(value['restart'][k]) for k in ('initial_dev_ce','final_dev_ce'))
    assert value['restart']['final_dev_ce']<value['restart']['initial_dev_ce']
    assert value['microbatch'] in (16,8,4)
    assert set(value['profiles'])=={'baseline','mhc_no_sinkhorn','mhc_channel_no_sinkhorn','gated_hc'}
    for variant,p in value['profiles'].items():
        assert p['variant']==variant and p['microbatch']==value['microbatch']
        assert p['source']==state['source'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60
        assert p['causality_passed'] and math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert len(p['losses'])==60 and all(math.isfinite(loss) for loss in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(t) and t>0 for t in p['step_seconds'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
    assert len({p['gpu_uuid'] for p in value['profiles'].values()})==1
    assert len({p['input_sha256'] for p in value['profiles'].values()})==1
    assert value['profiles']['gated_hc']['parameters']==59679157
    micro=value['micro'];assert micro['complete'] and micro['shape']==[value['microbatch'],2048,512,4]
    assert micro['gpu_uuid']==value['profiles']['gated_hc']['gpu_uuid']
    assert set(micro['results'])=={'fused','compiled_torch'}
    for r in micro['results'].values():
        assert math.isfinite(r['mean_seconds']) and r['mean_seconds']>0
