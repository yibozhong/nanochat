"""Bounded correctness, same-card implementation timing and real NTP restart."""
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from scripts.connection_attnres_gate import run


def main():
    root=Path('results/stream-gate');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-v','tests/test_stream_gated_hc.py','tests/test_connection_hardware_backend.py',
        '--junitxml='+str(root/'tests.xml')],root,'tests')
    counts={k:sum(int(s.get(k,0)) for s in ET.parse(root/'tests.xml').getroot().findall('testsuite')) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=17,failures=0,errors=0,skipped=0),counts
    profiles={}
    def measure(name,backend,variant,micro):
        run(['-m','scripts.connection_stream_gate_profile','--gate-backend',backend,
             '--variant',variant,'--sequence-length','2048','--microbatch',str(micro),
             '--attention-backend','hardware_auto','--output',str(root/(name+'.json'))],root,name)
        p=json.loads((root/(name+'.json')).read_text());p['gate_backend']=backend
        assert p['status'] in ('fits','oom','over_budget'),p
        return p
    attempts=[]
    for micro in (16,8,4):
        trial={}
        for backend,variant in (('hc','mhc_no_sinkhorn'),('torch','stream_gated_hc'),('fused','stream_gated_hc')):
            trial[backend]=measure(f'budget-{backend}-{micro}',backend if backend!='hc' else 'fused',variant,micro)
        attempts.append(trial)
        if all(p['status']=='fits' and p['complete'] for p in trial.values()):
            profiles.update(hc=trial['hc'],**{'abba-0':trial['torch'],'abba-1':trial['fused']});break
    else:raise RuntimeError('No common microbatch fits22GB')
    for i,backend in ((2,'fused'),(3,'torch')):
        profiles['abba-'+str(i)]=measure('abba-'+str(i),backend,'stream_gated_hc',micro)
    assert all(p['status']=='fits' and p['complete'] and p['optimizer_updates']==60 and p['causality_passed'] for p in profiles.values())
    assert len({p['gpu_uuid'] for p in profiles.values()})==1 and len({p['input_sha256'] for p in profiles.values()})==1
    assert profiles['abba-1']['parameters']-profiles['hc']['parameters']==16777216
    run(['-m','scripts.connection_stream_gate_probe','--microbatch',str(micro)],root,'first-step-probe')
    probe=json.loads((root/'first-step-probe.json').read_text());assert probe['passed'],probe
    run(['-m','scripts.connection_stream_gate_micro','--microbatch',str(micro)],root,'micro')
    args=['-m','scripts.connection_train','--variant','stream_gated_hc','--matched-backbone-init',
          '--steps','128','--schedule-steps','305176','--microbatch',str(micro),'--data-dir','../data',
          '--attention-backend','hardware_auto','--output-dir',str(root/'restart'),
          '--checkpoint-dir','../stream-gate-check-checkpoints','--run-id','stream-gate-check',
          '--segment-seconds','3600','--segment-steps','25']
    run(args,root,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,'restart-second')
    restart=json.loads((root/'restart/summary.json').read_text());config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert restart['complete'] and restart['steps']==128 and restart['final_dev_ce']<restart['initial_dev_ce']
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    import torch
    checkpoint=torch.load('../stream-gate-check-checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    reference=statistics.mean(profiles['abba-'+str(i)]['mean_seconds'] for i in (0,3))
    fused=statistics.mean(profiles['abba-'+str(i)]['mean_seconds'] for i in (1,2))
    result=dict(complete=True,formal_training=False,new_formal_run=False,source=json.loads(Path('source.json').read_text()),
        formula='g_s=sigmoid(RMS_channel(x_s) @ Wg_s.T); output=(x*g) @ H + branch*post',
        extra_gate_parameters=16777216,microbatch=micro,checks=counts,profiles=profiles,budget_attempts=attempts,
        first_step_probe=probe,restart=restart,micro=json.loads((root/'micro/summary.json').read_text()),
        reference_seconds=reference,fused_seconds=fused,speedup_over_torch=reference/fused,
        overhead_over_hc_percent=100*(fused/profiles['hc']['mean_seconds']-1),
        limitations='Short same-card validation only; no10B quality or long-term stability claim.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','budget_attempts','restart','micro','first_step_probe')}),flush=True)


if __name__=='__main__':main()
