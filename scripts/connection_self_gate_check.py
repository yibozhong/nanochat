"""Bounded self-gate checks, same-card training profiles and25->128 resume."""
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from scripts.connection_gated_hc_gate import run


def main():
    root=Path('results/self-gate');root.mkdir(parents=True,exist_ok=True)
    run(['-m','scripts.connection_self_gate_probe'],root,'first-step-probe')
    run(['-m','pytest','-v','tests/test_self_gated_hc.py',
        'tests/test_connection_hardware_backend.py',
        'tests/test_gated_hc.py::test_identity_init_routing_rng_optimizer',
        '--junitxml='+str(root/'tests.xml')],root,'tests')
    counts={k:sum(int(s.get(k,0)) for s in ET.parse(root/'tests.xml').getroot().findall('testsuite'))
            for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=17,failures=0,errors=0,skipped=0),counts
    probe=json.loads((root/'first-step-probe.json').read_text())
    assert probe['passed'],probe
    run(['-m','scripts.connection_self_gate_micro'],root,'micro')
    profiles={}
    def measure(name,variant):
        run(['-m','scripts.connection_context_profile','--variant',variant,'--sequence-length','2048',
             '--microbatch','16','--attention-backend','hardware_auto','--output',str(root/(name+'.json'))],root,name)
        p=json.loads((root/(name+'.json')).read_text())
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60 and p['causality_passed']
        assert len(p['losses'])==60 and all(math.isfinite(v) for v in p['losses'])
        profiles[name]=p
    for i,v in enumerate(('mhc_no_sinkhorn','self_gated_hc','self_gated_hc','mhc_no_sinkhorn')):measure('abba-'+str(i),v)
    assert profiles['abba-1']['parameters']-profiles['abba-0']['parameters']==65536
    assert len({p['gpu_uuid'] for p in profiles.values()})==1 and len({p['input_sha256'] for p in profiles.values()})==1
    args=['-m','scripts.connection_train','--variant','self_gated_hc','--matched-backbone-init',
          '--steps','128','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
          '--attention-backend','hardware_auto','--output-dir',str(root/'restart'),
          '--checkpoint-dir','../self-gate-check-checkpoints','--run-id','self-gate-check',
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
    checkpoint=torch.load('../self-gate-check-checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    old=statistics.mean(profiles['abba-'+str(i)]['mean_seconds'] for i in (0,3))
    new=statistics.mean(profiles['abba-'+str(i)]['mean_seconds'] for i in (1,2))
    result=dict(complete=True,formal_training=False,new_formal_run=False,
        source=json.loads(Path('source.json').read_text()),formula='x*silu(a*x+b)',first_step_probe=probe,
        placement='residual input before shared no-Sinkhorn H at attention and MLP connections',
        extra_gate_parameters=65536,checks=counts,profiles=profiles,restart=restart,
        micro=json.loads((root/'micro/summary.json').read_text()),old_seconds=old,new_seconds=new,
        overhead_percent=100*(new/old-1),target90ms_reached=new<=.09,
        limitations='One H200 and short NTP validation only; identity initialization does not guarantee long-term stability; no10B quality claim.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart','micro','source')}),flush=True)


if __name__=='__main__':main()
