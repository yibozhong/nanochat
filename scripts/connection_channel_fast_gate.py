"""Bounded H200 channelwise optimization gate; no formal run or W&B."""
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from scripts.connection_gated_hc_gate import run


def main():
    root=Path('results/channel-fast');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-v','tests/test_channel_mixing_fast.py',
        'tests/test_channel_mixing.py::test_fused_all_gradients_against_fp64',
        'tests/test_compact_routing.py::test_fp64_all_gradients_and_determinism',
        'tests/test_compact_routing.py::test_nvidia_read_equivalence_and_all_gradients',
        'tests/test_compact_routing.py::test_default_and_checkpoint_schema',
        'tests/test_connection_hardware_backend.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=48,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_channel_fast_micro'],root,'micro')
    profiles={}
    def measure(name,kernel,compact,variant='mhc_channel_no_sinkhorn'):
        args=['-m','scripts.connection_context_profile','--variant',variant,'--sequence-length','2048',
              '--microbatch','16','--attention-backend','hardware_auto','--channel-mixing-kernel',kernel,
              '--output',str(root/(name+'.json'))]
        if compact:args+=['--compact-routing']
        run(args,root,name)
        p=json.loads((root/(name+'.json')).read_text())
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60 and p['causality_passed']
        assert all(math.isfinite(v) for v in p['losses']) and len(p['losses'])==60
        profiles[name]=p;return p
    measure('baseline','legacy',False,'baseline')
    choices=[]
    for k in ('legacy','split','fused','fused_wide'):
        for compact in (False,True):
            p=measure(f'{k}-compact{int(compact)}',k,compact)
            choices.append((p['mean_seconds'],k,compact))
    _,kernel,compact=min(choices)
    # Fresh processes, fixed seed/data, A-B-B-A order. Do not compare
    # a single favorable step or local mixer time to the90ms full-step target.
    abba=[]
    for index,(k,cp) in enumerate([('legacy',False),(kernel,compact),(kernel,compact),('legacy',False)]):
        abba.append(measure(f'abba-{index}',k,cp))
    old=statistics.mean([abba[0]['mean_seconds'],abba[3]['mean_seconds']])
    new=statistics.mean([abba[1]['mean_seconds'],abba[2]['mean_seconds']])
    args=['-m','scripts.connection_train','--variant','mhc_channel_no_sinkhorn','--matched-backbone-init',
          '--channel-mixing-kernel',kernel,'--steps','128','--schedule-steps','305176','--microbatch','16',
          '--data-dir','../data','--attention-backend','hardware_auto','--output-dir',str(root/'restart'),
          '--checkpoint-dir','../channel-fast-check-checkpoints','--run-id','channel-fast-restart',
          '--segment-seconds','3600','--segment-steps','25']
    if compact:args+=['--compact-routing']
    run(args,root,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,'restart-second')
    restart=json.loads((root/'restart/summary.json').read_text())
    config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert restart['complete'] and restart['steps']==128 and restart['final_dev_ce']<restart['initial_dev_ce']
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert len({p['gpu_uuid'] for p in profiles.values()})==1 and len({p['input_sha256'] for p in profiles.values()})==1
    # Parent owns no CUDA context while any child is active.
    import torch
    state=torch.load('../channel-fast-check-checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    micro=json.loads((root/'micro/summary.json').read_text())
    assert micro['gpu_uuid']==abba[0]['gpu_uuid'] and micro['complete']
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,profiles=profiles,micro=micro,restart=restart,
        selected_kernel=kernel,selected_compact_routing=compact,old_seconds=old,new_seconds=new,
        speedup_percent=100*(1-new/old),target_seconds=.09,target_reached=new<=.09,
        recommended=new<old*.98,default_changed=False,new_formal_run=False,
        limitations='Single H200; ABBA60-update trials and short resume do not establish broader hardware performance or quality.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','micro','restart','source')}),flush=True)


if __name__=='__main__':main()
