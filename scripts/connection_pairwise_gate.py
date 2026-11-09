"""Bounded pairwise activation gate: correctness, ABBA speed and NTP recovery."""
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(args,root,name,expected=0):
    print('BEGIN '+name,flush=True)
    with (root/(name+'.log')).open('w') as log:
        proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in proc.stdout:log.write(line);print(line,end='',flush=True)
        proc.wait()
    assert proc.returncode==expected,(name,proc.returncode)
    print('PASS '+name,flush=True)


def restart(root,activation,micro):
    output=root/('restart-'+activation);checkpoint=Path('../pairwise-check-checkpoints')/activation
    shutil.rmtree(output,ignore_errors=True);shutil.rmtree(checkpoint,ignore_errors=True)
    args=['-m','scripts.connection_train','--variant','baseline','--streams','1','--mlp-activation',activation,
          '--pairwise-backend','triton','--matched-backbone-init','--steps','128','--schedule-steps','305176',
          '--microbatch',str(micro),'--data-dir','../data','--attention-backend','hardware_auto',
          '--output-dir',str(output),'--checkpoint-dir',str(checkpoint),'--run-id','pairwise-check-'+activation,
          '--segment-seconds','3600','--segment-steps','25']
    run(args,root,activation+'-first',85)
    assert json.loads((output/'progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,activation+'-second')
    summary=json.loads((output/'summary.json').read_text());config=json.loads((output/'config.json').read_text())
    rows=[json.loads(s) for s in (output/'steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==128
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert config['model']['streams']==1 and config['model']['mlp_activation']==activation
    assert config['parameters']==(58728448 if activation=='pairwise' else 58720256)
    assert summary['peak_reserved_bytes']<.9*22000*1024**2
    return dict(summary=summary,config=config)


def main():
    root=Path('results/pairwise');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-v','tests/test_pairwise_activation.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=24,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_pairwise_micro'],root,'micro')
    for micro in (16,8,4):
        profiles={}
        for index,activation in enumerate(('relu2','pairwise','pairwise','relu2')):
            name=f'profile-{index}-{activation}-{micro}'
            run(['-m','scripts.connection_context_profile','--variant','baseline','--mlp-activation',activation,
                 '--pairwise-backend','triton','--sequence-length','2048','--microbatch',str(micro),
                 '--attention-backend','hardware_auto','--output',str(root/(name+'.json'))],root,name)
            profiles[name]=json.loads((root/(name+'.json')).read_text())
        if all(v['complete'] and v['status']=='fits' for v in profiles.values()):break
        assert all(v['status'] in ('fits','oom','over_budget') for v in profiles.values())
    else:raise RuntimeError('No common microbatch fits22GB')
    assert len({v['gpu_uuid'] for v in profiles.values()})==1
    assert len({json.dumps(v['input_selection'],sort_keys=True) for v in profiles.values()})==1
    training={activation:restart(root,activation,micro) for activation in ('relu2','pairwise')}
    # Exclusive-process GPUs: load diagnostic state only after all children exit.
    import torch
    state=torch.load('../pairwise-check-checkpoints/pairwise/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    weights={k:dict(min=v.min().item(),max=v.max().item(),mean_abs=v.abs().mean().item(),std=v.std().item())
             for k,v in state['model'].items() if k.endswith('.rotation')}
    assert len(weights)==8 and all(v['mean_abs']>1e-7 and v['std']>1e-7 for v in weights.values())
    means={a:statistics.mean(v['mean_seconds'] for v in profiles.values() if v['model']['mlp_activation']==a) for a in ('relu2','pairwise')}
    overhead=100*(means['pairwise']/means['relu2']-1)
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,microbatch=micro,profiles=profiles,training=training,trained_rotation_statistics=weights,
        gpu=training['pairwise']['config']['gpu'],capability=training['pairwise']['config']['gpu_capability'],
        activation_micro=json.loads((root/'micro.json').read_text()),mean_step_seconds=means,
        overhead_percent=overhead,speed_target_percent=5.,speed_target_met=overhead<=5.,
        pilot_dev_difference=training['pairwise']['summary']['final_dev_ce']-training['relu2']['summary']['final_dev_ce'],
        limitations='Single-seed128-step engineering pilot does not establish quality or10B stability. No formal run authorized.')
    shutil.rmtree('../pairwise-check-checkpoints')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','training','source')}),flush=True)


if __name__=='__main__':main()
