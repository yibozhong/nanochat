"""Bounded implementation screen. No formal training or W&B."""
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


def restart(root,pattern,micro):
    output=root/('restart-'+pattern);checkpoint=Path('../dilated-check-checkpoints')/pattern
    args=['-m','scripts.connection_train','--variant','baseline','--streams','1','--attention-pattern',pattern,
          '--matched-backbone-init','--steps','128','--schedule-steps','305176','--microbatch',str(micro),
          '--data-dir','../data','--attention-backend','hardware_auto','--output-dir',str(output),
          '--checkpoint-dir',str(checkpoint),'--run-id','dilated-check-'+pattern,'--segment-seconds','3600','--segment-steps','25']
    run(args,root,pattern+'-first',85)
    assert json.loads((output/'progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,pattern+'-second')
    summary=json.loads((output/'summary.json').read_text());config=json.loads((output/'config.json').read_text())
    rows=[json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==128
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert config['model']['streams']==1 and config['model']['attention_pattern']==pattern
    assert config['parameters']==58720256 and summary['peak_reserved_bytes']<.9*22000*1024**2
    return dict(summary=summary,config=config)


def main():
    root=Path('results/dilated');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-v','tests/test_dilated_attention.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=29,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_dilated_micro'],root,'micro')
    patterns=('full','sink_swa','progressive_dilated')
    for micro in (16,8,4):
        profiles={}
        for index,pattern in enumerate((*patterns,*reversed(patterns))):
            name=f'profile-{index}-{pattern}-{micro}'
            run(['-m','scripts.connection_context_profile','--variant','baseline','--attention-pattern',pattern,
                 '--sequence-length','2048','--microbatch',str(micro),'--attention-backend','hardware_auto',
                 '--output',str(root/(name+'.json'))],root,name)
            profiles[name]=json.loads((root/(name+'.json')).read_text())
        if all(p['complete'] and p['status']=='fits' for p in profiles.values()):break
        assert all(p['status'] in ('fits','oom','over_budget') for p in profiles.values())
    else:raise RuntimeError('No common microbatch fits22GB')
    assert len({p['gpu_uuid'] for p in profiles.values()})==1
    assert len({json.dumps(p['input_selection'],sort_keys=True) for p in profiles.values()})==1
    training={p:restart(root,p,micro) for p in patterns}
    # Check all final diagnostic weights only after child CUDA processes finish.
    import torch
    for pattern in patterns:
        state=torch.load(f'../dilated-check-checkpoints/{pattern}/latest.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in state['model'].values())
        del state
    means={pattern:statistics.mean(p['mean_seconds'] for p in profiles.values() if p['model']['attention_pattern']==pattern) for pattern in patterns}
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,microbatch=micro,profiles=profiles,training=training,
        gpu=training['progressive_dilated']['config']['gpu'],capability=training['progressive_dilated']['config']['gpu_capability'],
        activation_micro=json.loads((root/'micro.json').read_text()),mean_step_seconds=means,
        relative_to_full_percent={p:100*(means[p]/means['full']-1) for p in patterns},
        final_dev_ce={p:training[p]['summary']['final_dev_ce'] for p in patterns},
        limitations='Single-seed128-step engineering pilot does not establish quality or10B stability; cache fallback functional, not optimized. No formal run authorized.')
    shutil.rmtree('../dilated-check-checkpoints')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','training','source','activation_micro')}),flush=True)


if __name__=='__main__':main()
