"""Bounded play-FFN validation: all VJPs, ABBA speed, paired 128-step NTP."""
import json
import math
from pathlib import Path
import shutil
import statistics
import xml.etree.ElementTree as ET

from scripts.connection_skip_scale_gate import run


def main():
    root=Path('results/play');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-xv','tests/test_play_memory.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=22,failures=0,errors=0,skipped=0),counts
    for micro in (16,8,4):
        profiles={}
        for index,memory in enumerate(('none','play','play','none')):
            name=f'profile-{index}-{memory}-{micro}'
            run(['-m','scripts.connection_context_profile','--variant','baseline','--mlp-memory',memory,
                 '--sequence-length','2048','--microbatch',str(micro),'--attention-backend','hardware_auto',
                 '--output',str(root/(name+'.json'))],root,name)
            profiles[name]=json.loads((root/(name+'.json')).read_text())
        if all(p['complete'] and p['status']=='fits' for p in profiles.values()):break
        assert all(p['status'] in ('fits','oom','over_budget') for p in profiles.values())
    else:raise RuntimeError('No common microbatch fits22GB')
    assert len({p['gpu_uuid'] for p in profiles.values()})==len({p['input_sha256'] for p in profiles.values()})==1
    assert all(p['optimizer_updates']==60 and p['causality_passed'] for p in profiles.values())
    summaries={};statistics_by_memory={}
    for memory in ('none','play'):
        output=root/('train-'+memory);checkpoint=Path('../play-checkpoints-'+memory)
        assert not output.exists() and not checkpoint.exists()
        args=['-m','scripts.connection_train','--variant','baseline','--streams','1','--mlp-memory',memory,
              '--steps','128','--schedule-steps','305176','--microbatch',str(micro),
              '--data-dir','../data','--attention-backend','hardware_auto',
              '--output-dir',str(output),'--checkpoint-dir',str(checkpoint),
              '--run-id','play-check-'+memory,'--segment-seconds','3600','--segment-steps','25']
        run(args,root,'train-'+memory+'-first',85)
        assert json.loads((output/'progress.json').read_text())['step']==25
        run(args+['--resume','auto','--segment-steps','0'],root,'train-'+memory+'-resumed')
        summary=json.loads((output/'summary.json').read_text())
        config=json.loads((output/'config.json').read_text())
        rows=[json.loads(s) for s in (output/'steps.jsonl').read_text().splitlines()]
        assert summary['complete'] and summary['steps']==128
        assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
        assert summary['final_dev_ce']<summary['initial_dev_ce']
        assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
        assert config['parameters']==(58720256 if memory=='none' else 58753024)
        summaries[memory]=summary
        # CPU load only: do not create a parent CUDA context between subprocesses.
        import torch
        state=torch.load(checkpoint/'latest.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(p).all() for p in state['model'].values())
        if memory=='play':
            weights={n:dict(mean=p.mean().item(),std=p.std().item(),max_abs=p.abs().max().item())
                     for n,p in state['model'].items() if n.endswith('memory_weight')}
            radii={n:dict(min=torch.nn.functional.softplus(p).min().item(),
                         max=torch.nn.functional.softplus(p).max().item(),
                         mean_change=(torch.nn.functional.softplus(p)-.25).abs().mean().item())
                   for n,p in state['model'].items() if n.endswith('raw_radius')}
            assert len(weights)==len(radii)==8
            assert all(p['max_abs']>1e-7 for p in weights.values())
            assert all(p['mean_change']>1e-7 for p in radii.values())
            statistics_by_memory=dict(weights=weights,radii=radii)
        del state
        shutil.rmtree(checkpoint)
    means={memory:statistics.mean(p['mean_seconds'] for p in profiles.values() if p['model']['mlp_memory']==memory)
           for memory in ('none','play')}
    result=dict(complete=True,formal_training=False,checks=counts,microbatch=micro,
        source=json.loads(Path('source.json').read_text()),profiles=profiles,training=summaries,
        memory_statistics=statistics_by_memory,mean_step_seconds=means,
        overhead_percent=100*(means['play']/means['none']-1),
        dev_ce_difference=summaries['play']['final_dev_ce']-summaries['none']['final_dev_ce'],
        limitations='Single-seed128step engineering/pilot check, not evidence of long-run quality or stability. No cached decoding support.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','training','source','memory_statistics')}),flush=True)


if __name__=='__main__':main()
