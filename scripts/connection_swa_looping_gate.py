"""Bounded SSSL/SWA-only looping validation; no formal training or W&B."""
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
        for line in proc.stdout:
            log.write(line);print(line,end='',flush=True)
        proc.wait()
    assert proc.returncode==expected,(name,proc.returncode)
    print('PASS '+name,flush=True)


def main():
    root=Path('results/swa-looping');root.mkdir(parents=True,exist_ok=True)
    run(['-m','pytest','-v','tests/test_swa_looping.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=21,failures=0,errors=0,skipped=0),counts
    options=dict(full=[],sssl=['--swa-looping','--swa-loops','1'],
                 sssl_loop3=['--swa-looping','--swa-loops','3'])
    for micro in (16,8,4):
        profiles={}
        for index,mode in enumerate(('full','sssl','sssl_loop3','sssl_loop3','sssl','full')):
            name=f'profile-{index}-{mode}-{micro}'
            run(['-m','scripts.connection_context_profile','--variant','baseline',
                 '--sequence-length','2048','--microbatch',str(micro),'--attention-backend','fa3',
                 '--swa-window','512','--window-pattern','SSSL',*options[mode],
                 '--output',str(root/(name+'.json'))],root,name)
            profiles[name]=dict(json.loads((root/(name+'.json')).read_text()),mode=mode)
        if all(p['complete'] and p['status']=='fits' for p in profiles.values()):break
        assert all(p['status'] in ('fits','oom','over_budget') for p in profiles.values()),profiles
    else:raise RuntimeError('No common microbatch fits the22GB budget')
    assert len({p['gpu_uuid'] for p in profiles.values()})==1
    assert len({p['input_sha256'] for p in profiles.values()})==1
    assert all(p['parameters']==58720256 and p['optimizer_updates']==60 and p['causality_passed'] for p in profiles.values())
    assert all(p['actual_attention_backend']=='fa3' for p in profiles.values())
    checkpoint_dir=Path('../swa-looping-check-checkpoints')
    shutil.rmtree(root/'restart',ignore_errors=True)
    shutil.rmtree(checkpoint_dir,ignore_errors=True)
    args=['-m','scripts.connection_train','--variant','baseline','--streams','1',
          '--swa-looping','--swa-loops','3','--swa-window','512','--window-pattern','SSSL',
          '--steps','128','--schedule-steps','305176','--microbatch',str(micro),
          '--data-dir','../data','--attention-backend','fa3',
          '--output-dir',str(root/'restart'),'--checkpoint-dir',str(checkpoint_dir),
          '--run-id','swa-looping-restart-check','--segment-seconds','3600','--segment-steps','25']
    run(args,root,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text())
    cfg=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==128
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']
    assert cfg['logging_backend']=='local' and cfg['protocol']['schedule_steps']==305176
    assert cfg['actual_attention_backend']=='fa3' and cfg['parameters']==58720256
    assert cfg['model']['swa_looping'] and cfg['model']['swa_loops']==3 and cfg['model']['swa_window']==512
    # Import torch only after all subprocesses exit, including on exclusive-process GPUs.
    import torch
    from nanochat.connection_cost import model_cost
    from nanochat.connection_model import ConnectionConfig
    state=torch.load(checkpoint_dir/'latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    branch_norms={n:v.float().norm().item() for n,v in state['model'].items() if n.endswith('c_proj.weight')}
    assert len(branch_norms)==16 and all(v>0 for v in branch_norms.values())
    assert summary['peak_reserved_bytes']<.9*22000*1024**2
    means={mode:statistics.mean(p['mean_seconds'] for p in profiles.values() if p['mode']==mode) for mode in options}
    costs={mode:model_cost(ConnectionConfig.from_checkpoint(next(p['model'] for p in profiles.values() if p['mode']==mode)),batch=16) for mode in options}
    result=dict(complete=True,formal_training=False,automatic_training=False,checks=counts,
        source=json.loads(Path('source.json').read_text()),microbatch=micro,gpu=cfg['gpu'],
        profiles=profiles,restart=summary,branch_projection_norms=branch_norms,
        mean_step_seconds=means,matrix_costs=costs,
        loop_overhead_vs_sssl_percent=100*(means['sssl_loop3']/means['sssl']-1),
        loop_overhead_vs_full_percent=100*(means['sssl_loop3']/means['full']-1),
        limitations='128-step diagnostic checks stability/restart only; no long-run quality conclusion or10B authorization. Cached decoding is not implemented; only native-window FA3 is supported.')
    shutil.rmtree(checkpoint_dir)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart','source','matrix_costs')}),flush=True)


if __name__=='__main__':main()
