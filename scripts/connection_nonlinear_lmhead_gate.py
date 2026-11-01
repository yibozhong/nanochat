"""Nonlinear LM-head regression checks, cost, and fresh-process restart."""
import json
import math
import os
from pathlib import Path
import sys
import uuid
import xml.etree.ElementTree as ET

from scripts.connection_pretrain_gate import execute


def main():
    root=Path('results');source=json.loads(Path('source.json').read_text())
    env=os.environ.copy();env['NVIDIA_TF32_OVERRIDE']='0'
    tests=['tests/test_nonlinear_lmhead.py']
    execute([sys.executable,'-m','pytest',*tests,'-v','--junitxml=results/nonlinear-tests.xml'],root/'tests.log',env)
    suites=ET.parse(root/'nonlinear-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=15,failures=0,errors=0,skipped=0),checks
    profiles={};trials={};configs={}
    for readout in ('linear','nonlinear_lmhead','nonlinear_residual_lmhead'):
        target=root/(readout+'-profile.json')
        execute([sys.executable,'-m','scripts.connection_nonlinear_lmhead_profile','--lm-head-transform',readout,
            '--schedule-steps','305176','--output',str(target)],root/(readout+'-profile.log'),env)
        profiles[readout]=json.loads(target.read_text())
    for readout in ('nonlinear_lmhead','nonlinear_residual_lmhead'):
        ident=uuid.uuid4().hex[:8];out=root/('resume-'+readout);ckpt=Path('../gate-checkpoints')/readout
        cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--lm-head-transform',readout,'--connection-backend','nvidia',
            '--steps','50','--schedule-steps','305176','--microbatch','16','--attention-backend','sdpa_cudnn',
            '--matched-backbone-init','--no-activation-checkpoint','--no-fused-connection-grad','--seed','42',
            '--data-dir','../data','--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id',ident,'--resume','auto']
        execute(cmd+['--segment-steps','25'],root/(readout+'-segment1.log'),env,85)
        assert json.loads((out/'progress.json').read_text())['step']==25
        execute(cmd,root/(readout+'-segment2.log'),env)
        trial=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
        rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(1,51)) and all(math.isfinite(x['loss']) for x in rows)
        assert trial['complete'] and trial['steps']==50 and trial['training_tokens']==50*32768
        assert trial['run_id']==ident and trial['source']==source and trial['final_dev_ce']<trial['initial_dev_ce']
        assert trial['peak_reserved_bytes']<.9*22000*1024**2
        assert config['model']['lm_head_transform']==readout and config['model']['variant']=='baseline' and config['model']['readout']=='collapse' and config['protocol']['schedule_steps']==305176
        assert config['logging_backend']=='local' and config['applied_sinkhorn_iterations']==0
        trials[readout]=trial;configs[readout]=config
    assert len({p['gpu'] for p in profiles.values()})==1
    assert not list(root.rglob('*.wandb')) and not (root/'wandb').exists()
    result=dict(complete=True,logging_backend='local',source=source,checks=checks,profiles=profiles,
        trials=trials,configs=configs,unique_training_steps_per_variant=50,
        restart='Each head exits85 at25 and restores in a fresh process to50, with full10B LR horizon',
        scope='Vanilla full-attention baseline with new head features only; no mHC, no alpha; gate weights stay in worker scratch')
    (root/'nonlinear-lmhead-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,checks=checks,step_ms={k:1000*v['mean_seconds'] for k,v in profiles.items()})),flush=True)


if __name__=='__main__':main()
