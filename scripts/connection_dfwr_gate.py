"""DFWR mathematics, GPU integration, memory, and fresh-process resume gate."""
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
    execute([sys.executable,'-m','pytest','tests/test_dfwr.py','-v','--junitxml=results/dfwr-tests.xml'],root/'tests.log',env)
    suites=ET.parse(root/'dfwr-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=11,failures=0,errors=0,skipped=0),checks
    profiles={}
    for mode,micro in [('baseline',16),('dfwr',1)]:
        output=root/(mode+'-profile.json')
        execute([sys.executable,'-m','scripts.connection_dfwr_profile','--variant',mode,
            '--microbatch',str(micro),'--output',str(output)],root/(mode+'-profile.log'),env)
        profiles[mode]=json.loads(output.read_text())
    ident=uuid.uuid4().hex[:8];out=root/'resume-dfwr';ckpt=Path('../gate-checkpoints')/'dfwr'
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','dfwr','--fw-bottleneck','8',
        '--steps','50','--schedule-steps','305176','--microbatch','1','--attention-backend','sdpa_cudnn',
        '--activation-checkpoint','--seed','42','--data-dir','../data','--output-dir',str(out),
        '--checkpoint-dir',str(ckpt),'--run-id',ident,'--resume','auto']
    execute(cmd+['--segment-steps','25'],root/'segment1.log',env,85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,root/'segment2.log',env)
    trial=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
    assert trial['complete'] and trial['steps']==50 and trial['training_tokens']==50*32768
    assert trial['source']==source and trial['run_id']==ident and trial['final_dev_ce']<trial['initial_dev_ce']
    assert trial['peak_reserved_bytes']<.9*22000*1024**2
    assert config['model']['variant']=='dfwr' and config['model']['fw_bottleneck']==8
    assert config['protocol']['schedule_steps']==305176 and config['logging_backend']=='local'
    rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,51)) and all(math.isfinite(r['loss']) for r in rows)
    assert not list(root.rglob('*.wandb'))
    result=dict(complete=True,source=source,checks=checks,profiles=profiles,trial=trial,config=config,
        logging_backend='local',formal_ready=True,selected_microbatch=1,selected_bottleneck=8,
        scope='Per-token depth-only FW, all16 sublayers, exact differentiable analytic inner gradients, no residual addition; frozen-anchor derivative oracle')
    (root/'dfwr-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,checks=checks)),flush=True)


if __name__=='__main__':main()
