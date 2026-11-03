"""Bounded SiLU gap validation only; no formal training or W&B."""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

from scripts.connection_moment_gate import execute
from nanochat.packed_training import json_write


def main():
    started=time.perf_counter()
    execute([sys.executable,'-m','pytest','tests/test_jensen_attention.py','tests/test_moment_attention.py',
        'tests/test_connection_training_resume.py','-v','--junitxml=results/jensen-tests.xml'],'tests')
    suites=ET.parse('results/jensen-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=25,failures=0,errors=0,skipped=0),checks
    profiles=[]
    for mode in ('none','variance','jensen_silu'):
        name='profile-'+mode
        execute([sys.executable,'-m','scripts.connection_moment_gate','--profile',mode,
            '--output','results/'+name+'.json'],name)
        profiles.append(json.loads(Path('results',name+'.json').read_text()))
    out=Path('results/resume-jensen_silu')
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline',
        '--attention-statistic','jensen_silu','--steps','50','--schedule-steps','305176',
        '--microbatch','16','--attention-backend','sdpa_cudnn','--matched-backbone-init',
        '--data-dir','../data','--output-dir',str(out),'--checkpoint-dir','../gate-checkpoints/jensen_silu',
        '--run-id','gate-jensen-silu','--resume','auto','--segment-seconds','1800']
    execute(cmd+['--segment-steps','25'],'resume1',85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,'resume2')
    result=json.loads((out/'summary.json').read_text())
    assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
    assert [json.loads(s)['step'] for s in (out/'steps.jsonl').read_text().splitlines()]==list(range(1,51))
    config=json.loads((out/'config.json').read_text())
    assert config['model']['attention_statistic']=='jensen_silu' and config['logging_backend']=='local'
    assert config['protocol']['schedule_steps']==305176
    assert not list(Path('results').rglob('*.wandb'))
    json_write('results/jensen-gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,restart=dict(summary=result,config=config,exact_steps_1_to_50=True),
        seconds=time.perf_counter()-started,formal_training=False,weights_returned=False))


if __name__=='__main__':main()
