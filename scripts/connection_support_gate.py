"""Effective-support gated variance fused-kernel correctness, matched throughput and restart."""
import json,os,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.connection_moment_gate import execute
from scripts.connection_triton_moment_gate import operator_profile
from nanochat.packed_training import json_write


def main():
    started=time.perf_counter()
    assert not os.environ.get('WANDB_API_KEY')
    execute([sys.executable,'-m','pytest','tests/test_triton_moments.py','tests/test_support_attention.py',
        '-v','--junitxml=results/support-tests.xml'],'support-tests')
    suites=ET.parse('results/support-tests.xml').getroot().findall('testsuite')
    checks={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=33,failures=0,errors=0,skipped=0),checks
    operator_profile(('none','variance','variance_support'))
    profiles=[]
    for repeat,order in enumerate([('none','variance','variance_support'),('variance_support','variance','none')]):
        for mode in order:
            name=f'profile-{repeat}-{mode}'
            execute([sys.executable,'-m','scripts.connection_moment_gate','--profile',mode,'--kernel','fla_triton',
                '--output',f'results/{name}.json'],name)
            profiles.append(json.loads(Path('results',name+'.json').read_text()))
    out=Path('results/resume-variance_support');ckpt=Path('../gate-checkpoints/variance_support')
    cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--attention-kernel','fla_triton',
        '--attention-statistic','variance_support','--steps','50','--schedule-steps','305176','--microbatch','16',
        '--attention-backend','sdpa_cudnn','--matched-backbone-init','--data-dir','../data',
        '--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id','gate-variance_support','--resume','auto',
        '--segment-seconds','1800']
    execute(cmd+['--segment-steps','25'],'resume1',85)
    assert json.loads((out/'progress.json').read_text())['step']==25
    execute(cmd,'resume2')
    result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
    assert result['complete'] and result['steps']==50 and result['training_tokens']==1638400
    assert config['logging_backend']=='local' and config['model']['attention_kernel']=='fla_triton'
    assert config['model']['attention_statistic']=='variance_support' and config['protocol']['schedule_steps']==305176
    assert [json.loads(x)['step'] for x in (out/'steps.jsonl').read_text().splitlines()]==list(range(1,51))
    assert not list(Path('results').rglob('*.wandb'))
    json_write('results/support-gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,profiles=profiles,restart=dict(summary=result,config=config,exact_steps_1_to_50=True),
        seconds=time.perf_counter()-started,formal_training=False,weights_returned=False))


if __name__=='__main__':main()
