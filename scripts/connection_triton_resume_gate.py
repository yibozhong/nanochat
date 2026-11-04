"""Three real 25->exit85->50 fresh-process restarts; full 10B LR horizon."""
import json,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.connection_moment_gate import execute
from nanochat.packed_training import json_write

MODES=('none','variance','jensen_silu')
def main():
    start=time.perf_counter()
    execute([sys.executable,'-m','pytest','tests/test_triton_resume.py','-v','--junitxml=results/resume-tests.xml'],'resume-tests')
    suites=ET.parse('results/resume-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert checks==dict(tests=4,failures=0,errors=0,skipped=0)
    trials={}
    for mode in MODES:
        out=Path('results')/('resume-'+mode);ckpt=Path('../gate-checkpoints')/mode
        cmd=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--attention-kernel','fla_triton',
            '--attention-statistic',mode,'--steps','50','--schedule-steps','305176','--microbatch','16',
            '--attention-backend','sdpa_cudnn','--matched-backbone-init','--data-dir','../data',
            '--output-dir',str(out),'--checkpoint-dir',str(ckpt),'--run-id','gate-'+mode,'--resume','auto',
            '--segment-seconds','1800']
        execute(cmd+['--segment-steps','25'],'first-'+mode,85)
        assert json.loads((out/'progress.json').read_text())['step']==25
        execute(cmd,'second-'+mode)
        result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
        assert result['complete'] and result['steps']==50 and result['training_tokens']==1638400
        assert config['logging_backend']=='local' and config['model']['attention_kernel']=='fla_triton'
        assert config['model']['attention_statistic']==mode and config['protocol']['schedule_steps']==305176
        assert [json.loads(x)['step'] for x in (out/'steps.jsonl').read_text().splitlines()]==list(range(1,51))
        trials[mode]=dict(summary=result,config=config,exact_steps_1_to_50=True)
    assert not list(Path('results').rglob('*.wandb'))
    json_write('results/triton-resume-gate.json',dict(complete=True,source=json.loads(Path('source.json').read_text()),
        checks=checks,trials=trials,formal_training=False,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
