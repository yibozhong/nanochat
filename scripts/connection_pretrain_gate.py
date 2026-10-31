"""Validate the new checkpoint writer and actual fresh-process training resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
import xml.etree.ElementTree as ET

from nanochat.experiment_logging import diagnostic_run

GROUP='mhc-pretrain-20261030'


def execute(command,path,env,expected=0):
    with path.open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,bufsize=1,env=env)
        def stream():
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
        reader=threading.Thread(target=stream,daemon=True);reader.start()
        try:code=proc.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait();reader.join();raise
        reader.join()
        assert code==expected,(command,code,expected)


def main():
    root=Path('results');source=json.loads(Path('source.json').read_text())
    ident=uuid.uuid4().hex[:8]
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=GROUP,id=ident,
        name='formal-pretraining-resume-gate',job_type='correctness',config=dict(source=source))
    run=diagnostic_run(**kwargs)
    try:
        env=os.environ.copy();env.pop('NVIDIA_TF32_OVERRIDE',None)
        execute([sys.executable,'-m','pytest','tests/test_connection_training_resume.py','-v',
                 '--junitxml=results/training-resume-tests.xml'],root/'training-resume-tests.log',env)
        suites=ET.parse(root/'training-resume-tests.xml').getroot().findall('testsuite')
        checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert checks==dict(tests=3,failures=0,errors=0,skipped=0),checks
        trials={}
        for variant in ('baseline','mhc'):
            ident_child=uuid.uuid4().hex[:8]
            if variant=='baseline':env['NVIDIA_TF32_OVERRIDE']='0'
            else:env.pop('NVIDIA_TF32_OVERRIDE',None)
            out=root/('resume-'+variant)
            checkpoints=Path('../gate-checkpoints')/variant
            cmd=[sys.executable,'-m','scripts.connection_train','--variant',variant,
                '--connection-backend','nvidia',
                '--steps','50','--schedule-steps','30518','--microbatch','16',
                '--attention-backend','sdpa_cudnn','--matched-backbone-init',
                '--data-dir','../data','--output-dir',str(out),'--checkpoint-dir',str(checkpoints),
                '--run-id',ident_child,'--wandb-group',GROUP,'--resume','auto']
            execute(cmd+['--segment-steps','25'],root/(variant+'-segment1.log'),env,85)
            progress=json.loads((out/'progress.json').read_text())
            assert progress['step']==25 and progress['run_id']==ident_child
            execute(cmd,root/(variant+'-segment2.log'),env)
            result=json.loads((out/'summary.json').read_text())
            assert result['complete'] and result['steps']==50 and result['run_id']==ident_child
            assert result['source']==source and result['training_tokens']==50*32768
            rows=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows]==list(range(1,51))
            assert (checkpoints/'latest.pt').exists() and (checkpoints/'best.pt').exists()
            trials[variant]=result
        result=dict(complete=True,source=source,run_id=run.id,checks=checks,trials=trials,
            checkpoint_io='Real process exit85 at step25, second process restores and completes step50',
            optimizer_validation='Exact round-trip model/optimizer/RNG and fixed-gradient next update; not a claim of bitwise identical independent backward trajectories')
        (root/'pretrain-gate.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(complete=True,checks_passed=True,restored_variants=2))
        run.save('results/*.json',base_path='.',policy='now');run.finish()
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
