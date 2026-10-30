"""Compute-node correctness and two 50-step integration smokes; no promotion."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

import wandb


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',default='../data')
    parser.add_argument('--work-dir',default='../work')
    a=parser.parse_args()
    out=Path('results');out.mkdir(exist_ok=True)
    work=Path(a.work_dir);work.mkdir(exist_ok=True)
    run_id=uuid.uuid4().hex[:8]
    source=json.loads(Path('source.json').read_text())
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group='mhc-official-kernels-20261030',
                job_type='correctness-gate',name='mhc-official-kernel-integration',id=run_id,
                config=dict(source=source,microbatch=16,batch_tokens=32768,smoke_steps_per_variant=50))
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    started=time.time()
    try:
        print('Starting official-kernel and integration checks', flush=True)
        with (out/'pytest.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','tests/test_mhc.py',
                'tests/test_attention_fallback.py','-q','--junitxml=results/tests.xml'],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        suites=ET.parse(out/'tests.xml').getroot().findall('testsuite')
        checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
        assert checks['failures']==checks['errors']==0
        print('Checks passed: '+json.dumps(checks), flush=True)
        run.log(checks)
        trials={}
        for variant in ('baseline','mhc'):
            print('Starting 50-step smoke: '+variant, flush=True)
            path=work/variant
            with (out/(variant+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-m','scripts.connection_train','--variant',variant,
                    '--steps','50','--data-dir',a.data_dir,'--output-dir',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            summary=json.loads((path/'summary.json').read_text())
            assert summary['complete'] and summary['steps']==50 and summary['training_tokens']==1638400
            assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']-.02
            dest=out/variant;dest.mkdir(exist_ok=True)
            for file in ('summary.json','config.json','steps.jsonl'):shutil.copy2(path/file,dest/file)
            trials[variant]=summary
            run.log({variant+'/'+k:v for k,v in summary.items() if isinstance(v,(int,float))})
        result=dict(passed=True,source=source,run_id=run_id,checks=checks,smokes=trials,
                    elapsed_seconds=time.time()-started,pretraining_launched=False)
        (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(passed=True,checks_passed=True,smoke_variants=2,
                                total_smoke_tokens=2*1638400))
        run.save('results/*.json',base_path='.',policy='now');run.finish()
        print(json.dumps(result),flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
