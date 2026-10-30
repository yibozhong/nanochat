"""Full NVIDIA mHC correctness gate and five same-GPU training-step profiles."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

import wandb

GROUP='mhc-nvidia-full-20261030'
TRIALS={
    'baseline':('baseline','deepseek','official',False,'0'),
    'mhc_deepseek':('mhc','deepseek','official',False,'0'),
    'mhc_nvidia_projection_only':('mhc','deepseek','nvidia',False,None),
    'mhc_nvidia_full_unfused':('mhc','nvidia','official',False,None),
    'mhc_nvidia_full_fused':('mhc','nvidia','official',True,None),
}


def environment(override):
    env=os.environ.copy()
    if override is None:env.pop('NVIDIA_TF32_OVERRIDE',None)
    else:env['NVIDIA_TF32_OVERRIDE']=override
    return env


def main():
    out=Path('results');source=json.loads(Path('source.json').read_text())
    ident=uuid.uuid4().hex[:8]
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=GROUP,
        name='full-nvidia-mhc-gate-and-comparison',job_type='correctness-and-profiling',
        id=ident,config=dict(source=source,trials=TRIALS,timed_steps=60,
            scope='all five NVIDIA mHC forward/backward operations, head and persistent interleaved layout'))
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    try:
        suites=[]
        gates=[('controlled',['tests/test_mhc.py','tests/test_mhc_torch_projection.py',
                             'tests/test_mhc_nvidia_projection.py','tests/test_mhc_nvidia_full.py'],'0'),
               ('native',['tests/test_mhc_nvidia_full.py','-k','pipeline or model_no'],None)]
        for label,tests,override in gates:
            print('Checks '+label,flush=True)
            with (out/f'pytest-{label}.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pytest',*tests,'-q',f'--junitxml=results/tests-{label}.xml'],
                    env=environment(override),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            suites+=ET.parse(out/f'tests-{label}.xml').getroot().findall('testsuite')
            shutil.copytree(out/'nvidia-full-checks',out/f'oracle-{label}')
        checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert checks==dict(tests=41,failures=0,errors=0,skipped=0),checks
        print('Checks passed '+json.dumps(checks),flush=True)
        manifest=json.loads(Path('../data/manifest.json').read_text())
        for name,info in manifest['files'].items():
            digest=hashlib.sha256()
            with Path('../data',name+'.bin').open('rb') as f:
                while chunk:=f.read(8*1024*1024):digest.update(chunk)
            assert digest.hexdigest()==info['sha256']
        trials={}
        for name,(variant,backend,projection,fused,override) in TRIALS.items():
            print('Profiling '+name,flush=True)
            cmd=[sys.executable,'-m','scripts.connection_profile','--variant',variant,
                '--projection-backend',projection,'--connection-backend',backend,
                '--fused-connection-grad' if fused else '--no-fused-connection-grad',
                '--wandb-group',GROUP,'--output-dir',str(out/name),
                '--timed-steps','60','--collect-after-warmup']
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(cmd,env=environment(override),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            v=json.loads((out/name/'summary.json').read_text())
            assert v['complete'] and v['source']==source and v['optimizer_updates']==106
            assert v['config']['nvidia_tf32_override']==override
            trials[name]=v;run.log({name+'/normal_step_ms':v['normal_ms']['mean']})
        result=dict(complete=True,source=source,run_id=ident,checks=checks,
            trials=trials,data_manifest=manifest,pretraining_launched=False)
        (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(complete=True,checks_passed=True,trials=5,total_diagnostic_tokens=5*106*32768))
        run.save('results/*.json',base_path='.',policy='now');run.finish()
        print('Comparison complete',flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
