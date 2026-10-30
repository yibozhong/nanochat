"""Same-GPU profiling of the three connection variants."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

import wandb


def main():
    out=Path('results');source=json.loads(Path('source.json').read_text())
    run_id=uuid.uuid4().hex[:8]
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group='mhc-profiling-20261030',
                name='same-gpu-profile',job_type='profiling-audit',id=run_id,config=dict(source=source))
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    try:
        manifest=json.loads(Path('../data/manifest.json').read_text())
        for name,info in manifest['files'].items():
            digest=hashlib.sha256()
            with Path('../data',name+'.bin').open('rb') as f:
                while chunk:=f.read(8*1024*1024):digest.update(chunk)
            assert digest.hexdigest()==info['sha256'],name
        variants={}
        for name in ('baseline','mhc','mhc_no_manifold'):
            print('Profiling variant '+name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-m','scripts.connection_profile','--variant',name,
                    '--output-dir',str(out/name)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            result=json.loads((out/name/'summary.json').read_text())
            assert result['complete'] and result['source']==source and result['optimizer_updates']==66
            variants[name]=result
            run.log({name+'/normal_step_ms':result['normal_ms']['mean']})
        result=dict(complete=True,source=source,run_id=run_id,variants=variants,
                    data_manifest=manifest,pretraining_launched=False)
        (out/'profile.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(complete=True,variants=3,total_diagnostic_tokens=3*66*32768))
        run.save('results/profile.json',base_path='.',policy='now');run.finish()
        print('Profiling complete',flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
