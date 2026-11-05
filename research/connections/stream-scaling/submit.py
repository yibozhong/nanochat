"""Submit one bounded GPU implementation check; no formal training."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
ASSETS=Path('/home/yzhong239/nanochat-runtime-assets')


def main(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'state.json').exists() and not (root/'STOPPED.json').exists()
        assert not (root/'submission-intent.json').exists(), 'Recover existing submission intent'
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
        for name in ('bundles','logs','results'):(root/name).mkdir(exist_ok=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        source=dict(commit=commit,formal_training=False,purpose='Frozen-weight mHC vs stream MLP speed scaling n4/8/16/32; no parameter updates',attention_backend='hardware_auto',gpu_family='Hopper_or_A100')
        (root/'bundles/source.json').write_text(json.dumps(source,indent=2)+'\n')
        subprocess.run(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(root/'bundles/source.tar'),'HEAD'],cwd=REPO,check=True)
        for name in ('gate-data.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (root/'bundles'/name).symlink_to(ASSETS/'bundles'/name)
        hashes=json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        for name in ('source.tar','source.json'):
            hashes[name]=hashlib.sha256((root/'bundles'/name).read_bytes()).hexdigest()
        (root/'bundles/inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/stream-scaling/run.sh',root/'run.sh')
        template=(ASSETS/'gate.sub').read_text().replace('executable = validate.sh','executable = run.sh')
        template=template.replace('gpus_minimum_capability = 9.0','gpus_minimum_capability = 8.0')
        template=template.replace('(Capability == 9.0)', '((Capability == 9.0) || ((Capability == 8.0) && regexp("A100",DeviceName)))')
        template=template.replace('22000M','40000M').replace('GlobalMemoryMb >= 22000','GlobalMemoryMb >= 40000')
        template=template.replace('stream-$(Cluster)','gate-$(Cluster)').replace('logs/stream.','logs/gate.')
        template=template.replace('nanochat-depth-diagnostic','nanochat-stream-scaling-gate').replace('DepthDiagnosticKey','StreamScalingKey').replace('depth-diagnostic-20261103',root.name)
        (root/'gate.sub').write_text(template)
        subprocess.run(['condor_submit','-dry-run',str(root/'dry-run.ads'),'gate.sub'],cwd=root,check=True,capture_output=True)
        (root/'submission-intent.json').write_text(json.dumps(source,indent=2)+'\n')
        result=subprocess.check_output(['condor_submit','-terse','gate.sub'],cwd=root,text=True).strip()
        job=result.split()[0]
        state=dict(status='submitted',job=job,source=source,hashes=hashes,formal_training=False)
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2))


if __name__=='__main__':main(sys.argv[1])
