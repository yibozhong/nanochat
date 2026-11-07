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
        source=dict(commit=commit,formal_training=False,purpose='Channelwise legacy-arithmetic tiling/fused backward optimization; fixed formula, H200 full-step target90ms; no formal run')
        (root/'bundles/source.json').write_text(json.dumps(source,indent=2)+'\n')
        subprocess.run(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(root/'bundles/source.tar'),'HEAD'],cwd=REPO,check=True)
        for name in ('gate-data.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (root/'bundles'/name).symlink_to(ASSETS/'bundles'/name)
        hashes=json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        for name in ('source.tar','source.json'):
            hashes[name]=hashlib.sha256((root/'bundles'/name).read_bytes()).hexdigest()
        (root/'bundles/inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/channel-opt/run.sh',root/'run.sh')
        template=(ASSETS/'gate.sub').read_text().replace('executable = validate.sh','executable = run.sh')
        template=template.replace('stream-$(Cluster)','gate-$(Cluster)').replace('logs/stream.','logs/gate.')
        template=template.replace('nanochat-depth-diagnostic','nanochat-channel-opt-gate').replace('DepthDiagnosticKey','ChannelOptGateKey').replace('depth-diagnostic-20261103',root.name)
        template=template.replace('(Capability == 9.0)', '(Capability == 9.0) && regexp("H200",DeviceName)')
        template=template.replace('!regexp("MIG",DeviceName)', '!regexp("MIG",DeviceName) && (DeviceUuid =!= "7067f38b-b4c2-5568-ec82-f698473ddd07")')
        template=template.replace('queue 1','hold = true\nqueue 1')
        (root/'gate.sub').write_text(template)
        subprocess.run(['condor_submit','-dry-run',str(root/'dry-run.ads'),'gate.sub'],cwd=root,check=True,capture_output=True)
        (root/'submission-intent.json').write_text(json.dumps(source,indent=2)+'\n')
        result=subprocess.check_output(['condor_submit','-terse','gate.sub'],cwd=root,text=True).strip()
        job=result.split()[0]
        state=dict(status='submitted',job=job,source=source,hashes=hashes,formal_training=False)
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        ad=json.loads(subprocess.check_output(['condor_q',job,'-json'],text=True))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestMemory']==16384
        assert '.netrc' not in ad['TransferInput'] and ad['TransferOutput']=='results.tar'
        assert ad['ChannelOptGateKey']==root.name and 'H200' in ad['RequireGPUs'] and ad['GPUsMinCapability']==9.0
        (root/'submission-verified.json').write_text(json.dumps(ad,indent=2)+'\n')
        subprocess.run(['condor_release',job],check=True,capture_output=True)
        print(json.dumps(state,indent=2))


if __name__=='__main__':main(sys.argv[1])
