"""Submit one immutable, read-only diagnostic of the preserved mixer checkpoint."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
TRAIN=Path('/home/yzhong239/mhc-partition-mix-20261031')
KEY='mhc-mixing-diagnostics-20261031'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(value):
    p=ROOT/'state.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(ROOT/'state.json')


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'state.json').exists(),'Inspect existing diagnostic; do not duplicate'
        assert not command(['git','status','--porcelain'],REPO)
        rows=json.loads(command(['condor_q','yzhong239','-constraint',f'MixingDiagnosticKey == "{KEY}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not rows
        assert json.loads((TRAIN/'STOPPED.json').read_text())['status']=='stopped_by_user'
        retained=json.loads((TRAIN/'stop-20261031/retained.json').read_text())
        progress=retained['progress'];assert progress['run_id']=='f643360b'
        archive=Path(retained['archives']['checkpoints.tar']['path'])
        assert archive.stat().st_size==retained['archives']['checkpoints.tar']['bytes']
        with tarfile.open(archive) as tar:
            h=hashlib.sha256();f=tar.extractfile('checkpoints/latest.pt')
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            member_hash=h.hexdigest()
        for n in ('bundles','logs','results'):(ROOT/n).mkdir(exist_ok=True)
        commit=command(['git','rev-parse','HEAD'],REPO)
        source=json.loads((TRAIN/'bundles/source.json').read_text())
        bundle=ROOT/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(TRAIN/'bundles'/wheel,bundle/wheel)
        hashes={k:v for k,v in source['hashes'].items() if k!='source.tar'}
        hashes['source.tar']=sha(bundle/'source.tar')
        source=dict(commit=commit,hashes=hashes,purpose='Read-only final-mixing diagnosis; no training, no W&B')
        shutil.copy2(REPO/'research/connections/mixing-diagnostics/validate.sh',ROOT/'validate.sh')
        source['launcher_sha256']=sha(ROOT/'validate.sh')
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        plan=dict(source=source,checkpoint_archives={archive.name:sha(archive)},cases=[dict(
            label='mix',path='../diagnostic-checkpoints/mix.pt',archive=archive.name,
            run_id='f643360b',expected_step=progress['step'],sha256=member_hash)])
        (ROOT/'diagnostic-inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
        inputs=[bundle/n for n in ('source.tar','source.json','inputs.sha256',wheel)]
        inputs += [ROOT/'diagnostic-inputs.json',archive,
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb',
            'osdf:///chtc/staging/y/yzhong239/nanochat-ebt-20261029/data-7444406.tar']
        submit=f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = {ROOT/'validate.sh'}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {','.join(map(str,inputs))}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/diagnostic-$(Cluster).$(Process).tar"
stream_output = True
stream_error = True
log = logs/diagnostic.$(Cluster).$(Process).log
output = logs/diagnostic.$(Cluster).$(Process).out
error = logs/diagnostic.$(Cluster).$(Process).err
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_disk = 45GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
+MixingDiagnosticKey = "{KEY}"
+JobBatchName = "nanochat-final-mixing-diagnostic"
queue 1
'''
        (ROOT/'diagnostic.sub').write_text(submit)
        command(['condor_submit','-dry-run',str(ROOT/'diagnostic-dryrun.ad'),str(ROOT/'diagnostic.sub')])
        state=dict(status='submitting',source=source,plan=plan,submitted_at=time.time(),reviewed_head=commit)
        save(state)
        ident=command(['condor_submit','-terse',str(ROOT/'diagnostic.sub')]).split()[0]
        assert len(ident.split('.'))==2 and all(x.isdigit() for x in ident.split('.'))
        state.update(status='submitted',job=ident);save(state)
        print(json.dumps(dict(job=ident,checkpoint_step=progress['step'],source=commit)))


if __name__=='__main__':main()
