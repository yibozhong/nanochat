"""Immutable diagnostic submissions; two checkpoints, no training or W&B."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(sys.argv[1]).resolve()
PHASE=sys.argv[2] if len(sys.argv)>2 else 'interim'
REPO=Path(__file__).resolve().parents[3]
PRETRAIN=Path('/home/yzhong239/mhc-pretrain-20261030')
PARTITION=Path('/home/yzhong239/mhc-vocab-partition-20261031')
KEY='mhc-rank-diagnostics-20261031'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()


def save(state):
    p=ROOT/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'state.json')


def main():
    if (ROOT/'STOPPED.json').exists():
        print('Workflow stopped by user; no monitoring or submissions')
        return
    assert PHASE in ('interim','final')
    ROOT.mkdir(exist_ok=True)
    for n in ('bundles','logs','results'):(ROOT/n).mkdir(exist_ok=True)
    with (ROOT/'state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((ROOT/'state.json').read_text()) if (ROOT/'state.json').exists() else dict(phases={},status='active')
        if PHASE in state['phases']:
            print(json.dumps(state['phases'][PHASE]));return
        existing=json.loads(command(['condor_q','yzhong239','-constraint',f'RankDiagnosticKey == "{KEY}" && RankDiagnosticPhase == "{PHASE}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not existing,'Recover existing marked job instead of submitting a duplicate'
        prior=json.loads((PRETRAIN/'results.json').read_text())
        if PHASE=='interim':
            meta=json.loads((ROOT/'snapshot.json').read_text())
            partition_archive=ROOT/'inputs/partition-snapshot.tar'
            pstep=meta['progress_at_copy']['step'];phash=meta['latest_member_sha256']
        else:
            final=json.loads((PARTITION/'results.json').read_text())
            assert final['complete'] and final['steps']==305176 and final['wandb_verified']
            partition_archive=Path(final['checkpoint_archive']);pstep=305176;phash=final['checkpoint_files']['latest.pt']['sha256']
        if 'source' not in state:
            assert not command(['git','status','--porcelain'],REPO)
            commit=command(['git','rev-parse','HEAD'],REPO)
            command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={ROOT/"bundles/source.tar"}',commit],REPO)
            base=json.loads((PARTITION/'bundles/source.json').read_text())
            hashes={k:v for k,v in base['hashes'].items() if k!='source.tar'}
            hashes['source.tar']=sha(ROOT/'bundles/source.tar')
            wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
            shutil.copy2(PARTITION/'bundles'/wheel,ROOT/'bundles'/wheel)
            shutil.copy2(REPO/'research/connections/rank-diagnostics/validate.sh',ROOT/'validate.sh')
            source=dict(commit=commit,hashes=hashes,launcher_sha256=sha(ROOT/'validate.sh'),purpose='Read-only spectrum, hidden-gradient and frozen-head diagnostics')
            (ROOT/'bundles/source.json').write_text(json.dumps(source,indent=2)+'\n')
            (ROOT/'bundles/inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
            state.update(source=source,reviewed_head=commit,created_at=time.time())
        else:
            assert sha(ROOT/'bundles/source.tar')==state['source']['hashes']['source.tar']
            assert sha(ROOT/'validate.sh')==state['source']['launcher_sha256']
        phase_dir=ROOT/PHASE;phase_dir.mkdir(exist_ok=True)
        reference=PRETRAIN/'results/mhc-7562105.1-checkpoints.tar'
        # Unique basenames prevent collisions in Condor's flat input sandbox.
        linked=phase_dir/'partition-snapshot.tar'
        if not linked.exists():linked.symlink_to(partition_archive)
        plan=dict(phase=PHASE,source=state['source'],checkpoint_archives={
            'partition-snapshot.tar':sha(partition_archive),reference.name:sha(reference)},cases=[
            dict(label='partition',path='../diagnostic-checkpoints/partition.pt',archive='partition-snapshot.tar',
                 run_id='a7ef0469',expected_step=pstep,readout='vocab_partition',sha256=phash),
            dict(label='mhc',path='../diagnostic-checkpoints/mhc.pt',archive=reference.name,
                 run_id='048c708c',expected_step=305176,readout='collapse',sha256=prior['trials']['mhc']['checkpoint_files']['latest.pt']['sha256'])])
        (phase_dir/'diagnostic-inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
        inputs=[ROOT/'bundles'/n for n in ('source.tar','source.json','inputs.sha256','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl')]
        inputs+= [phase_dir/'diagnostic-inputs.json',linked,reference,
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
transfer_output_remaps = "results.tar=results/{PHASE}-$(Cluster).$(Process).tar"
stream_output = True
stream_error = True
log = logs/{PHASE}.$(Cluster).$(Process).log
output = logs/{PHASE}.$(Cluster).$(Process).out
error = logs/{PHASE}.$(Cluster).$(Process).err
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
+RankDiagnosticKey = "{KEY}"
+RankDiagnosticPhase = "{PHASE}"
+JobBatchName = "nanochat-readout-rank-diagnostic-{PHASE}"
queue 1
'''
        sub=ROOT/(PHASE+'.sub');sub.write_text(submit)
        command(['condor_submit','-dry-run',str(ROOT/(PHASE+'-dryrun.ad')),str(sub)])
        state['phases'][PHASE]=dict(status='submitting',plan=plan,submitted_at=time.time());save(state)
        ident=command(['condor_submit','-terse',str(sub)]).split()[0]
        assert len(ident.split('.'))==2 and all(x.isdigit() for x in ident.split('.'))
        state['phases'][PHASE].update(job=ident,status='submitted');save(state)
        print(json.dumps(dict(phase=PHASE,job=ident,partition_step=pstep,reference_step=305176)))


if __name__=='__main__':main()
