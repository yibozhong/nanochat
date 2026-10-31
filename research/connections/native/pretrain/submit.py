"""Submit exactly the user's three native10B runs after the completed GPU gate."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[4]
KEY='mhc-native-pretrain-20261031'
MODEL_COMMIT='1855eb476269c513dc6471c862cd4ee7e2f9d526'
VARIANTS=('baseline','mhc','mhc_no_sinkhorn')
STEPS=305176
DATA='osdf:///chtc/staging/y/yzhong239/nanochat-mhc-10b-data-7560536.tar'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(state):
    p=ROOT/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'state.json')


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not command(['git','status','--porcelain'],REPO)
        assert not (ROOT/'submission-intent.json').exists(),'Recover existing intent; do not resubmit'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'NativePretrainKey == "{KEY}"',
                                  '-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not active,'Recover existing marked jobs; do not duplicate'
        gate=json.loads((REPO/'research/connections/native/validation.json').read_text())
        assert gate['complete'] and gate['job']=='7636681.0' and gate['source']['commit']==MODEL_COMMIT
        assert gate['checks']==dict(tests=11,failures=0,errors=0,skipped=0)
        for v in VARIANTS:
            assert gate['trials'][v]['summary']['complete'] and gate['trials'][v]['summary']['steps']==10
        prior=Path('/home/yzhong239/mhc-pretrain-20261030')
        data_state=json.loads((prior/'state.json').read_text())
        assert data_state['data_ready'] and data_state['data_uri']==DATA
        assert data_state['data_training_tokens']==STEPS*32768
        runtime=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert runtime['complete'] and runtime['actual_eviction']
        for folder in ('bundles','logs','results'):(ROOT/folder).mkdir(exist_ok=True)
        tooling_commit=command(['git','rev-parse','HEAD'],REPO)
        # Production training code and dependencies are exactly the GPU-tested source.
        changed=command(['git','diff','--name-only',MODEL_COMMIT,tooling_commit,'--','nanochat','scripts',
                         'research/connections/runtime','research/connections/toolchain'],REPO)
        assert not changed,changed
        paths=[prior/'train.sh',prior/'train.sub',*[prior/'training-bundles'/n for n in ('source.tar','source.json','inputs.sha256')]]
        other=Path('/home/yzhong239/mhc-no-sinkhorn-20261031')
        paths += [other/'train.sh',other/'train.sub',*[other/'bundles'/n for n in ('source.tar','source.json','inputs.sha256')]]
        immutable={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        (ROOT/'immutable-previous-jobs.json').write_text(json.dumps(immutable,indent=2)+'\n')
        run_ids={v:uuid.uuid4().hex[:8] for v in VARIANTS}
        (ROOT/'submission-intent.json').write_text(json.dumps(dict(model_commit=MODEL_COMMIT,
            tooling_commit=tooling_commit,run_ids=run_ids,steps=STEPS,key=KEY),indent=2)+'\n')
        bundle=ROOT/'bundles'
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(prior/'training-bundles'/wheel,bundle/wheel)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',MODEL_COMMIT],REPO)
        shutil.copy2(REPO/'research/connections/native/pretrain/train.sh',ROOT/'train.sh')
        hashes={k:v for k,v in gate['source']['hashes'].items() if k in (wheel,'mhc-official-runtime-7546767.tar.gz','cuda-compat-13-1_590.48.01-1_amd64.deb')}
        assert len(hashes)==3 and hashlib.sha256((bundle/wheel).read_bytes()).hexdigest()==hashes[wheel]
        hashes['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        assert hashes['source.tar']==gate['source']['hashes']['source.tar']
        data_name=DATA.rsplit('/',1)[-1]
        hashes[data_name]=data_state['data_archive']['sha256']
        source=dict(commit=MODEL_COMMIT,tooling_commit=tooling_commit,
            purpose='User-authorized native nanochat baseline/mHC/no-Sinkhorn10B formal training',
            gate_job=gate['job'],gate_verified=True,hashes=hashes,architecture='native',variants=VARIANTS,
            steps=STEPS,training_tokens=STEPS*32768,data_uri=DATA,checkpoint_upload_seconds=1800,
            local_save_every_steps=500,backfill_opt_in=True,wandb_group=KEY,
            launcher_sha256=hashlib.sha256((ROOT/'train.sh').read_bytes()).hexdigest())
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        (bundle/'inputs.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()))
        submit=f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = train.sh
arguments = $(variant) $(runid) {STEPS}
environment = "NANOCHAT_DATA_ARCHIVE={data_name} NANOCHAT_CHECKPOINT_SECONDS=1800"
encrypt_input_files = /home/yzhong239/.netrc
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_disk = 80GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
transfer_input_files = bundles/{wheel},bundles/source.tar,bundles/source.json,bundles/inputs.sha256,/home/yzhong239/.netrc,{DATA},osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz,https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/$(variant)-$(Cluster).$(Process).tar;checkpoints.tar=results/$(variant)-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+JobBatchName = "nanochat-native-three-way-10b"
+NativePretrainKey = "{KEY}"
+MHCVariant = "$(variant)"
+MHCRunID = "$(runid)"
queue variant,runid from (
'''+''.join(f'{v} {run_ids[v]}\n' for v in VARIANTS)+')\n'
        (ROOT/'train.sub').write_text(submit)
        command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'])
        # Persist all identities before the single queue3 mutation.
        state=dict(status='submitting',source=source,expected_steps=STEPS,expected_tokens=STEPS*32768,
            run_ids=run_ids,gate_verified=True,submitted_at=time.time(),formal_training=True)
        save(state)
        output=command(['condor_submit','-terse','train.sub'])
        cluster=output.split('.')[0];assert cluster.isdigit(),output
        state['jobs']={v:dict(job=f'{cluster}.{i}',run_id=run_ids[v]) for i,v in enumerate(VARIANTS)}
        state['status']='training_submitted';save(state)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in immutable.items())
        print(json.dumps(state['jobs'],indent=2))


if __name__=='__main__':main()
