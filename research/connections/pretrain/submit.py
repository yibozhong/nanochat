"""Submit the two authorized runs only after budget/data/GPU gates are recorded."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    with (ROOT/'submit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state.get('training_jobs'):
            print(json.dumps(state['training_jobs']));return
        assert state['gate_verified'] and state['scale_verified']
        assert state['wandb_cleanup_complete'] and not state.get('automation_paused_for_wandb_cleanup')
        assert state.get('budget_confirmed_by_user') is True
        assert state['requested_tokens'] in (1_000_000_000,10_000_000_000)
        assert state.get('data_ready') and state['data_uri']
        existing=json.loads(command(['condor_q','yzhong239','-constraint',
            'MHCPretrainKey == "mhc-pretrain-20261030"','-json']) or '[]')
        assert not existing,'Marked jobs already exist; recover their IDs instead of duplicating'
        core=['nanochat/connection_model.py','nanochat/manifold_connections.py','nanochat/nvidia_connections.py',
              'nanochat/nvidia_projection.py','nanochat/connection_experiment.py','nanochat/packed_training.py',
              'nanochat/gpt.py','nanochat/optim.py','nanochat/flash_attention.py','scripts/connection_train.py']
        changed=set(command(['git','diff','--name-only',state['gate_source']['commit'],'HEAD','--',*core],REPO).splitlines())
        assert changed <= {'scripts/connection_train.py'}
        audit=json.loads((REPO/'research/connections/pretrain/logging-audit.json').read_text())
        assert audit['complete'] and audit['gate_commit']==state['gate_source']['commit']
        for name,key in [('scripts/connection_train.py','training_script_sha256'),('nanochat/experiment_logging.py','logging_helper_sha256')]:
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==audit[key]
        assert not command(['git','status','--porcelain','--',*core],REPO)
        steps=math.ceil(state['requested_tokens']/32768)
        assert state['data_training_tokens']>=steps*32768
        intent=dict(run_ids={v:uuid.uuid4().hex[:8] for v in ('baseline','mhc')},
                    steps=steps,requested_tokens=state['requested_tokens'])
        (ROOT/'submission-intent.json').write_text(json.dumps(intent,indent=2)+'\n')
        bundle=ROOT/'training-bundles';bundle.mkdir(exist_ok=True)
        # The executable lives outside the source tar; refresh it from this commit.
        shutil.copy2(REPO/'research/connections/pretrain/train.sh',ROOT/'train.sh')
        for name in ('legacy-baseline.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            shutil.copy2(ROOT/'bundles'/name,bundle/name)
        commit=command(['git','rev-parse','HEAD'],REPO)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
        source=dict(state['gate_source'])
        source.update(commit=commit,purpose='Authorized matched vanilla/full-NVIDIA mHC pretraining',
            gate_commit=state['gate_source']['commit'],requested_tokens=state['requested_tokens'],
            steps=steps,training_tokens=steps*32768,data_uri=state['data_uri'],
            launcher_sha256=hashlib.sha256((ROOT/'train.sh').read_bytes()).hexdigest())
        source['hashes']=dict(source['hashes'])
        source['hashes']['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        (bundle/'inputs.sha256').write_text(''.join(f'{digest}  {name}\n' for name,digest in source['hashes'].items()))
        data_name=state['data_uri'].rsplit('/',1)[-1]
        disk_gb=80 if state['requested_tokens']==10_000_000_000 else 40
        submit=f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = train.sh
arguments = $(variant) $(runid) {steps}
environment = "NANOCHAT_DATA_ARCHIVE={data_name}"
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
request_disk = {disk_gb}GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
transfer_input_files = training-bundles/transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl,training-bundles/source.tar,training-bundles/legacy-baseline.tar,training-bundles/source.json,training-bundles/inputs.sha256,/home/yzhong239/.netrc,{state['data_uri']},osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz,https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/$(variant)-$(Cluster).$(Process).tar;checkpoints.tar=results/$(variant)-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+JobBatchName = "nanochat-vanilla-nvidia-mhc-pretrain"
+MHCPretrainKey = "mhc-pretrain-20261030"
+MHCVariant = "$(variant)"
+MHCRunID = "$(runid)"
queue variant,runid from (
baseline {intent['run_ids']['baseline']}
mhc {intent['run_ids']['mhc']}
)
'''
        (ROOT/'train.sub').write_text(submit)
        command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'])
        output=command(['condor_submit','-terse','train.sub']);cluster=output.split('.')[0]
        assert cluster.isdigit(),output
        state.update(training_source=source,status='training_submitted',training_steps=steps,
            training_jobs={v:dict(job=f'{cluster}.{i}',run_id=intent['run_ids'][v])
                           for i,v in enumerate(('baseline','mhc'))})
        path=ROOT/'state.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(ROOT/'state.json')
        print(json.dumps(state['training_jobs'],indent=2))


if __name__=='__main__':main()
