"""Submit the authorized single 10B run from the already validated GPU source."""
import fcntl
import hashlib
import json
import math
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[3]
GATE=Path('/home/yzhong239/state-tokens-triton-gate-20261105-r2')
ASSETS=Path('/home/yzhong239/nanochat-runtime-assets')
KEY='state-tokens-residual-10b-20261105'
STEPS=305176


def command(args,cwd):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():
            print((root/'state.json').read_text());return
        assert not (root/'submission-intent.json').exists(),'Recover the previous submission intent; do not duplicate'
        assert not command(['git','status','--porcelain'],REPO)
        head=command(['git','rev-parse','HEAD'],REPO)
        assert head==command(['git','ls-remote','origin','refs/heads/main'],REPO).split()[0]
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'StateTokensTrainKey == "{KEY}"',
                                  '-json','-attributes','ClusterId,ProcId'],root) or '[]')
        assert not active,'Recover existing marked job instead of duplicating'
        gate=json.loads((REPO/'research/connections/state-tokens/validation.json').read_text())
        gs=json.loads((GATE/'state.json').read_text())
        assert gate['complete'] and gs['status'] in ('published','complete') and gate['job']==gs['job']
        assert gs['pushed_commit']==head
        validate=runpy.run_path(str(REPO/'research/connections/state-tokens/workflow.py'))['validate_gate']
        validate(gate,gs)
        microbatch=gate['microbatch']
        peak=max(gate['profiles']['state_tokens']['peak_reserved_bytes'],gate['restart']['peak_reserved_bytes'])
        minimum_mb=max(22000,math.ceil(peak/.85/(1024**2)/1000)*1000)
        model_commit=gate['source']['commit']
        changed=command(['git','diff','--name-only',model_commit,head,'--','nanochat','scripts',
                         'research/connections/runtime','research/connections/toolchain'],REPO)
        assert not changed,changed
        data=json.loads(Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048 and data['manifest']['training_tokens']==STEPS*32768
        runtime=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert runtime['complete'] and runtime['actual_eviction']
        for name in ('bundles','logs','results'):(root/name).mkdir(exist_ok=True)
        bundle=root/'bundles';wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(GATE/'bundles/source.tar',bundle/'source.tar')
        (bundle/wheel).symlink_to(ASSETS/'bundles'/wheel)
        shutil.copy2(REPO/'research/connections/state-tokens/train.sh',root/'train.sh')
        shutil.copy2(REPO/'research/connections/state-tokens/preflight.py',bundle/'preflight.py')
        hashes=json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        hashes.pop('gate-data.tar')
        hashes['preflight.py']=hashlib.sha256((bundle/'preflight.py').read_bytes()).hexdigest()
        hashes['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        assert hashes['source.tar']==gs['hashes']['source.tar']
        data_name=data['data_uri'].rsplit('/',1)[-1]
        hashes[data_name]=data['archive']['sha256']
        profile=dict(gate['profiles']['state_tokens'])
        profile.update(additional_peak_allocated_bytes=gate['additional_peak_allocated_bytes'],
                       additional_peak_reserved_bytes=profile['peak_reserved_bytes']-gate['profiles']['baseline']['peak_reserved_bytes'])
        source=dict(commit=model_commit,launcher_commit=head,formal_training=True,gate_job=gs['job'],
                    purpose='Authorized state-token residual Write10B; K4/shared Read-Write/Triton/original ReLU-squared MLP',
                    steps=STEPS,training_tokens=STEPS*32768,data_uri=data['data_uri'],wandb_group=KEY,
                    launcher_sha256=hashlib.sha256((root/'train.sh').read_bytes()).hexdigest(),
                    profile_reference=dict(state_tokens=profile),checkpoint_upload_seconds=1800,
                    microbatch=microbatch,minimum_gpu_mb=minimum_mb,initialization="from_seed42_not_gate_weights")
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        hashes['source.json']=hashlib.sha256((bundle/'source.json').read_bytes()).hexdigest()
        (bundle/'inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        run_id=uuid.uuid4().hex[:8]
        template=(ASSETS/'gate.sub').read_text()
        template=template.replace('executable = validate.sh',f'executable = train.sh\narguments = {run_id} {microbatch}')
        template=template.replace('request_disk = 40GB','request_disk = 80GB')
        template=template.replace('gpus_minimum_capability = 9.0','gpus_minimum_capability = 8.0')
        template=template.replace('gpus_minimum_memory = 22000M',f'gpus_minimum_memory = {minimum_mb}M')
        template=template.replace('(GlobalMemoryMb >= 22000)',f'(GlobalMemoryMb >= {minimum_mb})')
        template=template.replace('(Capability == 9.0)','((Capability == 9.0) || ((Capability == 8.0) && regexp("A100",DeviceName)))')
        template=template.replace('when_to_transfer_output = ON_EXIT',
             'when_to_transfer_output = ON_EXIT\ncheckpoint_exit_code = 85\ntransfer_checkpoint_files = checkpoints.tar,results.tar\nencrypt_input_files = /home/yzhong239/.netrc')
        inputs=[str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs+=['/home/yzhong239/.netrc',data['data_uri'],
                 'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
                 'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        for line in template.splitlines():
            if line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            if line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            if line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            line=line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-state-tokens-residual-10b')
            line=line.replace('DepthDiagnosticKey','StateTokensTrainKey').replace('depth-diagnostic-20261103',KEY)
            lines.append(line)
        lines=[('hold = True\n'+line) if line=='queue 1' else line for line in lines]
        (root/'train.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'],root)
        state=dict(status='submitting',run_id=run_id,source=source,hashes=hashes,formal_training=True,
                   expected_steps=STEPS,expected_tokens=STEPS*32768,submitted_at=time.time())
        (root/'submission-intent.json').write_text(json.dumps(state,indent=2)+'\n')
        output=command(['condor_submit','-terse','train.sub'],root)
        job=output.split()[0];assert len(job.split('.'))==2 and all(x.isdigit() for x in job.split('.')),output
        state.update(status='training_submitted',job=job,
                     wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        rows=json.loads(command(['condor_q',job,'-json'],root));assert len(rows)==1
        ad=rows[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert 'A100' in ad['RequireGPUs'] and str(minimum_mb) in ad['RequireGPUs']
        assert ad['RequestGPUs']==1 and ad['RequestMemory']==16384 and ad['RequestDisk']==80*1024**2
        assert ad['SuccessCheckpointExitCode']==85
        assert set(ad['TransferCheckpoint'].split(','))=={'checkpoints.tar','results.tar'}
        assert set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert data_name in ad['TransferInput'] and 'gate-data.tar' not in ad['TransferInput']
        assert ad['StateTokensTrainKey']==KEY
        keys=['ClusterId','ProcId','JobStatus','NumJobStarts','RequestGPUs','RequestMemory','RequestDisk',
              'RequireGPUs','Rank','WantGPULab','SuccessCheckpointExitCode','TransferCheckpoint',
              'TransferOutput','TransferInput','EncryptInputFiles','ContainerImage','StateTokensTrainKey']
        (root/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job],root)
        print(json.dumps({k:state[k] for k in ('status','job','run_id','wandb_url','expected_steps','expected_tokens')},indent=2))


if __name__=='__main__':main(sys.argv[1])
