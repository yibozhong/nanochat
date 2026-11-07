"""Submit the authorized single 10B run from the already validated GPU source."""
import fcntl
import hashlib
import json
import runpy
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

REPO=Path(__file__).resolve().parents[3]
GATE=Path('/home/yzhong239/post-gate-opt-20261107')
PRE_GATE=Path('/home/yzhong239/stream-sigmoid-gate-20261107')
ASSETS=Path('/home/yzhong239/nanochat-runtime-assets')
KEY='pre-gated-hc-10b-20261107'
STEPS=305176


def command(args,cwd):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()



def validate_gate(gate,gs,pre,ps):
    assert gate['complete'] and gs['status']=='complete' and gate['job']==gs['job']
    runpy.run_path(str(REPO/'research/connections/post-gate-opt/collect.py'))['validate'](gate,gs)
    assert gate['checks']==dict(tests=36,failures=0,errors=0,skipped=0)
    assert gate['restart']['steps']==128 and gate['restart']['complete']
    assert gate['profiles']['pre']['model']['variant']=='stream_gated_hc'
    assert gate['profiles']['pre']['implementation']=='pre'
    assert gate['microbatch']==16 and gate['profiles']['pre']['actual_attention_backend']=='fa3'
    assert ps['status']=='complete' and pre['job']==ps['job']
    runpy.run_path(str(REPO/'research/connections/stream-gate/collect.py'))['validate'](pre,ps)
    assert pre['restart']['steps']==128 and pre['restart']['complete']
    assert pre['restart']['variant']=='stream_gated_hc' and pre['microbatch']==gate['microbatch']


def main(root):
    root=Path(root).resolve();assert root==Path('/home/yzhong239')/KEY
    root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():
            print((root/'state.json').read_text());return
        assert not (root/'submission-intent.json').exists(),'Recover the previous submission intent; do not duplicate'
        assert not (root/'checkpoints').exists() and not (root/'checkpoints.tar').exists()
        assert not (root/'results.tar').exists(),'New formal run must start with no checkpoint or results'
        assert not (root/'results').exists() or not any((root/'results').iterdir())
        assert not command(['git','status','--porcelain'],REPO)
        head=command(['git','rev-parse','HEAD'],REPO)
        assert head==command(['git','ls-remote','origin','refs/heads/main'],REPO).split()[0]
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'PreGatedHCTrainKey == "{KEY}"',
                                  '-json','-attributes','ClusterId,ProcId'],root) or '[]')
        assert not active,'Recover existing marked job instead of duplicating'
        gate=json.loads((REPO/'research/connections/post-gate-opt/validation.json').read_text())
        gs=json.loads((GATE/'state.json').read_text())
        pre=json.loads((REPO/'research/connections/stream-gate/validation.json').read_text())
        ps=json.loads((PRE_GATE/'state.json').read_text())
        validate_gate(gate,gs,pre,ps)
        assert pre==json.loads((PRE_GATE/'validation.json').read_text())
        # Original pre-gate math and its independently validated resume are unchanged.
        core=['nanochat/stream_gate_mixing.py','nanochat/stream_gated_connections.py',
              'nanochat/nvidia_connections.py','nanochat/nvidia_ops.py','nanochat/gpt.py',
              'nanochat/packed_training.py','nanochat/optim.py','tests/test_stream_gated_hc.py']
        assert not command(['git','diff','--name-only',pre['source']['commit'],head,'--',*core],REPO)
        assert gate==json.loads((GATE/'validation.json').read_text())
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
        shutil.copy2(REPO/'research/connections/pre-gate-train/train.sh',root/'train.sh')
        hashes=json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        hashes.pop('gate-data.tar')
        hashes['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        assert hashes['source.tar']==gs['hashes']['source.tar']
        data_name=data['data_uri'].rsplit('/',1)[-1]
        hashes[data_name]=data['archive']['sha256']
        profile=dict(gate['profiles']['pre'])
        source=dict(commit=model_commit,launcher_commit=head,formal_training=True,gate_job=gs['job'],
                    purpose='Pre-H independent-stream sigmoid gated HC10B, paired with post-gate run7952576.0; vanilla8layers D512 original ReLU-squared MLP',
                    comparison_post_job='7952576.0',pre_restart_gate_job=ps['job'],
                    model_variant='stream_gated_hc',gate_position='before_shared_H_branch_write_ungated',
                    initialization='seed42_from_zero_no_validation_weights',
                    steps=STEPS,training_tokens=STEPS*32768,data_uri=data['data_uri'],wandb_group=KEY,
                    launcher_sha256=hashlib.sha256((root/'train.sh').read_bytes()).hexdigest(),
                    profile_reference=dict(stream_gated_hc=profile),checkpoint_upload_seconds=1800,
                    passed_families=[9],microbatch=gate['microbatch'])
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        hashes['source.json']=hashlib.sha256((bundle/'source.json').read_bytes()).hexdigest()
        (bundle/'inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        run_id=uuid.uuid4().hex[:8]
        template=(ASSETS/'gate.sub').read_text()
        template=template.replace('executable = validate.sh',f'executable = train.sh\narguments = {run_id} {gate["microbatch"]}')
        template=template.replace('request_disk = 40GB','request_disk = 80GB')
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
            line=line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-pre-gated-hc-10b')
            line=line.replace('DepthDiagnosticKey','PreGatedHCTrainKey').replace('depth-diagnostic-20261103',KEY)
            lines.append(line)
        template='\n'.join(lines)+'\n'
        # Formal training is Hopper-only; A100 remains allowed in submit.py's validation jobs.
        template=template.replace('(Capability == 9.0)', '(Capability == 9.0) && regexp("H100|H200",DeviceName)')
        assert 'gpus_minimum_capability = 9.0' in template
        template=template.replace('queue 1','hold = true\nqueue 1')
        (root/'train.sub').write_text(template)
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
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==16384 and ad['RequestDisk']==80*1024**2
        assert ad['GPUsMinCapability']==9.0
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' not in ad['RequireGPUs']
        assert ad['SuccessCheckpointExitCode']==85
        assert set(ad['TransferCheckpoint'].split(','))=={'checkpoints.tar','results.tar'}
        assert set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert data_name in ad['TransferInput'] and 'gate-data.tar' not in ad['TransferInput']
        assert ad['PreGatedHCTrainKey']==KEY
        keys=['ClusterId','ProcId','JobStatus','NumJobStarts','RequestGPUs','RequestMemory','RequestDisk',
              'RequireGPUs','GPUsMinCapability','Rank','WantGPULab','SuccessCheckpointExitCode','TransferCheckpoint',
              'TransferOutput','TransferInput','EncryptInputFiles','ContainerImage','PreGatedHCTrainKey']
        (root/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job],root)
        print(json.dumps({k:state[k] for k in ('status','job','run_id','wandb_url','expected_steps','expected_tokens')},indent=2))


if __name__=='__main__':
    raise SystemExit('Retired independent-stream sigmoid workflow; see STOPPED.json')
