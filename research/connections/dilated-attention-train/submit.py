"""Submit one fresh, resumable 10B run of validated progressive dilated attention."""
import fcntl
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import uuid
from policy import require_gpus, RANK

REPO = Path(__file__).resolve().parents[3]
HOME_ROOT = Path('/home/yzhong239')
ASSETS = HOME_ROOT/'nanochat-runtime-assets'
KEY = 'dilated-attention-10b-20261109'
STEPS = 305176
REPORT_DIR = REPO/'research/connections/dilated-attention'
GATE_ROOT = HOME_ROOT/'dilated-attention-gate-20261109-r3'
validate = runpy.run_path(str(REPORT_DIR/'validation.py'))['validate']


def command(args, cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=45).strip()


def submission_text(run_id, microbatch, inputs):
    template = (ASSETS/'gate.sub').read_text()
    template = template.replace('executable = validate.sh',f'executable = train.sh\narguments = {run_id} {microbatch}')
    template = template.replace('request_disk = 40GB','request_disk = 80GB')
    template = template.replace('when_to_transfer_output = ON_EXIT',
        'when_to_transfer_output = ON_EXIT\ncheckpoint_exit_code = 85\n'
        'transfer_checkpoint_files = checkpoints.tar,results.tar\n'
        'encrypt_input_files = /home/yzhong239/.netrc')
    lines=[]
    for line in template.splitlines():
        if line.startswith('transfer_input_files ='):
            line='transfer_input_files = '+','.join(inputs)
        elif line.startswith('transfer_output_files ='):
            line='transfer_output_files = results.tar,checkpoints.tar'
        elif line.startswith('transfer_output_remaps ='):
            line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
        elif line.startswith('require_gpus ='):
            line='require_gpus = '+require_gpus(False)
        elif line.startswith('rank ='):
            line='rank = '+RANK
        elif line.startswith('gpus_minimum_capability ='):
            line='gpus_minimum_capability = 8.0'
        line=line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-dilated-attention-10b')
        line=line.replace('DepthDiagnosticKey','DilatedAttentionTrainKey').replace('depth-diagnostic-20261103',KEY)
        lines.append(line)
    return ('\n'.join(lines)+'\n').replace('queue 1','hold = true\nqueue 1')


def main(root):
    root=Path(root).resolve()
    assert root == HOME_ROOT/KEY
    root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((root/n).exists() for n in ('STOPPED','STOPPED.json'))
        if (root/'state.json').exists():
            state=json.loads((root/'state.json').read_text())
            print(json.dumps({k:state[k] for k in ('status','job','run_id')},indent=2))
            return
        assert not (root/'submission-intent.json').exists(), 'Recover existing intent; never duplicate submission'
        assert not any((root/n).exists() for n in ('checkpoints','checkpoints.tar','results.tar'))
        assert not (root/'results').exists() or not any((root/'results').iterdir())
        assert not command(['git','status','--porcelain'])
        head=command(['git','rev-parse','HEAD'])
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'DilatedAttentionTrainKey == "{KEY}"',
                                  '-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not active, 'Recover existing marked job instead of duplicating'
        gate_state=json.loads((GATE_ROOT/'state.json').read_text())
        assert gate_state['status']=='complete'
        assert not any((GATE_ROOT/n).exists() for n in ('STOPPED','STOPPED.json'))
        gate=json.loads((REPORT_DIR/'validation.json').read_text())
        validate(gate,gate_state['source'])
        assert gate['job']==gate_state['job']=='8245772.0'
        cfg=gate['training']['progressive_dilated']['config']['model']
        expected=dict(variant='baseline',architecture='vanilla',streams=1,depth=8,width=512,
            heads=4,sequence_length=2048,vocab_size=32768,attention_pattern='progressive_dilated',
            sparse_budget=256,sparse_core=128,sparse_sinks=4,sparse_backend='triton',
            attention_kernel='original',activation_checkpoint=False,fused_connection_grad=False)
        assert all(cfg[k]==value for k,value in expected.items())
        history=json.loads(command(['condor_history',gate['job'],'-limit','1','-json','-attributes','JobStatus,ExitCode']) or '[]')
        assert len(history)==1 and history[0]['JobStatus']==4 and history[0]['ExitCode']==0
        microbatch=gate['microbatch']
        data=json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048
        assert data['manifest']['training_tokens']==STEPS*32768
        assert data['data_uri'].endswith('/nanochat-2k-10b-data-7685535.tar')
        runtime=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert runtime['complete'] and runtime['actual_eviction']
        assert (HOME_ROOT/'.netrc').stat().st_mode & 0o777 == 0o600
        for name in ('bundles','logs','results'):
            (root/name).mkdir(exist_ok=True)
        bundle=root/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(bundle/'source.tar'),head])
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        (bundle/wheel).symlink_to(ASSETS/'bundles'/wheel)
        shutil.copy2(REPO/'research/connections/dilated-attention-train/train.sh',root/'train.sh')
        source=dict(commit=head,launcher_commit=head,formal_training=True,gate_job=gate['job'],
            validated_model_commit=gate['source']['commit'],
            purpose='Vanilla progressive dilated attention plus real prefix sinks10B; fresh seed42,8layers D512 original ReLU-squared MLP',
            model_variant='baseline',attention_pattern='progressive_dilated',sparse_backend='triton',
            sparse_budget=256,sparse_core=128,sparse_sinks=4,stride_schedule=[1,3,5,7,9,11,14,None],
            initialization='seed42_from_zero_no_validation_weights',steps=STEPS,training_tokens=STEPS*32768,
            data_uri=data['data_uri'],wandb_group=KEY,checkpoint_upload_seconds=1800,
            passed_families=[gate['capability']],microbatch=microbatch,
            family_preflight='30checks and25-to60 NTP restart for a new family; diagnostic weights deleted',
            scheduling='H100/H200 first; expand to A100/PRO6000 after3600 continuous Idle seconds',
            profile_reference={},validation_step_seconds=gate['mean_step_seconds'],file_hash_checks=False)
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        run_id=uuid.uuid4().hex[:8]
        inputs=[str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs += [str(HOME_ROOT/'.netrc'),data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        (root/'train.sub').write_text(submission_text(run_id,microbatch,inputs))
        command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'],root)
        state=dict(status='submitting',run_id=run_id,source=source,formal_training=True,
                   expected_steps=STEPS,expected_tokens=STEPS*32768,submitted_at=time.time())
        (root/'submission-intent.json').write_text(json.dumps(state,indent=2)+'\n')
        output=command(['condor_submit','-terse','train.sub'],root);job=output.split()[0]
        assert len(job.split('.'))==2 and all(x.isdigit() for x in job.split('.')),output
        state.update(status='training_submitted',job=job,wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        rows=json.loads(command(['condor_q',job,'-json'],root));assert len(rows)==1
        ad=rows[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==16384
        assert ad['RequestDisk']==80*1024**2 and ad['GPUsMinCapability']==8.0
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' not in ad['RequireGPUs']
        assert ad['SuccessCheckpointExitCode']==85
        assert set(ad['TransferCheckpoint'].split(','))=={'checkpoints.tar','results.tar'}
        assert set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert data['data_uri'].rsplit('/',1)[-1] in ad['TransferInput']
        assert all(name not in ad['TransferInput'] for name in ('gate-data.tar','inputs.sha256','checkpoints.tar','results.tar'))
        assert ad['DilatedAttentionTrainKey']==KEY and ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        assert 'countMatches' in ad['Rank'] and 'RTX PRO 6000' not in ad['RequireGPUs']
        keys=['ClusterId','ProcId','JobStatus','NumJobStarts','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
              'RequireGPUs','GPUsMinCapability','Rank','WantGPULab','SuccessCheckpointExitCode','TransferCheckpoint',
              'TransferOutput','TransferInput','EncryptInputFiles','ContainerImage','DilatedAttentionTrainKey']
        (root/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job],root)
        print(json.dumps({k:state[k] for k in ('status','job','run_id','wandb_url','expected_steps','expected_tokens')},indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
