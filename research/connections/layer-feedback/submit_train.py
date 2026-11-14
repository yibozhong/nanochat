"""Submit the single authorized 10B layer-attention feedback run with durable restart."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from policy import RANK, require_gpus

REPO=Path(__file__).resolve().parents[3]
HOME_ROOT=Path('/home/yzhong239')
ASSETS=HOME_ROOT/'nanochat-runtime-assets'
ROOT=HOME_ROOT/'layer-feedback-10b-20261113'
KEY=ROOT.name
STEPS=305176


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((ROOT/n).exists() for n in ('STOPPED','STOPPED.json'))
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover previous submission; do not duplicate'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'LayerFeedbackTrainKey == "{KEY}"','-json']) or '[]')
        assert not active
        assert not command(['git','status','--porcelain'])
        commit=command(['git','rev-parse','HEAD'])
        gate=json.loads((REPO/'research/connections/latent-feedback/validation.json').read_text())
        assert gate['complete'] and gate['gpu_tests_passed']==12 and gate['exact_checkpoint_transport_and_fixed_gradient_update']
        assert gate['model']==dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768)
        assert gate['attention_backend']=='fa2' and gate['microbatch']==16 and gate['training_pass_mix']==[.15,.85]
        # The shared runtime is validated; new model checks run in this allocation.
        assert not command(['git','diff','--name-only',gate['model_commit'],'HEAD','--',
            'research/connections/runtime','research/connections/toolchain'])
        transport=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        data=json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048
        assert data['manifest']['training_tokens']==STEPS*32768
        assert (HOME_ROOT/'.netrc').stat().st_mode&0o777==0o600
        for name in ('bundles','logs','results'):(ROOT/name).mkdir(exist_ok=True)
        assert not any((ROOT/'results').iterdir())
        bundle=ROOT/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(bundle/'source.tar'),commit])
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        (bundle/wheel).symlink_to(ASSETS/'bundles'/wheel)
        shutil.copy2(REPO/'research/connections/layer-feedback/train.sh',ROOT/'train.sh')
        (ROOT/'train.sh').chmod(0o755)
        run_id=uuid.uuid4().hex[:8]
        source=dict(commit=commit,formal_training=True,plan=KEY,run_id=run_id,
            model_variant='layer_feedback',reference_model_commit=gate['model_commit'],reference_validation_job=gate['job'],
            validation='same-allocation 39 GPU tests, 96 steps per comparator, CLI recovery, then independent formal process',
            model=gate['model'],steps=STEPS,training_tokens=STEPS*32768,batch_tokens=32768,microbatch=16,
            seed=42,initialization='seed42_from_zero_no_diagnostic_weights',
            pass_mix=[.15,.85],eval_passes=4,earlier_passes='no_grad_detached',
            feedback_source='layers',feedback_attention_backend='triton',query_and_output='full_width_512',
            attention_backend='fa2',data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock',
            checkpoint_seconds=1800,evaluate_test=True,
            scheduling='H100/H200 first; A100 after 3600 continuous idle seconds, with same-allocation feedback preflight',
            passed_families=[],compatible_fallback_backend='A100 FA2/CCE runtime validated in skip-scale/validation.json',
            reference_top_step_seconds=gate['feedback_step_ms']/1000,
            estimated_step_seconds=gate['feedback_step_ms']/1000*1.15)
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        inputs=[str(p.relative_to(ROOT)) for p in bundle.iterdir()]
        inputs += [str(HOME_ROOT/'.netrc'),data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        for line in (ASSETS/'gate.sub').read_text().splitlines():
            if line.startswith('executable ='):line=f'executable = train.sh\narguments = {run_id}'
            elif line.startswith('request_disk ='):line='request_disk = 80GB'
            elif line.startswith('gpus_minimum_capability ='):line='gpus_minimum_capability = 8.0'
            elif line.startswith('require_gpus ='):line='require_gpus = '+require_gpus()
            elif line.startswith('rank ='):line='rank = '+RANK
            elif line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            elif line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            elif line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            elif line.startswith('queue '):continue
            lines.append(line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-layer-feedback-10b')
                .replace('DepthDiagnosticKey','LayerFeedbackTrainKey').replace('depth-diagnostic-20261103',KEY))
        lines += ['checkpoint_exit_code = 85','transfer_checkpoint_files = checkpoints.tar,results.tar',
            'encrypt_input_files = /home/yzhong239/.netrc','+InteractiveJob = true',
            'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
            'hold = true','queue 1']
        (ROOT/'train.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dry-run.ad','train.sub'],ROOT)
        state=dict(status='submitting',run_id=run_id,source=source,formal_training=True,submitted_at=time.time(),
            wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}',
            estimated_h200_training_hours=STEPS*source['estimated_step_seconds']/3600)
        (ROOT/'submission-intent.json').write_text(json.dumps(state,indent=2)+'\n')
        job=command(['condor_submit','-terse','train.sub'],ROOT).split()[0]
        state.update(job=job,status='held_for_audit')
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==16384
        assert ad['RequestDisk']==80*1024**2 and ad['GPUsMinCapability']==8.0
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' not in ad['RequireGPUs']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and data['data_uri'] in ad['TransferInput']
        assert 'gate-data.tar' not in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
        assert ad['SuccessCheckpointExitCode']==85 and set(ad['TransferCheckpoint'].split(','))=={'results.tar','checkpoints.tar'}
        assert set(ad['TransferOutput'].split(','))=={'results.tar','checkpoints.tar'}
        assert ad['LayerFeedbackTrainKey']==KEY and ad['InteractiveJob'] and ad['WantGPULab']
        assert 'backfill' in ad['Rank'] and 'countMatches' in ad['Rank']
        keys=('ClusterId','ProcId','JobStatus','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
            'RequireGPUs','TransferInput','TransferOutput','TransferCheckpoint','SuccessCheckpointExitCode',
            'EncryptInputFiles','ContainerImage','Rank','OnExitHold')
        (ROOT/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job])
        state['status']='training_submitted'
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:state[k] for k in ('status','job','run_id','wandb_url','estimated_h200_training_hours')},indent=2))


if __name__=='__main__':main()
