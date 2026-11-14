"""Submit one selected, validated projected-feedback 10B run; two modes authorized."""
import argparse
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
STEPS=305176


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('embedding_qkv','per_layer_qkv'))
    mode=parser.parse_args().mode
    ROOT=HOME_ROOT/('projected-feedback-'+mode.replace('_','-')+'-10b-20261114')
    KEY=ROOT.name
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((ROOT/n).exists() for n in ('STOPPED','STOPPED.json'))
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover previous submission; do not duplicate'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'ProjectedFeedbackTrainKey == "{KEY}"','-json']) or '[]')
        assert not active
        assert not command(['git','status','--porcelain'])
        commit=command(['git','rev-parse','HEAD'])
        gate=json.loads((REPO/'research/connections/projected-feedback/validation.json').read_text())
        assert gate['complete'] and gate['gpu_tests_passed']==97 and gate['exact_checkpoint_transport_and_fixed_gradient_update']
        assert mode in gate['variants'] and gate['projection_order']=='reassociated'
        measured=gate['variants'][mode]
        assert measured['finite_steps']==96 and measured['all_projections_learned'] and measured['resume_complete']
        assert measured['efficiency_review_passed']
        assert gate['model']==dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768)
        assert gate['attention_backend']=='fa2' and gate['microbatch']==16 and gate['training_pass_mix']==[.15,.85]
        # Normal Git diff, no content-hash audit. Model/runtime remain validated.
        assert not command(['git','diff','--name-only',gate['model_commit'],'HEAD','--',
            'nanochat','scripts/connection_train.py','research/connections/runtime','research/connections/toolchain'])
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
        shutil.copy2(REPO/'research/connections/projected-feedback/train.sh',ROOT/'train.sh')
        (ROOT/'train.sh').chmod(0o755)
        run_id=uuid.uuid4().hex[:8]
        source=dict(commit=commit,formal_training=True,plan=KEY,run_id=run_id,
            model_variant='projected_feedback',feedback_source=mode,feedback_projection_order='reassociated',
            feedback_attention_backend='triton',validated_model_commit=gate['model_commit'],validation_job=gate['job'],
            model=gate['model'],steps=STEPS,training_tokens=STEPS*32768,batch_tokens=32768,microbatch=16,
            seed=42,initialization='seed42_from_zero_no_diagnostic_weights',
            pass_mix=[.15,.85],eval_passes=4,earlier_passes='no_grad_detached',
            attention_backend='fa2',data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock',
            checkpoint_seconds=1800,evaluate_test=True,
            scheduling='H100/H200 first; A100 after 3600 continuous idle seconds, with same-allocation feedback preflight',
            passed_families=[[9,0]],compatible_fallback_backend='A100 FA2/CCE runtime validated in skip-scale/validation.json',
            validation_step_seconds=measured['mean_step_seconds'],
            validation_p2_step_seconds=measured['p2_step_seconds'],validation_peak_bytes=measured['peak_allocated_bytes'])
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        inputs=[str(p.relative_to(ROOT)) for p in bundle.iterdir()]
        inputs += [str(HOME_ROOT/'.netrc'),data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        for line in (ASSETS/'gate.sub').read_text().splitlines():
            if line.startswith('executable ='):line=f'executable = train.sh\narguments = {run_id} {mode}'
            elif line.startswith('request_disk ='):line='request_disk = 80GB'
            elif line.startswith('gpus_minimum_capability ='):line='gpus_minimum_capability = 8.0'
            elif line.startswith('require_gpus ='):line='require_gpus = '+require_gpus()
            elif line.startswith('rank ='):line='rank = '+RANK
            elif line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            elif line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            elif line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            elif line.startswith('queue '):continue
            lines.append(line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-projected-feedback-'+mode+'-10b')
                .replace('DepthDiagnosticKey','ProjectedFeedbackTrainKey').replace('depth-diagnostic-20261103',KEY))
        lines += ['checkpoint_exit_code = 85','transfer_checkpoint_files = checkpoints.tar,results.tar',
            'encrypt_input_files = /home/yzhong239/.netrc','+InteractiveJob = true',
            'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
            'hold = true','queue 1']
        (ROOT/'train.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dry-run.ad','train.sub'],ROOT)
        state=dict(status='submitting',run_id=run_id,source=source,formal_training=True,submitted_at=time.time(),
            wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}',
            estimated_h200_training_hours=STEPS*source['validation_step_seconds']/3600)
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
        assert ad['ProjectedFeedbackTrainKey']==KEY and ad['InteractiveJob'] and ad['WantGPULab']
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
