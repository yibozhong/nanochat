"""Submit one named nonzero-init residual run with durable restart."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid

from policy import RANK, require_gpus

REPO=Path(__file__).resolve().parents[3]
HOME_ROOT=Path('/home/yzhong239')
ASSETS=HOME_ROOT/'nanochat-runtime-assets'
PRESETS={'vanilla':'vanilla', 'ngpt':'ngpt', 'ngpt_momentum':'ngpt_heavy_ball',
         'ngpt_momentum_transport_rms':'ngpt_heavy_ball', 'heavy_ball_rms':'heavy_ball_rms',
         'ngpt_hbformer':'ngpt_hbformer', 'ngpt_hbformer_no_wn':'ngpt_hbformer',
         'ngpt_hbformer_velocity_ln':'ngpt_hbformer', 'ngpt_lookahead':'ngpt_lookahead'}
STEPS=305176


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('preset',choices=PRESETS)
    parser.add_argument('--extend-to-schedule', action='store_true', help='Continue the completed lookahead pilot to its original 10B schedule')
    args=parser.parse_args()
    preset=args.preset
    if args.extend_to_schedule and preset != 'ngpt_lookahead':
        parser.error('Only the lookahead pilot has a continuation preset')
    mode=PRESETS[preset]
    short_control = preset == 'ngpt_hbformer_no_wn'
    steps = 5000 if short_control or (mode == 'ngpt_lookahead' and not args.extend_to_schedule) else STEPS
    budget = '5k-steps' if steps == 5000 else '10b'
    ROOT=HOME_ROOT/f'residual-nonzero-{preset}-{budget}-20261117'
    KEY=ROOT.name
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((ROOT/n).exists() for n in ('STOPPED','STOPPED.json'))
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover previous submission; do not duplicate'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'NgptTrainKey == "{KEY}"','-json']) or '[]')
        assert not active
        assert not command(['git','status','--porcelain'])
        commit=command(['git','rev-parse','HEAD'])
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
        shutil.copy2(REPO/'research/connections/ngpt/train.sh',ROOT/'train.sh')
        (ROOT/'train.sh').chmod(0o755)
        run_id=uuid.uuid4().hex[:8]
        source=dict(commit=commit,formal_training=True,plan=KEY,run_id=run_id,
            model_variant=preset,preset=preset,residual_dynamics=mode,velocity_embedding=mode in ('ngpt_heavy_ball','heavy_ball_rms','ngpt_hbformer','ngpt_lookahead'),
            ngpt_alpha_init=.05,ngpt_force_rms_floor=1e-6,ngpt_velocity_mode='ambient',
            residual_projection_init='normal',residual_projection_init_std=.02,
            output_projection_actual_std=.005,momentum_beta_init=.5,momentum_gamma_init=.5,
            momentum_eta_init=.1 if mode=='ngpt_heavy_ball' else 1.,
            model=dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768),steps=steps,training_tokens=steps*32768,batch_tokens=32768,microbatch=16,
            seed=42,initialization='seed42_from_scratch_no_diagnostic_weights',
            preflight='29 tests; target128 + other two modes32 updates each; initial unit targets; exact CLI resume; fresh seed42 formal',
            gate_timeout_seconds=2400,failure_debug_seconds=900,
            attention_backend='fa2',data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock',
            checkpoint_seconds=1800,evaluate_test=True,
            scheduling='H100/H200 first; A100 after 3600 continuous idle seconds, with same-allocation nGPT preflight',
            passed_families=[],compatible_fallback_backend='A100 FA2/CCE runtime validated in skip-scale/validation.json',
            prior_h100_reference_step_seconds=.06704051233828068 if mode=='vanilla' else .0756970327347517,
            timing_reference='previous zero-init H100 diagnostics, estimate only; measure all three nonzero modes in preflight')
        if mode == 'ngpt_lookahead':
            del source['momentum_gamma_init'], source['momentum_eta_init']
            source.update(velocity_init='token_position', velocity_embedding_std=.02, momentum_beta_trainable=False,
                          lookahead_mu_init=0., ngpt_velocity_mode='lookahead_transport', ngpt_weight_norm=False, schedule_steps=STEPS,
                          reference_run='d065c132', backbone_weight_constraint='none',
                          preflight='Zero-gate equality with nGPT, trainable lookahead, tangent geometry, causal KV, gradients, exact resume and same-GPU speed')
        if mode=='ngpt_hbformer':
            source.pop('output_projection_actual_std')
            source.update(velocity_init='token_position', velocity_embedding_std=.02, momentum_eta_init=1.,
                          momentum_eta_trainable=False, momentum_gamma_init=math.log(2.), ngpt_velocity_mode='transport_rms',
                          momentum_velocity_rms_floor=.1, backbone_weight_constraint='ngpt_unit_l2_init_and_post_step',
                          output_projection_std_before_normalization=.005,
                          preflight='HBFormer velocity/step, unit weight axes and post-step projection, cache, gradients, exact resume; target128 and two controls32',
                          timing_reference='prior H100 diagnostics, excludes new post-step weight normalization overhead')
        if short_control or preset == 'ngpt_hbformer_velocity_ln':
            source.update(ngpt_weight_norm=False, backbone_weight_constraint='none', schedule_steps=STEPS,
                          preflight='Unprojected initialization and optimizer, same-card 128 steps, cache/gradients and exact CLI resume',
                          timing_reference='prior H100 residual diagnostics, confirm same-card speed in preflight')
        if preset == 'ngpt_hbformer_velocity_ln':
            source.update(ngpt_velocity_mode='transport_layernorm', velocity_layernorm_eps=1e-5,
                          velocity_gain_init=1., velocity_layernorm_bias=False,
                          preflight='Affine velocity LayerNorm before projection, geometric and gradient tests, exact resume; target128, vanilla and matched RMS 32 each')
        if mode=='heavy_ball_rms':
            source.update(velocity_init='token', velocity_embedding_std=.02,
                          momentum_eta_init=.05, momentum_velocity_rms_floor=.1,
                          preflight='9 Euclidean velocity RMS tests; target128, vanilla and transport_rms 32 each; exact CLI resume',
                          timing_reference='estimate from prior H100 residual diagnostics; measure same GPU in preflight')
        if preset=='ngpt_momentum_transport_rms':
            report_path=HOME_ROOT/'ngpt-transport-rms-debug-20261117/validation.json'
            report=json.loads(report_path.read_text())
            assert report['complete'] and report['exit_code']==0 and report['resume']['complete']
            assert report['tests']==dict(tests=57, failures=0, errors=0, skipped=0)
            profile=report['profile']
            target=profile['models'][mode]
            assert profile['complete'] and profile['gpu']=='NVIDIA H200' and target['steps']==128
            validated=report['source']['commit']
            changes=command(['git','diff','--name-only',validated,commit]).splitlines()
            assert all(path.startswith('research/connections/ngpt/') for path in changes), changes
            source.update(ngpt_velocity_mode='transport_rms',momentum_velocity_rms_floor=.1,
                momentum_eta_init=.05,velocity_init='token',velocity_embedding_std=.02,
                validation_job=report['job'],validated_model_commit=validated,passed_families=['sm90'],
                preflight='reuse verified sm90 model and runtime; full transport checks on sm80',
                reference_step_seconds=target['median_step_seconds'],reference_gpu=profile['gpu'],
                timing_reference='transport_rms 16x2048 same-card H200 median over steps 17-32')
            del source['prior_h100_reference_step_seconds']
            config=target['config']
            for key in ('residual_dynamics','ngpt_velocity_mode','momentum_velocity_rms_floor',
                        'velocity_init','velocity_embedding_std','ngpt_force_rms_floor','ngpt_alpha_init',
                        'momentum_eta_init','momentum_beta_init','momentum_gamma_init',
                        'residual_projection_init','residual_projection_init_std'):
                assert config[key]==source[key], key
            assert all(config[key]==value for key,value in source['model'].items())
            assert profile['seed']==source['seed'] and profile['microbatch']==source['microbatch']
            assert profile['batch_tokens']==source['batch_tokens'] and profile['attention_backend']=='fa2'
            shutil.copy2(report_path,bundle/'transport-validation.json')
        if args.extend_to_schedule:
            parent=HOME_ROOT/f'residual-nonzero-{preset}-5k-steps-20261117'
            summary=json.loads((parent/'summary.json').read_text())
            previous=json.loads((parent/'state.json').read_text())
            old=summary['source']
            assert previous['status']=='completed' and summary['complete'] and summary['steps']==5000
            assert previous['source']==old and summary['run_id']==previous['run_id']==old['run_id']
            assert not json.loads(command(['condor_q',previous['job'],'-json']) or '[]')
            assert old['schedule_steps']==steps and old['steps']==summary['steps']
            run_id=source['run_id']=summary['run_id']
            metadata={'plan','commit','steps','training_tokens'}
            assert {k:v for k,v in old.items() if k not in metadata} == {k:v for k,v in source.items() if k not in metadata}
            changed=command(['git','diff','--name-only',old['commit'],commit]).splitlines()
            assert all(p.startswith('research/connections/ngpt/') or p=='scripts/connection_train.py' for p in changed),changed
            archives=parent/'results'
            checkpoints=archives/f"train-{previous['job']}-checkpoints.tar"
            with tarfile.open(checkpoints,'r:') as tar:
                for name in ('latest.pt','best.pt'):
                    member=tar.getmember('checkpoints/'+name)
                    assert member.isfile() and member.size==summary['checkpoint_files'][name]['bytes']
            (bundle/'parent-checkpoints.tar').symlink_to(checkpoints)
            (bundle/'parent-results.tar').symlink_to(archives/f"train-{previous['job']}.tar")
            source['continuation']=dict(source=old,job=previous['job'],step=summary['steps'],dev_ce=summary['final_dev_ce'],
                                        checkpoint_files=summary['checkpoint_files'])
            source['reference_step_seconds']=summary['steady_step_seconds']
            source['reference_gpu']=summary['gpu']
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        inputs=[str(p.relative_to(ROOT)) for p in bundle.iterdir()]
        inputs += [str(HOME_ROOT/'.netrc'),data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        job_args=f'{run_id} {mode}'
        if source['ngpt_velocity_mode'] != 'ambient':
            job_args+=' '+source['ngpt_velocity_mode']
        if args.extend_to_schedule:
            job_args+=' --extend-to-schedule'
        if mode == 'ngpt_hbformer' and source.get('ngpt_weight_norm') is False:
            job_args+=' --no-ngpt-weight-norm'
        for line in (ASSETS/'gate.sub').read_text().splitlines():
            if line.startswith('executable ='):line=f'executable = train.sh\narguments = {job_args}'
            elif line.startswith('request_disk ='):line='request_disk = 80GB'
            elif line.startswith('gpus_minimum_capability ='):line='gpus_minimum_capability = 8.0'
            elif line.startswith('require_gpus ='):line='require_gpus = '+require_gpus()
            elif line.startswith('rank ='):line='rank = '+RANK
            elif line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            elif line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            elif line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            elif line.startswith('queue '):continue
            lines.append(line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic',f'nanochat-nonzero-residual-{budget}-{preset}')
                .replace('DepthDiagnosticKey','NgptTrainKey').replace('depth-diagnostic-20261103',KEY))
        lines += ['checkpoint_exit_code = 85','transfer_checkpoint_files = checkpoints.tar,results.tar',
            'encrypt_input_files = /home/yzhong239/.netrc','+InteractiveJob = true',
            'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
            'hold = true','queue 1']
        (ROOT/'train.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dry-run.ad','train.sub'],ROOT)
        state=dict(status='submitting',run_id=run_id,source=source,formal_training=True,submitted_at=time.time(),
            wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        if 'reference_step_seconds' in source:
            state['estimated_h200_training_hours']=(steps-source.get('continuation',{}).get('step',0))*source['reference_step_seconds']/3600
        else:
            state['estimated_h100_training_hours']=steps*source['prior_h100_reference_step_seconds']/3600
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
        if preset=='ngpt_momentum_transport_rms':
            assert 'transport-validation.json' in ad['TransferInput']
        assert ad['SuccessCheckpointExitCode']==85 and set(ad['TransferCheckpoint'].split(','))=={'results.tar','checkpoints.tar'}
        assert set(ad['TransferOutput'].split(','))=={'results.tar','checkpoints.tar'}
        assert ad['NgptTrainKey']==KEY and ad['InteractiveJob'] and ad['WantGPULab']
        assert ad['Args']==job_args
        if args.extend_to_schedule:
            assert 'parent-checkpoints.tar' in ad['TransferInput'] and 'parent-results.tar' in ad['TransferInput']
        assert 'backfill' in ad['Rank'] and 'countMatches' in ad['Rank']
        keys=('ClusterId','ProcId','JobStatus','Args','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
            'RequireGPUs','TransferInput','TransferOutput','TransferCheckpoint','SuccessCheckpointExitCode',
            'EncryptInputFiles','ContainerImage','Rank','OnExitHold')
        (ROOT/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job])
        state['status']='training_submitted'
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:v for k,v in state.items() if k in ('status','job','run_id','wandb_url') or k.startswith('estimated_')},indent=2))


if __name__=='__main__':main()
