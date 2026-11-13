"""One authorized run; AP-only freezing, submission and bounded-state monitoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
import traceback
import uuid

from run_policy import PLAN,STEPS,TOKENS,BATCH_TOKENS,VERIFIED,validate_source,validate_complete,finite_formal_row

REPO = Path(__file__).resolve().parents[3]
HOME_ROOT = Path('/home/yzhong239')
ROOT = HOME_ROOT/PLAN
WHEEL = 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def stopped():
    return any((ROOT/name).exists() for name in ('STOP','STOPPED','STOPPED.json'))


def description(run_id,data_uri):
    return f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = train.sh
arguments = {run_id}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
encrypt_input_files = /home/yzhong239/.netrc
request_cpus = 2
request_memory = 48GB
request_disk = 100GB
request_gpus = 1
gpus_minimum_memory = 78000M
gpus_minimum_capability = 9.0
requirements = (HasCHTCStaging == true)
require_gpus = (Capability == 9.0) && regexp("H100|H200",DeviceName) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 78000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
+JobBatchName = "{PLAN}"
+FFNReadoutPlan = "{PLAN}"
+FFNReadoutRunId = "{run_id}"
+FFNReadoutFormal = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
transfer_input_files = bundles/source.tar,bundles/source.json,bundles/{WHEEL},/home/yzhong239/.netrc,{data_uri},osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz,https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))
periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > 7200)
hold = true
queue 1
'''


def submit():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not stopped()
        if (ROOT/'state.json').exists():
            return json.loads((ROOT/'state.json').read_text())
        assert not (ROOT/'submission-intent.json').exists(),'Recover prior submission intent; never duplicate'
        assert not json.loads(command(['condor_q','yzhong239','-constraint',f'FFNReadoutPlan == "{PLAN}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not command(['git','status','--porcelain']),'Commit the launch workflow before freezing it'
        commit=command(['git','rev-parse','HEAD'])
        assert not command(['git','diff','--name-only',VERIFIED,commit,'--','nanochat','scripts/connection_train.py',
                            'research/connections/runtime','research/connections/toolchain'])
        report=json.loads((REPO/'research/connections/ffn-readout/results.json').read_text())
        transport=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        data=json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048
        assert data['manifest']['training_steps']==STEPS and data['manifest']['training_tokens']==TOKENS
        assert data['manifest']['tokenizer']['vocab_size']==32768
        assert stat.S_IMODE((HOME_ROOT/'.netrc').stat().st_mode)==0o600
        for name in ('bundles','results','logs'):(ROOT/name).mkdir(exist_ok=True)
        assert not any((ROOT/'results').iterdir()),'Do not reuse a prior output directory'
        run_id=uuid.uuid4().hex[:8]
        source=dict(plan=PLAN,commit=commit,run_id=run_id,formal_training=True,model=report['workload'],
            steps=STEPS,training_tokens=TOKENS,batch_tokens=BATCH_TOKENS,microbatch=16,seed=42,
            initialization='seed42_from_zero_no_diagnostic_weights',attention_backend='fa2',
            data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            checkpoint_seconds=1800,evaluate_test=True,verified_model_commit=VERIFIED,
            prior_validation='research/connections/ffn-readout/results.json',
            original_model_implementation=report['implementation_commit'],
            validated_hardware_family='Hopper SM90; H100/H200; no unvalidated-family expansion',
            validation_step_seconds=report['paired_throughput']['models']['triton']['median_ms']/1000)
        validate_source(source,report)
        write(ROOT/'bundles/source.json',source)
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(ROOT/'bundles/source.tar'),commit])
        wheel=ROOT/'bundles'/WHEEL
        if not wheel.exists():wheel.symlink_to(HOME_ROOT/'nanochat-runtime-assets/bundles'/WHEEL)
        assert wheel.is_file()
        shutil.copy2(REPO/'research/connections/ffn-readout/train.sh',ROOT/'train.sh')
        (ROOT/'train.sh').chmod(0o755)
        command(['bash','-n','train.sh'],ROOT)
        (ROOT/'train.sub').write_text(description(run_id,data['data_uri']))
        command(['condor_submit','-dry-run','dry-run.ads','train.sub'],ROOT)
        write(ROOT/'submission-intent.json',dict(source=source,created_at=time.time()))
        job=command(['condor_submit','-terse','train.sub'],ROOT).split()[0]
        state=dict(job=job,run_id=run_id,status='held_for_audit',source=source,formal_started=False,
                   submitted_at=time.time(),wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}',
                   estimated_pure_training_hours=STEPS*source['validation_step_seconds']/3600)
        write(ROOT/'state.json',state)
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert (ad['RequestGPUs'],ad['RequestCpus'],ad['RequestMemory'],ad['RequestDisk'])==(1,2,49152,100*1024**2)
        assert ad['SuccessCheckpointExitCode']==85
        assert set(ad['TransferCheckpoint'].split(','))=={'checkpoints.tar','results.tar'}
        assert set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
        assert 'nanochat-2k-10b-data-7685535.tar' in ad['TransferInput']
        assert ad['Args']==run_id and ad['FFNReadoutRunId']==run_id and ad['FFNReadoutPlan']==PLAN
        assert ad['FFNReadoutFormal'] and ad['WantGPULab'] and 'backfill' in ad['Rank']
        assert 'H100|H200' in ad['RequireGPUs'] and '78000' in ad['RequireGPUs']
        assert ad['StreamOut'] and ad['StreamErr']
        fields=['ClusterId','ProcId','JobStatus','Args','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
                'TransferInput','EncryptInputFiles','TransferOutput','TransferCheckpoint','SuccessCheckpointExitCode',
                'RequireGPUs','Rank','WantGPULab','ContainerImage','FFNReadoutPlan','FFNReadoutRunId','OnExitHold']
        write(ROOT/'submission-verified.json',{k:ad.get(k) for k in fields})
        command(['condor_release',job])
        state['status']='queued';write(ROOT/'state.json',state)
        return state


def consume_log(state):
    path=ROOT/'logs'/f'train.{state["job"]}.out'
    if not path.exists():return
    offset=state.get('log_offset',0)
    if path.stat().st_size<offset:
        offset=0;state['current_segment_marker']=False;state['log_partial']=''
    with path.open('rb') as f:
        f.seek(offset);text=f.read().decode(errors='replace');state['log_offset']=f.tell()
    lines=(state.pop('log_partial','')+text).split('\n');state['log_partial']=lines.pop()
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN ffn_readout_r64 '+state['run_id']):
            state['current_segment_marker']=True
        if line.startswith('FFN_READOUT_FAILED'):
            state['failure']=line
        if not state.get('current_segment_marker'):continue
        try:row=json.loads(line)
        except ValueError:continue
        if finite_formal_row(row):
            state['formal_started']=True;state['last_formal_update']=row


def collect(state):
    archive=ROOT/'results'/f'train-{state["job"]}.tar'
    checkpoints=ROOT/'results'/f'train-{state["job"]}-checkpoints.tar'
    with tarfile.open(archive) as t:
        summary=json.load(t.extractfile('results/train/summary.json'))
        validate_complete(summary,state['source'])
        small={}
        for name in ('config.json','progress.json','readout-initialization.json'):
            small[name]=json.load(t.extractfile('results/train/'+name))
        offline=sorted({m.name.split('/')[2] for m in t.getmembers()
                        if m.name.startswith('results/wandb/offline-run-') and len(m.name.split('/'))>=3})
    with tarfile.open(checkpoints) as t:
        for name,meta in summary['checkpoint_files'].items():
            assert t.getmember('checkpoints/'+name).size==meta['bytes']
    write(ROOT/'final-summary.json',summary)
    for name,value in small.items():write(ROOT/('final-'+name),value)
    state.update(status='complete',summary=summary,offline_logs=offline)


def tick():
    state=json.loads((ROOT/'state.json').read_text())
    if stopped():
        state['status']='stopped';write(ROOT/'state.json',state);return False
    consume_log(state)
    attrs='ClusterId,ProcId,JobStatus,HoldReason,RemoteHost,NumJobStarts,TransferringInput,ExitCode,ExitBySignal'
    ads=json.loads(command(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
    if ads:
        ad=ads[0];state['scheduler']=ad
        if ad['JobStatus']==5:
            state.update(status='needs_attention',failure=ad.get('HoldReason','Held'))
        elif ad['JobStatus']==3:state['status']='removed'
        elif state.get('failure'):state['status']='needs_attention'
        elif ad['JobStatus']==1:state['status']='queued_for_resume' if state['formal_started'] else 'queued'
        elif ad['JobStatus']==2:
            state['status']='formal_training' if state['formal_started'] else 'preparing_environment'
    else:
        history=json.loads(command(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
        if history:
            ad=history[0];state['scheduler']=ad
            if ad.get('JobStatus')==4 and ad.get('ExitCode')==0 and not ad.get('ExitBySignal',False):collect(state)
            else:state.update(status='needs_attention',failure='Job ended without successful completion',history=ad)
        else:state['status']='awaiting_scheduler_history'
    state['updated_at']=time.time();write(ROOT/'state.json',state)
    return state['status'] not in ('complete','needs_attention','removed','stopped')


def monitor():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (ROOT/'monitor.pid').write_text(str(os.getpid())+'\n')
        while True:
            try:
                if not tick():return
            except Exception:
                (ROOT/'monitor-error.txt').write_text(traceback.format_exc())
                raise
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('submit','monitor','tick'));a=p.parse_args()
    if a.action=='submit':
        state=submit();print(json.dumps({k:state[k] for k in ('job','run_id','status','wandb_url','estimated_pure_training_hours')},indent=2))
    elif a.action=='monitor':monitor()
    else:tick()
