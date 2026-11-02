"""Durable two-run workflow: fresh16K data, GPU gate, publish, formal training.

All job mutations have a persistent intent and unique role marker. Errors stop
continuation; rerunning the monitor never silently resubmits a failed job.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[2]
KEY='lookback-16k-10b-20261102'
STEPS=305176
WHEEL='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
RUNTIME='mhc-official-runtime-7546767.tar.gz'
COMPAT='cuda-compat-13-1_590.48.01-1_amd64.deb'
CONTAINER='file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif'
ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,LookbackKey,LookbackRole,Args,TransferInput,SuccessCheckpointExitCode,TransferCheckpoint,LookbackVariant,LookbackRunID'


def command(args,cwd=ROOT,timeout=60):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=timeout).strip()


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def save(state):write(ROOT/'state.json',state)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def stopped():assert not (ROOT/'STOPPED.json').exists(),'Stopped by user'


def status(ident):
    ads=json.loads(command(['condor_q',ident,'-json','-attributes',ATTRS]) or '[]')
    if not ads:ads=json.loads(command(['condor_history',ident,'-limit','1','-json','-attributes',ATTRS]) or '[]')
    if not ads:return None
    a=ads[0];assert a['LookbackKey']==KEY
    if a['JobStatus'] in (3,5) or a['JobStatus']==4 and a.get('ExitCode')!=0:
        raise RuntimeError(f'Job failed or held: {a}')
    return a


def note(message):
    p=Path('/home/yzhong239/AGENTS.md');s=p.read_text()
    heading='## 新授权：16K vanilla与2K-chunk Shared Lookback各10B（2026-11-02）\n'
    if heading in s:p.write_text(s.replace(heading,heading+'\n'+message+'\n',1))


def collect_data(state,prefix=False):
    ident=state['gate_data_job' if prefix else 'data_job']
    label='gate-data' if prefix else 'data'
    with tarfile.open(ROOT/'results'/f'{label}-{ident}.tar') as tf:
        result=json.load(tf.extractfile('results/data.json'))
        archive=json.load(tf.extractfile('results/data-archive.json'))
    assert result['complete'] and result['sequence_length']==16384 and result['batch_tokens']==32768
    assert result['training_steps']==(128 if prefix else STEPS)
    assert result['training_tokens']==result['training_steps']*32768
    assert result['files']['dev']['tokens']==131072 and result['files']['test']['tokens']==2097152
    if prefix:
        path=ROOT/'results'/f'gate-data-{ident}-cache.tar'
        assert path.stat().st_size==archive['bytes'] and sha(path)==archive['sha256']
        # This archive is ~13MB. Verify every split before any GPU launch.
        with tarfile.open(path) as tf:
            assert json.load(tf.extractfile('data/manifest.json'))==result
            for split,meta in result['files'].items():
                data=tf.extractfile(f'data/{split}.bin').read()
                assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        state['gate_data_ready']=True;state['gate_data_manifest']=result;state['gate_data_archive']=archive
    else:
        path=Path('/staging/y/yzhong239')/state['data_uri'].rsplit('/',1)[-1]
        assert path.stat().st_size==archive['bytes']
        state['data_ready']=True;state['data_manifest']=result;state['data_archive']=archive
    write(ROOT/(label+'.json'),result);save(state)


def common_submit(role,exe,disk):
    return f'''universe = vanilla
container_image = {CONTAINER}
executable = {exe}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_disk = {disk}GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
+LookbackKey = "{KEY}"
+LookbackRole = "{role}"
+JobBatchName = "nanochat-lookback16k-{role}"
'''


def inputs(bundle,source,data,formal=False):
    hashes=source['hashes'];write(bundle/'source.json',source)
    (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
    names=[str(p.relative_to(ROOT)) for p in (bundle/WHEEL,bundle/'source.tar',bundle/'source.json',bundle/'inputs.sha256')]
    names += [data,'osdf:///chtc/staging/y/yzhong239/'+RUNTIME,
        'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/'+COMPAT]
    if formal:names.append('/home/yzhong239/.netrc')
    return ','.join(names)


def submit_gate(state):
    stopped();assert state['gate_data_ready'] and not state.get('gate_job')
    intent=ROOT/'gate-intent.json';assert not intent.exists(),'Recover existing gate intent before any submission'
    assert not command(['git','status','--porcelain'],REPO)
    commit=command(['git','rev-parse','HEAD'],REPO)
    bundle=ROOT/'bundles';bundle.mkdir(exist_ok=True)
    command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
    shutil.copy2(Path('/home/yzhong239/mhc-native-20261031/bundles')/WHEEL,bundle/WHEEL)
    shutil.copy2(ROOT/'results'/f'gate-data-{state["gate_data_job"]}-cache.tar',bundle/'gate-data.tar')
    shutil.copy2(REPO/'research/lookback/validate.sh',ROOT/'validate.sh');(ROOT/'validate.sh').chmod(0o755)
    old=json.loads(Path('/home/yzhong239/mhc-native-20261031/bundles/source.json').read_text())
    hashes={k:old['hashes'][k] for k in (WHEEL,RUNTIME,COMPAT)}
    hashes.update({'source.tar':sha(bundle/'source.tar'),'gate-data.tar':sha(bundle/'gate-data.tar')})
    assert sha(bundle/WHEEL)==hashes[WHEEL]
    source=dict(commit=commit,hashes=hashes,purpose='Lookback16K implementation gate',formal_training=False,
        sequence_length=16384,chunk_size=2048,chunk_attention='full_causal',
        launcher_sha256=sha(ROOT/'validate.sh'))
    sub=common_submit('gate','validate.sh',45)+f'''transfer_input_files = {inputs(bundle,source,'bundles/gate-data.tar')}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/gate-$(Cluster).$(Process).tar"
log = logs/gate.$(Cluster).$(Process).log
output = logs/gate.$(Cluster).$(Process).out
error = logs/gate.$(Cluster).$(Process).err
queue 1
'''
    (ROOT/'gate.sub').write_text(sub);command(['condor_submit','-dry-run','gate-dryrun.ad','gate.sub'])
    write(intent,dict(source=source,time=time.time()));stopped()
    output=command(['condor_submit','-terse','gate.sub']);ident=output.split()[0];assert ident.endswith('.0'),output
    state.update(gate_job=ident,gate_source=source,status='gpu_validation_pending',reviewed_head=commit);save(state)
    ads=json.loads(command(['condor_q',ident,'-json','-attributes',ATTRS]));assert len(ads)==1
    assert '.netrc' not in ads[0]['TransferInput'] and CONTAINER in ads[0]['TransferInput']
    write(ROOT/'gate-submission-verified.json',ads)
    note(f'GPU gate{ident}已实际提交，source{commit}；12项检查、16K同卡两组吞吐及各25→exit85→新进程50恢复，诊断无W&B/凭据/权重回传。正式两个10B尚未提交；完整16K数据及GPU gate均通过后自动发布并提交，不重复提交。')


def collect_gate(state):
    with tarfile.open(ROOT/'results'/f'gate-{state["gate_job"]}.tar') as tf:
        assert not any(m.name.endswith(('.pt','.wandb')) for m in tf)
        result=json.load(tf.extractfile('results/lookback-gate.json'))
        evidence={Path(m.name).stem:json.load(tf.extractfile(m)) for m in tf
            if m.isfile() and m.name.startswith('results/lookback-checks/') and m.name.endswith('.json')}
    assert result['complete'] and result['source']==state['gate_source']
    assert result['checks']==dict(tests=12,failures=0,errors=0,skipped=0)
    assert result['logging_backend']=='local' and result['selected_microbatch'] in (1,2)
    assert result['data_manifest']==state['gate_data_manifest']
    assert set(result['resumes'])=={'baseline','lookback'} and len(result['trials'])==4
    for trial in result['trials']:
        assert trial['causality_passed'] and trial['status']=='fits' and trial['sequence_length']==16384
        assert len(trial['step_seconds'])==30 and trial['peak_reserved_bytes']<int(.9*22000*1024**2)
    for variant,item in result['resumes'].items():
        r=item['summary'];assert r['complete'] and r['steps']==50 and r['training_tokens']==50*32768
        assert r['source']==state['gate_source'] and r['variant']==variant
        assert item['config']['protocol']['schedule_steps']==STEPS
    result.update(job=state['gate_job'],evidence=evidence)
    write(ROOT/'validation.json',result)
    state.update(gate_ready=True,selected_microbatch=result['selected_microbatch'],profile_summary=result['summary']);save(state)


def publish_and_submit(state):
    stopped();assert state['gate_ready'] and state['data_ready']
    data=state['data_manifest'];prefix=state['gate_data_manifest']
    assert data['first_128_steps_sha256']==prefix['files']['train']['sha256']
    for split in ('dev','test'):assert data['files'][split]==prefix['files'][split]
    assert data['tokenizer']==prefix['tokenizer']
    write(ROOT/'data-prefix-verified.json',dict(complete=True,full_job=state['data_job'],prefix_job=state['gate_data_job'],
        prefix_train_sha256=data['first_128_steps_sha256'],dev_sha256=data['files']['dev']['sha256'],test_sha256=data['files']['test']['sha256']))
    if not state.get('published_commit'):
        assert command(['git','rev-parse','HEAD'],REPO)==state['reviewed_head']
        assert not command(['git','status','--porcelain'],REPO)
        shutil.copy2(ROOT/'validation.json',REPO/'research/lookback/validation.json')
        shutil.copy2(ROOT/'data-prefix-verified.json',REPO/'research/lookback/data-validation.json')
        command(['git','add','research/lookback/validation.json','research/lookback/data-validation.json'],REPO)
        command(['git','commit','-m','Validate16K Lookback causality gradients performance and recovery'],REPO)
        state['report_commit']=command(['git','rev-parse','HEAD'],REPO);save(state)
        state['published_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']();save(state)
    stopped();intent=ROOT/'training-intent.json';assert not intent.exists(),'Recover existing formal intent; do not duplicate'
    active=json.loads(command(['condor_q','yzhong239','-constraint',f'LookbackKey == "{KEY}" && LookbackRole == "train"','-json','-attributes','ClusterId,ProcId']) or '[]')
    assert not active,active
    bundle=ROOT/'training-bundles';bundle.mkdir(exist_ok=True)
    for name in (WHEEL,'source.tar'):shutil.copy2(ROOT/'bundles'/name,bundle/name)
    assert sha(bundle/'source.tar')==state['gate_source']['hashes']['source.tar']
    shutil.copy2(REPO/'research/lookback/train.sh',ROOT/'train.sh');(ROOT/'train.sh').chmod(0o755)
    hashes={k:v for k,v in state['gate_source']['hashes'].items() if k!='gate-data.tar'}
    data_name=state['data_uri'].rsplit('/',1)[-1];hashes[data_name]=state['data_archive']['sha256']
    source=dict(commit=state['gate_source']['commit'],hashes=hashes,gate_job=state['gate_job'],gate_verified=True,
        report_commit=state['published_commit'],purpose='User-authorized two formal16K10B runs',formal_training=True,
        sequence_length=16384,chunk_size=2048,chunk_attention='full_causal',steps=STEPS,training_tokens=STEPS*32768,
        launcher_sha256=sha(ROOT/'train.sh'),profile_reference=state['profile_summary'])
    ids={v:uuid.uuid4().hex[:8] for v in ('baseline','lookback')};micro=state['selected_microbatch']
    sub=common_submit('train','train.sh',80)+f'''arguments = $(variant) $(runid) {STEPS} {micro}
environment = "NANOCHAT_DATA_ARCHIVE={data_name} NANOCHAT_CHECKPOINT_SECONDS=1800"
encrypt_input_files = /home/yzhong239/.netrc
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
transfer_input_files = {inputs(bundle,source,state['data_uri'],True)}
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/$(variant)-$(Cluster).$(Process).tar;checkpoints.tar=results/$(variant)-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+LookbackVariant = "$(variant)"
+LookbackRunID = "$(runid)"
queue variant,runid from (
'''+''.join(f'{v} {ids[v]}\n' for v in ids)+')\n'
    (ROOT/'train.sub').write_text(sub);command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'])
    write(intent,dict(source=source,run_ids=ids,time=time.time()));stopped()
    output=command(['condor_submit','-terse','train.sub']);cluster=output.split('.')[0];assert cluster.isdigit(),output
    state.update(training_jobs={v:dict(job=f'{cluster}.{i}',run_id=ids[v]) for i,v in enumerate(ids)},
        training_source=source,status='training_submitted',training_submitted_at=time.time());save(state)
    ads=json.loads(command(['condor_q',cluster,'-json','-attributes',ATTRS]));assert len(ads)==2
    for a in ads:
        assert a['SuccessCheckpointExitCode']==85 and a['TransferCheckpoint']=='checkpoints.tar,results.tar'
        assert CONTAINER in a['TransferInput'] and data_name in a['TransferInput']
    write(ROOT/'training-submission-verified.json',ads)
    note(f'GPU gate{state["gate_job"]}及16K数据prefix核验已通过，实现/报告已推送{state["published_commit"]}。正式两组各10B已提交：'+', '.join(f'{v}={i["job"]}/W&B{i["run_id"]}' for v,i in state['training_jobs'].items())+f'；micro{micro}/accum{2//micro}。提交不等于已开跑，继续监控，不重复提交。')


def collect_training(state,variant,info):
    with tarfile.open(ROOT/'results'/f'{variant}-{info["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/train/summary.json'))
        config=json.load(tf.extractfile('results/train/config.json'))
    assert result['complete'] and result['source']==state['training_source'] and result['run_id']==info['run_id']
    assert result['steps']==STEPS and result['training_tokens']==STEPS*32768 and result['variant']==variant
    assert result['test_evaluated'] and result['test_tokens']==2097152 and result['dev_tokens']==131072
    assert config['logging_backend']=='wandb' and config['data_manifest']==state['data_manifest']
    assert config['model']==json.loads((ROOT/'validation.json').read_text())['resumes'][variant]['config']['model']
    assert config['protocol']==dict(steps=STEPS,schedule_steps=STEPS,batch_tokens=32768,
        microbatch=state['selected_microbatch'],grad_accum=2//state['selected_microbatch'],seed=42,
        attention_backend='sdpa_cudnn',matched_backbone_init=True,evaluate_test=True)
    with tarfile.open(ROOT/'results'/f'{variant}-{info["job"]}-checkpoints.tar') as tf:
        assert {m.name for m in tf if m.isfile()}=={'checkpoints/latest.pt','checkpoints/best.pt'}
        for name,meta in result['checkpoint_files'].items():assert tf.getmember('checkpoints/'+name).size==meta['bytes']
    import wandb
    run=wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+info['run_id'])
    assert run.group==KEY and run.config['source']==state['training_source']
    if run.state!='finished':raise ValueError('Waiting for W&B finished state')
    for key in ('steps','training_tokens','final_dev_ce','final_test_ce','best_selected_test_ce'):
        if not math.isclose(run.summary.get(key,float('nan')),result[key],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('Waiting for W&B summary '+key)
    result.update(wandb_verified=True,checkpoint_sizes_verified=True,config=config)
    write(ROOT/'results'/f'{variant}-verified.json',result);return result


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state['status'] in ('complete','needs_attention','stopped_by_user'):return
        state['monitor_pid']=os.getpid();save(state)
        errors=0
        while not (ROOT/'STOPPED.json').exists():
            try:
                state['last_poll']=time.time();state['last_jobs']={}
                for flag,key,prefix in (('gate_data_ready','gate_data_job',True),('data_ready','data_job',False)):
                    if not state.get(flag):
                        row=status(state[key]);state['last_jobs'][key]=row
                        if row and row['JobStatus']==4:collect_data(state,prefix)
                if state.get('gate_data_ready') and not state.get('gate_job'):submit_gate(state)
                if state.get('gate_job') and not state.get('gate_ready'):
                    row=status(state['gate_job']);state['last_jobs']['gate']=row
                    if row and row['JobStatus']==4:collect_gate(state)
                if state.get('gate_ready') and state.get('data_ready') and not state['training_jobs']:
                    publish_and_submit(state)
                state.setdefault('training_results',{})
                for variant,info in state['training_jobs'].items():
                    if variant in state['training_results']:continue
                    row=status(info['job']);state['last_jobs'][variant]=row
                    if row and row['JobStatus']==4:state['training_results'][variant]=collect_training(state,variant,info)
                if len(state['training_results'])==2:
                    state['status']='complete';state['completed_at']=time.time();save(state);return
                errors=0;state['monitor_errors']=0;save(state)
            except Exception as exc:
                errors+=1;state.update(last_error=repr(exc),monitor_errors=errors,last_poll=time.time())
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';save(state);return
                save(state)
            time.sleep(30)
        state.update(status='stopped_by_user',automatic_continuation=False);save(state)


if __name__=='__main__':main()
