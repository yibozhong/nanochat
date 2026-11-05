"""TokenRes GPU gate, publication and one authorized 10B run."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile
import time
import uuid

REPO=Path(__file__).resolve().parents[3]
KEY='tokenres-10b-20261104'
PRIOR=Path('/home/yzhong239/nanochat-runtime-assets')
READY=Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json')
MODES=('tokenres',)
WHEEL='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
RUNTIME='osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz'
COMPAT='https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb'


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=90).strip()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def inputs(bundle,source):
    write(bundle/'source.json',source)
    hashes=dict(source['hashes'])
    hashes['source.json']=hashlib.sha256((bundle/'source.json').read_bytes()).hexdigest()
    (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))


def common_submit():
    # Reuse the audited resource policy, without old experiment-specific attrs.
    text=(PRIOR/'gate.sub').read_text()
    start=text.index('should_transfer_files =')
    end=text.index('transfer_input_files =')
    return 'universe = vanilla\ncontainer_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif\n'+text[start:end]


def submit(root, experiment_dir=None):
    experiment_dir = experiment_dir or REPO/'research/connections/tokenres'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():print((root/'state.json').read_text());return
        assert not (root/'submission-intent.json').exists(),'Recover submission intent before retrying'
        assert not command(['git','status','--porcelain'])
        assert not json.loads(command(['condor_q','yzhong239','-constraint',f'TokenResKey == "{KEY}"','-json']) or '[]')
        for name in ('bundles','logs','results'):(root/name).mkdir(exist_ok=True)
        bundle=root/'bundles';ready=json.loads(READY.read_text());assert ready['ready'] and ready['sequence_length']==2048
        for name,path in [(WHEEL,PRIOR/'bundles'/WHEEL),('gate-data.tar',Path(ready['gate_archive']['path']))]:
            (bundle/name).symlink_to(path)
        commit=command(['git','rev-parse','HEAD'])
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit])
        old=json.loads((PRIOR/'state.json').read_text())['source']['hashes']
        hashes={k:v for k,v in old.items() if k not in ('source.tar','gate-data.tar')}
        hashes.update({'source.tar':hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest(),
                       'gate-data.tar':ready['gate_archive']['sha256']})
        source=dict(commit=commit,hashes=hashes,formal_training=False,purpose=KEY+' correctness/profile/restart')
        inputs(bundle,source)
        shutil.copy2(experiment_dir/'validate.sh',root/'validate.sh')
        files=[str(p.relative_to(root)) for p in sorted(bundle.iterdir())]+[RUNTIME,COMPAT]
        text=common_submit()+f'''executable = validate.sh
transfer_input_files = {','.join(files)}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/gate-$(Cluster).$(Process).tar"
log = logs/gate.$(Cluster).$(Process).log
output = logs/gate.$(Cluster).$(Process).out
error = logs/gate.$(Cluster).$(Process).err
+JobBatchName = "{KEY}-gate"
+TokenResKey = "{KEY}"
+TokenResRole = "gate"
queue 1
'''
        (root/'gate.sub').write_text(text)
        command(['condor_submit','-dry-run','dryrun.ad','gate.sub'],root)
        write(root/'submission-intent.json',dict(source=source,time=time.time()))
        job=command(['condor_submit','-terse','gate.sub'],root).split()[0]
        state=dict(status='gate_submitted',job=job,source=source,submitted_at=time.time())
        write(root/'state.json',state);print(json.dumps(state,indent=2))


def job_info(job):
    attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,Args,TokenResKey,TokenResRole'
    rows=json.loads(command(['condor_q',job,'-json','-attributes',attrs]) or '[]')
    if not rows:
        rows=json.loads(command(['condor_history',job,'-limit','1','-scanlimit','100000','-json','-attributes',attrs]) or '[]')
    if rows:assert rows[0]['TokenResKey']==KEY
    return rows[0] if rows else None


def audit_publish(root,state):
    with tarfile.open(root/'results'/f'gate-{state["job"]}.tar') as tf:
        result=json.load(tf.extractfile('results/gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=32,failures=0,errors=0,skipped=0)
        assert not result['formal_training'] and not result['weights_returned']
        assert len(result['profiles'])==4 and len({p['gpu'] for p in result['profiles']})==1
        for p in result['profiles']:
            assert p['attention_backend']=='fa3' and p['attention_kernel']=='original'
            assert p['mode'] in ('baseline','tokenres')
            assert p['parameters']==(58720256 if p['mode']=='baseline' else 67108864)
            assert len(p['samples_seconds'])==30 and all(math.isfinite(x) and x>0 for x in p['samples_seconds'])
            assert p['peak_reserved_bytes']<.9*22000*1024**2 and math.isfinite(p['dev_ce'])
            if p['mode']=='tokenres':
                assert p['extended_probe']['steps']==128 and p['extended_probe']['training_tokens']==4194304
                for diagnosis in (p['stability_before'],p['stability_after'],p['extended_probe']['stability']):
                    assert len(diagnosis['layers'])==16 and math.isfinite(diagnosis['ce'])
                    assert all(0<x['output_rms']<100 for x in diagnosis['layers'])
                    assert all(math.isfinite(v) for v in diagnosis['parameter_gradient_norms'].values())
        resume=result['resume'];assert resume['exact_steps_1_to_50'] and resume['summary']['complete']
        assert resume['summary']['steps']==50 and resume['config']['logging_backend']=='local'
        assert resume['config']['model']['tokenres'] and resume['config']['model']['tokenres_window']==128
        assert resume['config']['parameters']==67108864 and resume['config']['protocol']['schedule_steps']==305176
        probe=json.load(tf.extractfile('results/backend-probe.json'))
        assert probe['backend']=='fa3' and probe['revision']=='de87b9b5af06dd9984df595bef90b2eba44b181a'
        assert probe['windows']==[[127,0],[-1,0]]
        artifacts=json.load(tf.extractfile('results/fa3-artifacts.json'))
        assert artifacts==json.loads((REPO/'research/connections/runtime/fa3-artifacts.json').read_text())
        for member in tf.getmembers():
            assert not member.name.endswith(('.pt','.wandb'))
            if not member.isfile():continue
            path=Path(member.name);assert not path.is_absolute() and path.parts[0]=='results' and '..' not in path.parts
            dest=root/'collected'/Path(*path.parts[1:]);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(tf.extractfile(member).read())
    summary={}
    for mode in ('baseline','tokenres'):
        rows=[p for p in result['profiles'] if p['mode']==mode];assert len(rows)==2
        summary[mode]=dict(mean_seconds=sum(p['mean_seconds'] for p in rows)/2,
            peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in rows),
            peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in rows),parameters=rows[0]['parameters'])
    for row in summary.values():
        row['time_relative_to_baseline']=row['mean_seconds']/summary['baseline']['mean_seconds']
    result.update(job=state['job'],summary=summary,backend_probe=probe,fa3_artifacts=artifacts,
        comparison_note='No new formal baseline; prior full 10B baseline uses FLA. Backbone output projections require nonzero initialization for the no-identity architecture. Short profiles establish numerical operation, not quality.')
    write(root/'validation.json',result)
    assert not command(['git','status','--porcelain'])
    assert not command(['git','diff','--name-only',state['source']['commit'],'HEAD','--','nanochat','scripts','tests',
        'research/connections/tokenres','research/connections/runtime','research/connections/toolchain'])
    report=REPO/'research/connections/tokenres/validation.json';write(report,result)
    command(['git','add',str(report)]);command(['git','commit','-m','Validate TokenRes causal routing, throughput and checkpoint recovery'])
    head=command(['git','rev-parse','HEAD'])
    try:runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
    except Exception:
        assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==head
    write(root/'published.json',dict(commit=head,gate_commit=state['source']['commit'],job=state['job'],remote_verified=True))
    return summary


def submit_training(root,state,summary,experiment_dir=None):
    experiment_dir = experiment_dir or REPO/'research/connections/tokenres'
    assert (root/'published.json').exists() and not (root/'STOPPED.json').exists()
    with (root/'training-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not state.get('training_jobs')
        assert not (root/'training-intent.json').exists(),'Recover interrupted formal submission, do not duplicate'
        ready=json.loads(READY.read_text());assert ready['ready']
        bundle=root/'training-bundles';bundle.mkdir()
        for name in (WHEEL,'source.tar'):(bundle/name).symlink_to(root/'bundles'/name)
        hashes={k:v for k,v in state['source']['hashes'].items() if k!='gate-data.tar'}
        hashes[ready['data_uri'].rsplit('/',1)[-1]]=ready['archive']['sha256']
        source=dict(commit=state['source']['commit'],hashes=hashes,gate_job=state['job'],formal_training=True,
            purpose=KEY,steps=305176,training_tokens=10000007168,profile_reference=summary,data_uri=ready['data_uri'])
        inputs(bundle,source)
        shutil.copy2(experiment_dir/'train.sh',root/'train.sh')
        run_ids={mode:uuid.uuid4().hex[:8] for mode in MODES}
        files=[str(p.relative_to(root)) for p in sorted(bundle.iterdir())]+['/home/yzhong239/.netrc',ready['data_uri'],RUNTIME,COMPAT]
        text=common_submit().replace('request_disk = 40GB','request_disk = 80GB')+f'''executable = train.sh
arguments = baseline $(runid) 305176 $(mode)
environment = "NANOCHAT_DATA_ARCHIVE={ready['data_uri'].rsplit('/',1)[-1]} NANOCHAT_CHECKPOINT_SECONDS=1800"
encrypt_input_files = /home/yzhong239/.netrc
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
transfer_input_files = {','.join(files)}
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/$(mode)-$(Cluster).$(Process).tar;checkpoints.tar=results/$(mode)-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+JobBatchName = "{KEY}"
+TokenResKey = "{KEY}"
+TokenResRole = "$(mode)"
+TokenResRunID = "$(runid)"
queue mode,runid from (
{chr(10).join(mode+' '+run_ids[mode] for mode in MODES)}
)
'''
        (root/'train.sub').write_text(text);command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'],root)
        write(root/'training-intent.json',dict(source=source,run_ids=run_ids,time=time.time()))
        cluster=command(['condor_submit','-terse','train.sub'],root).split('.')[0];assert cluster.isdigit()
        state.update(status='training_submitted',training_source=source,
            training_jobs={mode:dict(job=f'{cluster}.{i}',run_id=run_ids[mode]) for i,mode in enumerate(MODES)})
        write(root/'state.json',state)
        ads=json.loads(command(['condor_q',cluster,'-json','-attributes',
            'ClusterId,ProcId,JobStatus,Args,TransferInput,ContainerImage,SuccessCheckpointExitCode,TransferCheckpoint,RequestGpus,RequestMemory,RequestDisk,TokenResKey,TokenResRole,TokenResRunID']))
        assert len(ads)==len(MODES)
        for ad in ads:
            assert ad['RequestGpus']==1 and ad['TokenResRole'] in MODES
            assert ad['Args']==f"baseline {run_ids[ad['TokenResRole']]} 305176 {ad['TokenResRole']}"
            assert ad['SuccessCheckpointExitCode']==85 and 'checkpoints.tar' in ad['TransferCheckpoint']
            assert 'nanochat-uv-v4.sif' in ad['TransferInput']
        write(root/'training-submission-verified.json',ads)


def monitor(root):
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('training_started','needs_attention','stopped_by_user'):return
            try:
                if not state.get('training_jobs'):
                    info=job_info(state['job']);state['last_gate']=info
                    if info and info['JobStatus']==4:
                        assert info.get('ExitCode')==0,info
                        summary=audit_publish(root,state)
                        submit_training(root,state,summary)
                    elif info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                else:
                    first=state.setdefault('first_steps',{})
                    for mode,spec in state['training_jobs'].items():
                        if mode in first:continue
                        info=job_info(spec['job']);spec['last_job']=info
                        if info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                        log=root/'logs'/f'train.{spec["job"]}.out'
                        if log.exists():
                            for line in log.read_text(errors='replace').splitlines():
                                try:row=json.loads(line)
                                except (ValueError,TypeError):continue
                                if isinstance(row,dict) and row.get('step',0)>=1 and isinstance(row.get('loss'),(int,float)) and math.isfinite(row['loss']):
                                    first[mode]=dict(job=spec['job'],run_id=spec['run_id'],evidence=row,source=state['training_source']);break
                        if mode not in first and info and info['JobStatus']==4:raise RuntimeError(str(info))
                    if len(first)==len(MODES):
                        state['status']='training_started';write(root/'first-steps.json',first)
                state['last_poll']=time.time();write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));write(root/'state.json',state);raise
            if state['status']=='training_started':return
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('action',choices=('submit','monitor'))
    args=parser.parse_args();globals()[args.action](args.root.resolve())
