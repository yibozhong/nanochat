"""Launch the authorized single absolute-gap 10B run from the verified GPU bundle."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
GATE=Path('/home/yzhong239/jensen-abs-gate-20261103/retry-2')
KEY='jensen-abs-10b-20261103'
spec=importlib.util.spec_from_file_location('moment_workflow',HERE.parent/'moments/workflow.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.KEY=KEY


def submit(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():print((root/'state.json').read_text());return
        assert not (root/'submission-intent.json').exists(),'Recover existing intent instead of resubmitting'
        assert not json.loads(base.command(['condor_q','yzhong239','-constraint',f'MomentKey == "{KEY}"','-json']) or '[]')
        assert not base.command(['git','status','--porcelain'])
        gate=json.loads((GATE/'state.json').read_text())
        pub=json.loads((GATE/'published.json').read_text())
        result=json.loads((GATE/'collected/abs-jensen-gate.json').read_text())
        assert gate['status']=='complete' and pub['remote_verified']
        assert result['complete'] and result['source']==gate['source']
        assert result['checks']==dict(tests=28,failures=0,errors=0,skipped=0)
        assert result['restart']['exact_steps_1_to_50'] and result['restart']['summary']['complete']
        assert result['restart']['config']['model']['attention_kernel']=='fla_triton'
        assert result['restart']['config']['model']['attention_statistic']=='jensen_abs'
        commit=gate['source']['commit'];assert pub['gate_commit']==commit
        assert not base.command(['git','diff','--name-only',commit,'HEAD','--','nanochat','scripts',
            'research/connections/runtime','research/connections/toolchain','research/connections/triton-moments/pretrain/train.sh'])
        ready=json.loads(base.READY.read_text());assert ready['ready'] and ready['sequence_length']==2048
        for name in ('training-bundles','logs','results'):(root/name).mkdir(exist_ok=True)
        bundle=root/'training-bundles'
        for name in (base.WHEEL,'source.tar'):(bundle/name).symlink_to(GATE/'bundles'/name)
        profiles={}
        for mode in ('none','jensen_silu','jensen_abs'):
            rows=[r for r in result['profiles'] if r['mode']==mode];assert len(rows)==2
            profiles[mode]=dict(mean_seconds=sum(r['mean_seconds'] for r in rows)/2,
                peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
                parameters=rows[0]['parameters'],gpu=rows[0]['gpu'])
        for r in profiles.values():
            for kind in ('allocated','reserved'):
                r['additional_peak_'+kind+'_bytes']=r['peak_'+kind+'_bytes']-profiles['none']['peak_'+kind+'_bytes']
        hashes={k:v for k,v in gate['source']['hashes'].items() if k!='gate-data.tar'}
        hashes[ready['data_uri'].rsplit('/',1)[-1]]=ready['archive']['sha256']
        source=dict(commit=commit,hashes=hashes,gate_job=gate['job'],formal_training=True,
            purpose=KEY,steps=305176,training_tokens=10000007168,profile_reference=profiles,
            data_uri=ready['data_uri'],attention_statistic='jensen_abs',attention_kernel='fla_triton',alpha=1,phi='abs')
        launcher=base.command(['git','show',commit+':research/connections/triton-moments/pretrain/train.sh'])+'\n'
        assert '--attention-kernel fla_triton' in launcher and '--wandb-group triton-moments-10b-20261103' in launcher
        (root/'train.sh').write_text(launcher);(root/'train.sh').chmod(0o755)
        source['launcher_sha256']=hashlib.sha256((root/'train.sh').read_bytes()).hexdigest()
        base.inputs(bundle,source)
        run_id=uuid.uuid4().hex[:8]
        # Same resource/transfer/checkpoint policy as the three existing matched runs.
        template=Path('/home/yzhong239/triton-moments-10b-20261103/train.sub').read_text()
        assert ready['data_uri'] in template
        text=template.split('queue mode,runid from (')[0].replace('triton-moments-10b-20261103',KEY)
        text+=f'queue mode,runid from (\njensen_abs {run_id}\n)\n'
        (root/'train.sub').write_text(text)
        base.command(['bash','-n','train.sh'],root)
        base.command(['condor_submit','-dry-run','dryrun.ad','train.sub'],root)
        base.write(root/'submission-intent.json',dict(source=source,run_id=run_id,time=time.time()))
        job=base.command(['condor_submit','-terse','train.sub'],root).split()[0]
        state=dict(status='submitted',job=job,run_id=run_id,source=source,submitted_at=time.time(),formal_training=True)
        base.write(root/'state.json',state)
        ads=json.loads(base.command(['condor_q',job,'-json','-attributes',
            'ClusterId,ProcId,JobStatus,Args,TransferInput,ContainerImage,SuccessCheckpointExitCode,TransferCheckpoint,RequestGpus,RequestMemory,RequestDisk,MomentKey,MomentRole,MomentRunID']))
        assert len(ads)==1
        ad=ads[0]
        assert ad['Args']==f'baseline {run_id} 305176 jensen_abs'
        assert ad['RequestGpus']==1 and ad['SuccessCheckpointExitCode']==85
        assert 'checkpoints.tar' in ad['TransferCheckpoint'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        base.write(root/'submission-verified.json',ads)
        print(json.dumps(dict(job=job,run_id=run_id,steps=305176,status=ad['JobStatus']),indent=2))


def monitor(root):
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('training_started','needs_attention','stopped_by_user'):return
            try:
                info=base.job_info(state['job']);state['last_job']=info
                if info and info['JobStatus'] in (3,5):raise RuntimeError(str(info))
                log=root/'logs'/f'train.{state["job"]}.out'
                if log.exists():
                    for line in log.read_text(errors='replace').splitlines():
                        try:row=json.loads(line)
                        except (ValueError,TypeError):continue
                        if isinstance(row,dict) and row.get('step',0)>=1 and isinstance(row.get('loss'),(int,float)) and math.isfinite(row['loss']):
                            base.write(root/'first-step.json',dict(job=state['job'],run_id=state['run_id'],evidence=row,source=state['source']))
                            state['status']='training_started';break
                if info and info['JobStatus']==4 and state['status']!='training_started':raise RuntimeError(str(info))
                state['last_poll']=time.time();base.write(root/'state.json',state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));base.write(root/'state.json',state);raise
            if state['status']=='training_started':return
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('action',choices=('submit','monitor'))
    a=p.parse_args();globals()[a.action](a.root.resolve())
