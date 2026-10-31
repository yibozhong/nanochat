"""Track and audit exactly the three submitted formal runs; never submit more."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[4]
KEY='mhc-native-pretrain-20261031'
ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,NativePretrainKey,MHCVariant,MHCRunID'


def command(args,timeout=60):
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=timeout).strip()


def save(state):
    p=ROOT/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'state.json')


def job(info,submitted):
    rows=json.loads(command(['condor_q',info['job'],'-json','-attributes',ATTRS]) or '[]')
    if not rows:
        rows=json.loads(command(['condor_history',info['job'],'-match','1',
            '-since',f'CompletionDate < {int(submitted)-60}','-json','-attributes',ATTRS]) or '[]')
    if rows:
        assert rows[0]['NativePretrainKey']==KEY and rows[0]['MHCRunID']==info['run_id']
        return rows[0]


def collect(state,variant,info):
    archive_path=ROOT/'results'/f'{variant}-{info["job"]}.tar'
    with tarfile.open(archive_path) as archive:
        result=json.load(archive.extractfile('results/train/summary.json'))
        config=json.load(archive.extractfile('results/train/config.json'))
        offline=[m for m in archive if m.isfile() and '/offline-run-' in m.name]
        if offline:
            dest=ROOT/'results'/(variant+'-offline')
            for member in offline:
                target=dest/member.name
                assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            for directory in sorted((dest/'results/wandb').glob('offline-run-*')):
                marker=directory/'.native-audit-synced'
                if not marker.exists():
                    command(['wandb','sync',str(directory)],timeout=300)
                    marker.touch()
    assert result['complete'] and result['source']==state['source']
    assert result['variant']==variant and result['run_id']==info['run_id']
    assert result['steps']==state['expected_steps'] and result['training_tokens']==state['expected_tokens']
    assert result['test_evaluated'] and result['test_tokens']==2097152 and result['dev_tokens']==131072
    assert config['logging_backend']=='wandb' and config['source']==state['source']
    gate=json.loads((REPO/'research/connections/native/validation.json').read_text())
    assert config['model']==gate['trials'][variant]['config']['model']
    assert config['optimizer_groups']==gate['trials'][variant]['config']['optimizer_groups']
    expected=dict(steps=state['expected_steps'],schedule_steps=state['expected_steps'],batch_tokens=32768,
        microbatch=16,grad_accum=1,seed=42,attention_backend='auto',matched_backbone_init=True,evaluate_test=True)
    assert config['protocol']==expected
    assert config['checkpoint_transport']['upload_interval_seconds']==1800
    data=json.loads(Path('/home/yzhong239/mhc-pretrain-20261030/data.json').read_text())
    assert config['data_manifest']==data
    checkpoint=ROOT/'results'/f'{variant}-{info["job"]}-checkpoints.tar'
    with tarfile.open(checkpoint) as archive:
        assert {m.name for m in archive if m.isfile()}=={'checkpoints/latest.pt','checkpoints/best.pt'}
        for name,meta in result['checkpoint_files'].items():
            assert archive.getmember('checkpoints/'+name).size==meta['bytes']
    import wandb
    run=wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+info['run_id'])
    assert run.group==KEY and run.config['model']['architecture']=='native'
    assert run.config['model']['variant']==variant and run.config['source']==state['source']
    if run.state!='finished':raise ValueError('Waiting for final W&B state')
    for name in ('final_dev_ce','final_test_ce','best_selected_test_ce','training_tokens','steps'):
        if not math.isclose(run.summary.get(name,float('nan')),result[name],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('Waiting for W&B final summary: '+name)
    result.update(wandb_verified=True,checkpoint_member_sizes_verified=True,
        checkpoint_archive=str(checkpoint),results_archive=str(archive_path),config=config)
    (ROOT/'results'/f'{variant}-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state['status'] in ('complete','needs_attention'):return
        assert len(state['jobs'])==3
        state['monitor_pid']=os.getpid();state.setdefault('results',{});save(state)
        errors=0
        while True:
            try:
                rows={v:job(info,state['submitted_at']) for v,info in state['jobs'].items()}
                state['last_jobs']=rows
                for variant,row in rows.items():
                    if row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0):
                        raise RuntimeError(f'{variant}: {row}')
                    if row and row['JobStatus']==4 and variant not in state['results']:
                        state['results'][variant]=collect(state,variant,state['jobs'][variant]);save(state)
                if len(state['results'])==3:
                    report=dict(complete=True,source=state['source'],jobs=state['jobs'],results=state['results'])
                    (ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
                    state['status']='complete';state['completed_at']=time.time();save(state);return
                errors=0;state['monitor_errors']=0;state['last_poll']=time.time();save(state)
            except Exception as exc:
                errors+=1;state.update(last_error=repr(exc),monitor_errors=errors,last_poll=time.time())
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';save(state);return
                save(state)
            time.sleep(30)


if __name__=='__main__':main()
