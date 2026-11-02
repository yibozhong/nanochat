"""Collect only existing vanilla7669762.0; no submissions or other job mutations."""
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
JOB='7669762.0'
RUN='0bb03c6e'
ATTRS='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost,NumJobStarts,LookbackKey,LookbackVariant,LookbackRunID'


def write(value):
    p=ROOT/'state.tmp';p.write_text(json.dumps(value,indent=2)+'\n');p.replace(ROOT/'state.json')


def command(args):
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=60).strip()


def collect(state):
    with tarfile.open(ROOT/'results'/f'baseline-{JOB}.tar') as tf:
        r=json.load(tf.extractfile('results/train/summary.json'))
        c=json.load(tf.extractfile('results/train/config.json'))
    assert r['complete'] and r['run_id']==RUN and r['variant']=='baseline'
    assert r['source']==state['training_source'] and c['source']==state['training_source']
    assert r['steps']==305176 and r['training_tokens']==10000007168
    assert r['test_evaluated'] and r['dev_tokens']==131072 and r['test_tokens']==2097152
    gate=json.loads((ROOT/'validation.json').read_text())
    assert c['model']==gate['resumes']['baseline']['config']['model']
    assert c['data_manifest']==state['data_manifest'] and c['logging_backend']=='wandb'
    assert c['protocol']==dict(steps=305176,schedule_steps=305176,batch_tokens=32768,
        microbatch=2,grad_accum=1,seed=42,attention_backend='sdpa_cudnn',
        matched_backbone_init=True,evaluate_test=True)
    assert c['checkpoint_transport']['upload_interval_seconds']==1800
    with tarfile.open(ROOT/'results'/f'baseline-{JOB}-checkpoints.tar') as tf:
        assert {m.name for m in tf if m.isfile()}=={'checkpoints/latest.pt','checkpoints/best.pt'}
        for name,meta in r['checkpoint_files'].items():
            assert tf.getmember('checkpoints/'+name).size==meta['bytes']
    import wandb
    run=wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+RUN)
    assert run.config['source']==state['training_source']
    if run.state!='finished':raise ValueError('Waiting for W&B finished state')
    for key in ('steps','training_tokens','final_dev_ce','final_test_ce','best_selected_test_ce'):
        if not math.isclose(run.summary.get(key,float('nan')),r[key],rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError('Waiting for W&B summary '+key)
    r.update(config=c,wandb_verified=True,checkpoint_sizes_verified=True)
    (ROOT/'results/baseline-verified.json').write_text(json.dumps(r,indent=2)+'\n')
    return r


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        assert state['training_jobs']['baseline']==dict(job=JOB,run_id=RUN)
        assert 'lookback' in state.get('stopped_variants',{})
        if state['status']=='baseline_complete_lookback_stopped':return
        state.update(status='baseline_only_running',monitor_pid=os.getpid(),monitor_errors=0)
        write(state);errors=0
        while not (ROOT/'STOPPED-baseline.json').exists():
            try:
                rows=json.loads(command(['condor_q',JOB,'-json','-attributes',ATTRS]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',JOB,'-limit','1','-json','-attributes',ATTRS]) or '[]')
                state.update(last_jobs={'baseline':rows[0] if rows else None},last_poll=time.time())
                if rows:
                    row=rows[0];assert row['LookbackVariant']=='baseline' and row['LookbackRunID']==RUN
                    if row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0:
                        raise RuntimeError(row)
                    if row['JobStatus']==4:
                        state.setdefault('training_results',{})['baseline']=collect(state)
                        state.update(status='baseline_complete_lookback_stopped',completed_at=time.time())
                        write(state);return
                errors=0;state['monitor_errors']=0;write(state)
            except Exception as exc:
                errors+=1;state.update(monitor_errors=errors,last_error=repr(exc));print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    state['status']='needs_attention';write(state);return
                write(state)
            time.sleep(30)


if __name__=='__main__':main()
