"""Observe formal start and apply the authorized one-hour queue fallback."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from policy import desired_fallback, require_gpus


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def command(args):
    return subprocess.check_output(args,text=True,timeout=45).strip()


def first_formal_step(lines,run_id):
    formal=False
    marker='FORMAL_TRAINING_BEGIN qk_scale '+run_id
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN '):formal=line==marker
        if not formal or not line.startswith('{'):continue
        try:v=json.loads(line)
        except json.JSONDecodeError:continue
        if 'step' in v and 'loss' in v and v['step']>=1 and math.isfinite(v['loss']):return v
    return None


def main(root):
    root=Path(root).resolve()
    with (root/'scheduling.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'scheduling.json'
        state=json.loads(path.read_text()) if path.exists() else dict(transitions=[],poll_errors=0)
        while not any((root/n).exists() for n in ('STOPPED','STOPPED.json')):
            training=json.loads((root/'state.json').read_text());job=training['job']
            assert training['source']['model_variant']=='qk_scale'
            state.update(pid=os.getpid(),job=job,last_poll=time.time())
            try:
                attrs='ClusterId,ProcId,JobStatus,EnteredCurrentStatus,ExitCode,HoldReason,RemoteHost,NumJobStarts,RequireGPUs,QKScaleTrainKey'
                rows=json.loads(command(['condor_q',job,'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',job,'-limit','1','-json','-attributes',attrs]) or '[]')
                ad=rows[0] if rows else None;state['last_job']=ad
                if ad:
                    assert ad['QKScaleTrainKey']=='qk-scale-10b-20261108'
                    want=desired_fallback(ad,time.time())
                    current='A100' in ad.get('RequireGPUs','')
                    if want is not None and want!=current:
                        cluster,proc=map(int,job.split('.'))
                        # Atomically restrict the edit to the observed status epoch;
                        # a concurrent match/eviction cannot broaden a fresh queue wait.
                        condition=f'ClusterId=={cluster} && ProcId=={proc} && JobStatus=={ad["JobStatus"]} && EnteredCurrentStatus=={ad["EnteredCurrentStatus"]}'
                        if want:condition+=' && time()-EnteredCurrentStatus>=3600'
                        result=command(['condor_qedit','-constraint',condition,'RequireGPUs',require_gpus(want)])
                        state['transitions'].append(dict(time=time.time(),fallback=want,status=ad['JobStatus'],idle_since=ad['EnteredCurrentStatus'],result=result))
                    if ad['JobStatus'] in (3,5) or ad['JobStatus']==4 and ad.get('ExitCode')!=0:
                        state['status']='needs_attention'
                        training.update(status='needs_attention',last_job=ad);save(root/'state.json',training)
                    elif ad['JobStatus']==4:
                        state['status']='completed_job'
                    else:state['status']='monitoring'
                if 'first_formal_step' not in state:
                    log=root/'logs'/f'train.{job}.out'
                    value=first_formal_step(log.read_text(errors='replace').splitlines(),training['run_id']) if log.exists() else None
                    if value:
                        state['first_formal_step']=value
                        if state.get('status')!='needs_attention':
                            training.update(status='training_started',first_observed_step=value)
                            save(root/'state.json',training)
                        save(root/'startup-observation.json',dict(job=job,run_id=training['run_id'],first_formal_step=value,observed_at=time.time(),last_job=ad))
                state['poll_errors']=0;save(path,state)
            except Exception as exc:
                state.update(poll_errors=state.get('poll_errors',0)+1,last_error=repr(exc));save(path,state)
                if state['poll_errors']>=3:
                    state['status']='needs_attention';save(path,state);return
            if state.get('status') in ('needs_attention','completed_job'):return
            time.sleep(60)


if __name__=='__main__':main(sys.argv[1])
