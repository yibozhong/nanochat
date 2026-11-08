"""Stop watching after the first finite training step. Never submits a job."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def first_formal_step(lines,run_id):
    """Only accept a finite row after this run's explicit formal-start marker."""
    marker='FORMAL_TRAINING_BEGIN scaled_hc_all '+run_id
    formal=False
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN '):formal=line==marker
        if not formal or not line.startswith('{'):continue
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if 'step' in value and 'loss' in value and value['step']>=1 and math.isfinite(value['loss']):
            return value
    return None


def main(root):
    root=Path(root)
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text());job=state['job']
            if state['status'] in ('training_started','needs_attention'):return
            state.update(monitor_pid=os.getpid(),last_poll=time.time())
            try:
                args=['-json','-attributes','ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost,NumJobStarts']
                rows=json.loads(subprocess.check_output(['condor_q',job,*args],text=True,timeout=60).strip() or '[]')
                if not rows:rows=json.loads(subprocess.check_output(['condor_history',job,'-limit','1',*args],text=True,timeout=60).strip() or '[]')
                state['last_job']=rows[0] if rows else None
                log=root/'logs'/f'train.{job}.out'
                if log.exists():
                    value=first_formal_step(log.read_text(errors='replace').splitlines(),state['run_id'])
                    if value is not None:
                        state.update(status='training_started',first_observed_step=value,monitor_complete=True)
                if rows and rows[0]['JobStatus'] in (3,4,5) and state['status']!='training_started':
                    state.update(status='needs_attention',monitor_complete=True)
            except Exception as exc:
                state['poll_errors']=state.get('poll_errors',0)+1;state['last_error']=repr(exc)
                if state['poll_errors']>=3:state['status']='needs_attention'
            tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
            if state['status'] in ('training_started','needs_attention'):return
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
