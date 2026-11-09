"""Collect one bounded GPU gate, then exit; never publish or submit training."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def command(args):
    return subprocess.check_output(args,text=True,timeout=60).strip()


def save(root,state):
    tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')


def validate(value,state):
    assert value['complete'] and value['formal_training'] is False and value['automatic_training'] is False
    assert value['source']==state['source']
    assert value['checks']==dict(tests=21,failures=0,errors=0,skipped=0)
    assert value['restart']['complete'] and value['restart']['steps']==128
    assert math.isfinite(value['restart']['final_dev_ce'])
    assert value['microbatch'] in (16,8,4)
    profiles=list(value['profiles'].values());assert len(profiles)==6
    assert [p['mode'] for p in profiles]==['full','sssl','sssl_loop3','sssl_loop3','sssl','full']
    for p in profiles:
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60
        assert p['causality_passed'] and math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert all(math.isfinite(loss) for loss in p['losses'])
        assert p['parameters']==58720256 and p['actual_attention_backend']=='fa3'
        assert p['source']==state['source'] and p['microbatch']==value['microbatch']
        c=p['model'];assert c['architecture']=='vanilla' and c['variant']=='baseline'
        assert c['window_pattern']=='SSSL' and c['swa_window']==512
        assert c['swa_looping']==(p['mode']!='full')
        if c['swa_looping']:assert c['swa_loops']==(1 if p['mode']=='sssl' else 3)
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1


def collect(root,state,ad):
    archive=root/'results'/('gate-'+state['job']+'.tar')
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            path=Path(m.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not m.name.endswith(('.pt','.wandb')) and not m.issym() and not m.islnk()
            if m.isfile():
                dest=root/'collected'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/swa-looping/validation.json').read_text())
    validate(value,state)
    value.update(job=state['job'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (root/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
    state.update(status='validated',mean_step_seconds=value['mean_step_seconds'],
                 final_dev_ce=value['restart']['final_dev_ce'])


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('validated','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason,NumJobStarts'
                rows=json.loads(command(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
                save(root,state)
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(root,state);return
            if state['status'] in ('validated','needs_attention'):return
            time.sleep(60)


if __name__=='__main__':main(sys.argv[1])
