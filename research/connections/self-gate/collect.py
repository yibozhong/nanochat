"""Collect only: no training submission, retries, credentials or publication."""
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


def save(root,state):
    p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')


def validate(v,state):
    assert v['complete'] and not v['formal_training'] and not v['new_formal_run']
    assert v['source']==state['source'] and v['formula']=='x*silu(x)' and v['extra_gate_parameters']==0
    assert v['first_step_probe']['passed'] and v['first_step_probe']['source']==state['source']
    assert v['checks']==dict(tests=17,failures=0,errors=0,skipped=0)
    assert set(v['profiles'])=={'baseline','mhc_no_sinkhorn','gated_hc','self_gated_hc',*[f'abba-{i}' for i in range(4)]}
    for name,p in v['profiles'].items():
        expected=('mhc_no_sinkhorn' if name in ('abba-0','abba-3') else 'self_gated_hc') if name.startswith('abba-') else name
        assert p['variant']==expected and p['source']==state['source']
        assert p['complete'] and p['status']=='fits' and p['optimizer_updates']==60 and p['causality_passed']
        assert p['microbatch']==16 and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['peak_reserved_bytes']<.9*22000*1024**2
        assert 'H200' in p['gpu'] and p['actual_attention_backend']=='fa3'
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
    assert len({p['gpu_uuid'] for p in v['profiles'].values()})==1
    assert len({p['input_sha256'] for p in v['profiles'].values()})==1
    assert v['profiles']['self_gated_hc']['parameters']==v['profiles']['mhc_no_sinkhorn']['parameters']
    assert v['restart']['complete'] and v['restart']['steps']==128
    assert v['restart']['source']==state['source']
    assert all(math.isfinite(v['restart'][k]) for k in ('initial_dev_ce','final_dev_ce'))
    assert v['restart']['final_dev_ce']<v['restart']['initial_dev_ce']
    m=v['micro'];assert m['complete'] and m['optimizer_updates']==0 and m['source']==state['source']
    assert m['gpu_uuid']==v['profiles']['self_gated_hc']['gpu_uuid']
    assert set(m['results'])=={'hc_write','eager_self_gate','compiled_self_gate'}
    for name,rows in m['results'].items():
        assert len(rows)==2 and m['traces'][name]['kernel_count']>0
        assert all(math.isfinite(r['mean_seconds']) and r['mean_seconds']>0 for r in rows)
    old=(v['profiles']['abba-0']['mean_seconds']+v['profiles']['abba-3']['mean_seconds'])/2
    new=(v['profiles']['abba-1']['mean_seconds']+v['profiles']['abba-2']['mean_seconds'])/2
    assert v['old_seconds']==old and v['new_seconds']==new and v['target90ms_reached']==(new<=.09)
    assert v['overhead_percent']==100*(new/old-1)


def collect(root,state,ad):
    with tarfile.open(root/'results'/('gate-'+state['job']+'.tar')) as tf:
        for m in tf.getmembers():
            path=Path(m.name)
            assert not path.is_absolute() and '..' not in path.parts and not m.issym() and not m.islnk()
            assert not m.name.endswith(('.pt','.wandb'))
            if m.isfile():
                target=root/'collected'/path;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(tf.extractfile(m).read())
    assert ad.get('ExitCode')==0,ad
    value=json.loads((root/'collected/results/self-gate/validation.json').read_text());validate(value,state)
    for name,digest in json.loads((root/'existing-training-inputs.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    (root/'validation.json').write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    state.update(status='complete',old_ms=value['old_seconds']*1000,new_ms=value['new_seconds']*1000,
                 target90ms_reached=value['target90ms_reached'],new_formal_run=False)


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention'):return
            state['monitor_pid']=os.getpid()
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                text=subprocess.check_output(['condor_q',state['job'],'-json','-attributes',attrs],text=True,timeout=60)
                rows=json.loads(text or '[]')
                if not rows:rows=json.loads(subprocess.check_output(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs],text=True,timeout=60) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
            except Exception as exc:state.update(status='needs_attention',error=repr(exc))
            save(root,state)
            if state['status'] in ('complete','needs_attention'):return
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
