"""Collect only; no publication, retries, or formal training submission."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def validate(v,state):
    assert v['complete'] and not v['formal_training'] and not v['new_formal_run']
    assert v['source']==state['source'] and v['extra_gate_parameters']==16777216
    assert v['checks']==dict(tests=36,failures=0,errors=0,skipped=0)
    assert v['microbatch'] in (16,8,4)
    assert set(v['profiles'])=={'torch',*[f'abba-{i}' for i in range(4)]}
    for name,p in v['profiles'].items():
        assert p['variant']==('stream_gated_hc' if name in ('abba-0','abba-3') else 'stream_gated_hc_post')
        assert p['source']==state['source'] and p['complete'] and p['status']=='fits'
        assert p['optimizer_updates']==60 and p['causality_passed']
        assert p['microbatch']==v['microbatch'] and p['batch_tokens']==32768 and p['sequence_length']==2048
        assert p['peak_reserved_bytes']<.9*22000*1024**2
        assert any(gpu in p['gpu'] for gpu in ('H100','H200')) and p['actual_attention_backend']=='fa3'
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert len(p['step_seconds'])==40 and all(math.isfinite(x) and x>0 for x in p['step_seconds'])
        assert p['gate_backend']==('torch' if name=='torch' else 'fused')
    assert len({p['gpu_uuid'] for p in v['profiles'].values()})==1
    assert len({p['input_sha256'] for p in v['profiles'].values()})==1
    assert {p['parameters'] for p in v['profiles'].values()}=={76292533}
    assert v['extra_parameters_over_pre']==0
    assert v['restart']['complete'] and v['restart']['steps']==128 and v['restart']['source']==state['source']
    assert all(math.isfinite(v['restart'][k]) for k in ('initial_dev_ce','final_dev_ce'))
    assert v['restart']['final_dev_ce']<v['restart']['initial_dev_ce']
    m=v['micro'];assert m['complete'] and m['optimizer_updates']==0 and m['source']==state['source']
    assert m['gpu_uuid']==v['profiles']['abba-0']['gpu_uuid'] and m['shape']==[v['microbatch'],2048,512,4]
    assert set(m['results'])=={'pre_fused','post_fused','post_torch'}
    for name,rows in m['results'].items():
        assert len(rows)==2 and m['traces'][name]['kernel_count']>0
        assert all(math.isfinite(r['mean_seconds']) and r['mean_seconds']>0 for r in rows)
    reference=(v['profiles']['abba-0']['mean_seconds']+v['profiles']['abba-3']['mean_seconds'])/2
    fused=(v['profiles']['abba-1']['mean_seconds']+v['profiles']['abba-2']['mean_seconds'])/2
    assert v['pre_seconds']==reference and v['post_seconds']==fused
    assert v['post_latency_change_percent']==100*(fused/reference-1)


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
    value=json.loads((root/'collected/results/post-stream-gate/validation.json').read_text());validate(value,state)
    (root/'validation.json').write_text(json.dumps(dict(value,job=state['job']),indent=2)+'\n')
    state.update(status='complete',post_ms=value['post_seconds']*1000,
                 pre_ms=value['pre_seconds']*1000,new_formal_run=False)


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text())
            if state['status'] in ('complete','needs_attention','stopped_by_user'):return
            state['monitor_pid']=os.getpid()
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                rows=json.loads(subprocess.check_output(['condor_q',state['job'],'-json','-attributes',attrs],text=True,timeout=60) or '[]')
                if not rows:rows=json.loads(subprocess.check_output(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs],text=True,timeout=60) or '[]')
                state.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:collect(root,state,rows[0])
                elif rows and rows[0]['JobStatus'] in (3,5):state['status']='needs_attention'
            except Exception as exc:state.update(status='needs_attention',error=repr(exc))
            p=root/'state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'state.json')
            if state['status'] in ('complete','needs_attention'):return
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
