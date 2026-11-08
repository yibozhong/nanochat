"""Collect one bounded scale validation; never publish, submit or train."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def validate(v,s):
    assert v['complete'] and not v['formal_training'] and not v['preflight']
    assert v['source']==s['source']
    assert v['checks']==dict(tests=52,failures=0,errors=0,skipped=0)
    assert v['restart']['complete'] and v['restart']['steps']==128
    assert math.isfinite(v['restart']['final_dev_ce']) and v['restart']['final_dev_ce']<v['restart']['initial_dev_ce']
    assert v['microbatch'] in (16,8,4)
    profiles=list(v['profiles'].values())
    assert sorted(p['variant'] for p in profiles)==['mhc_no_sinkhorn','mhc_no_sinkhorn','scaled_hc','scaled_hc','scaled_hc_all','scaled_hc_all']
    assert len({p['gpu_uuid'] for p in profiles})==len({p['input_sha256'] for p in profiles})==1
    for p in profiles:
        assert p['source']==s['source'] and p['complete'] and p['status']=='fits'
        assert p['optimizer_updates']==60 and p['causality_passed'] and p['microbatch']==v['microbatch']
        assert p['batch_tokens']==32768 and p['sequence_length']==2048
        assert math.isfinite(p['mean_seconds']) and p['mean_seconds']>0
        assert len(p['losses'])==60 and all(math.isfinite(x) for x in p['losses'])
        assert p['peak_reserved_bytes']<.9*22000*1024**2
        if p['variant']=='scaled_hc_all':assert p['parameters']==59615669
    assert v['micro']['complete'] and v['micro']['shape']==[v['microbatch'],2048,512,4]
    assert v['micro']['gpu_uuid']==profiles[0]['gpu_uuid']
    assert set(v['micro']['results'])=={'read_shared','read_scaled','write_mix_only','write_all'}


def save(root,s):
    p=root/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(root/'state.json')


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not any((root/n).exists() for n in ('STOPPED','STOPPED.json')):
            s=json.loads((root/'state.json').read_text())
            if s['status'] in ('complete','needs_attention'):return
            s['monitor_pid']=os.getpid()
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason,NumJobStarts'
                def query(args):return json.loads(subprocess.check_output(args,text=True,timeout=45).strip() or '[]')
                rows=query(['condor_q',s['job'],'-json','-attributes',attrs])
                if not rows:rows=query(['condor_history',s['job'],'-limit','1','-json','-attributes',attrs])
                s.update(last_job=rows[0] if rows else None,last_poll=time.time())
                if rows and rows[0]['JobStatus']==4:
                    archive=root/'results'/('gate-'+s['job']+'.tar')
                    with tarfile.open(archive) as tf:
                        for m in tf.getmembers():
                            p=Path(m.name)
                            assert not p.is_absolute() and '..' not in p.parts
                            assert not m.issym() and not m.islnk() and not m.name.endswith(('.pt','.wandb'))
                            if m.isfile():
                                dest=root/'collected'/p;dest.parent.mkdir(parents=True,exist_ok=True)
                                dest.write_bytes(tf.extractfile(m).read())
                    assert rows[0].get('ExitCode')==0,rows[0]
                    value=json.loads((root/'collected/results/scaled-hc-all/validation.json').read_text())
                    validate(value,s)
                    (root/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
                    s.update(status='complete',automatic_training=False)
                elif rows and rows[0]['JobStatus'] in (3,5):s['status']='needs_attention'
                save(root,s)
            except Exception as exc:
                s.update(status='needs_attention',error=repr(exc));save(root,s);return
            if s['status'] in ('complete','needs_attention'):return
            time.sleep(300)


if __name__=='__main__':main(sys.argv[1])
