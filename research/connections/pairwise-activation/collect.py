"""Collect bounded GPU results only. No publication, resubmission or training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from validation import validate


def query(args):return subprocess.check_output(args,text=True,timeout=45).strip()


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not any((root/x).exists() for x in ('STOPPED','STOPPED.json')):
            s=json.loads((root/'state.json').read_text())
            if s['status'] in ('complete','needs_attention'):return
            assert not s['automatic_training'];s['monitor_pid']=os.getpid()
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason,NumJobStarts'
                rows=json.loads(query(['condor_q',s['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(query(['condor_history',s['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                s.update(last_poll=time.time(),last_job=rows[0] if rows else None)
                if rows and rows[0]['JobStatus']==4:
                    archive=root/'results'/('gate-'+s['job']+'.tar')
                    with tarfile.open(archive) as tf:
                        for m in tf.getmembers():
                            path=Path(m.name)
                            assert not path.is_absolute() and '..' not in path.parts and not m.issym() and not m.islnk()
                            assert not m.name.endswith(('.pt','.wandb'))
                            if m.isfile():
                                dest=root/'collected'/path;dest.parent.mkdir(parents=True,exist_ok=True)
                                dest.write_bytes(tf.extractfile(m).read())
                    assert rows[0]['ExitCode']==0,rows[0]
                    v=json.loads((root/'collected/results/pairwise/validation.json').read_text());validate(v,s['source'])
                    v.update(job=s['job'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
                    (root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
                    s.update(status='complete',speed_target_met=v['speed_target_met'],overhead_percent=v['overhead_percent'])
                elif rows and rows[0]['JobStatus'] in (3,5):s['status']='needs_attention'
            except Exception as e:s.update(status='needs_attention',error=repr(e))
            tmp=root/'state.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(root/'state.json')
            if s['status'] in ('complete','needs_attention'):return
            time.sleep(60)


if __name__=='__main__':main(sys.argv[1])
