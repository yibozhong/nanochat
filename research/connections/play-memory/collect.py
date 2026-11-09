"""Read-only job collection. Never submits, retries, publishes or trains."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def query(command):
    return json.loads(subprocess.check_output(command,text=True) or '[]')


def main(root):
    root=Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((root/'state.json').read_text());job=state['job']
        while True:
            if (root/'STOPPED').exists():return
            queued=query(['condor_q',job,'-json','-attributes','ClusterId,ProcId,JobStatus,NumJobStarts,HoldReason'])
            if queued and queued[0]['JobStatus']!=5:
                time.sleep(60);continue
            if queued:
                state.update(status='needs_attention',queue=queued);break
            history=query(['condor_history',job,'-limit','1','-json'])
            if not history:
                time.sleep(60);continue
            state['history']=history[0]
            archive=root/'results'/('gate-'+job+'.tar')
            if archive.exists():
                dest=root/'collected';dest.mkdir(exist_ok=True)
                with tarfile.open(archive) as tar:
                    for member in tar.getmembers():
                        path=Path(member.name)
                        assert not path.is_absolute() and '..' not in path.parts
                        assert path.parts[0]=='results' and (member.isfile() or member.isdir())
                        assert path.suffix not in ('.pt','.pth','.safetensors')
                    tar.extractall(dest)
                report=dest/'results/play/validation.json'
                if report.exists():
                    data=json.loads(report.read_text())
                    assert data['source']==state['source']
                    assert data['complete'] and data['formal_training'] is False
                    assert data['checks']==dict(tests=23,failures=0,errors=0,skipped=0)
                    assert all(v['steps']==128 for v in data['training'].values())
                    (root/'validation.json').write_text(json.dumps(data,indent=2)+'\n')
            ok=history[0].get('ExitCode')==0 and (root/'validation.json').exists()
            state.update(status='complete' if ok else 'needs_attention',automatic_training=False)
            break
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:v for k,v in state.items() if k!='history'}),flush=True)


if __name__=='__main__':main(sys.argv[1])
