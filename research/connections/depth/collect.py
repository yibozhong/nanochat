"""Collect immutable diagnostic outputs; never submit or resume training."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main(root):
    root=Path(root)
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state=json.loads((root/'state.json').read_text());job=state['job']
            if state['status'] in ('complete','needs_attention'):return
            query=['-json','-attributes','ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason']
            rows=json.loads(subprocess.check_output(['condor_q',job,*query],text=True) or '[]')
            if not rows:rows=json.loads(subprocess.check_output(['condor_history',job,'-limit','1',*query],text=True) or '[]')
            state.update(last_job=rows[0] if rows else None,last_poll=time.time())
            archive=root/'results'/('depth-'+job+'.tar')
            if rows and rows[0]['JobStatus']==4:
                try:
                    assert rows[0]['ExitCode']==0,rows[0]
                    with tarfile.open(archive) as tf:
                        result=json.load(tf.extractfile('results/depth/results.json'))
                        assert result['complete'] and not result['formal_training']
                        assert set(result['models'])=={'vanilla','mhc','native'}
                        for model in result['models'].values():
                            assert model['checks']['weights_and_buffers_unchanged']
                            assert model['checks']['unmodified_forward_exact']
                            assert max(model['checks']['skip_oracle_max_abs'])==0
                        for member in tf.getmembers():
                            assert not member.name.endswith(('.pt','.wandb'))
                            if member.isfile() and member.name.startswith('results/depth/'):
                                dest=root/'collected'/Path(member.name).name
                                dest.parent.mkdir(exist_ok=True)
                                dest.write_bytes(tf.extractfile(member).read())
                    state['status']='complete'
                except Exception as exc:state.update(status='needs_attention',error=repr(exc))
            elif rows and rows[0]['JobStatus'] in (3,5):state.update(status='needs_attention')
            tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
            if state['status'] in ('complete','needs_attention'):return
            time.sleep(30)


if __name__=='__main__':main(sys.argv[1])
