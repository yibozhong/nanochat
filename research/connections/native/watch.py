"""Collect this implementation gate; never submit or release formal training."""
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=45).strip()


def save(state):
    path=ROOT/'state.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(ROOT/'state.json')


def collect(state):
    archive=ROOT/'results'/f'gate-{state["job"]}.tar'
    with tarfile.open(archive) as tar:
        assert not any(m.name.endswith(('.pt','.wandb')) or 'wandb' in Path(m.name).parts for m in tar)
        result=json.load(tar.extractfile('results/native-gate.json'))
        assert result['complete'] and result['source']==state['source']
        assert result['checks']==dict(tests=11,failures=0,errors=0,skipped=0)
        assert result['logging_backend']=='local' and result['diagnostic_tokens']==983040
        evidence={Path(m.name).stem:json.load(tar.extractfile(m)) for m in tar
                  if m.isfile() and m.name.startswith('results/native-checks/') and m.name.endswith('.json')}
        for variant,trial in result['trials'].items():
            assert trial['config']['model']['architecture']=='native'
            assert trial['config']['model']['variant']==variant
            assert trial['summary']['complete'] and trial['summary']['steps']==10
        result.update(job=state['job'],device=json.load(tar.extractfile('results/device.json')),evidence=evidence)
    assert command(['git','rev-parse','HEAD'],REPO)==state['reviewed_head'],'Review changed HEAD before publishing'
    assert not command(['git','status','--porcelain'],REPO),'Review outstanding edits before publishing'
    report=REPO/'research/connections/native/validation.json'
    report.write_text(json.dumps(result,indent=2)+'\n')
    command(['git','add',str(report)],REPO)
    command(['git','commit','-m','Record native NVIDIA connection integration validation'],REPO)
    # Reuse the audited credential-safe push helper; runpy does not start main.
    helper=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))
    state['published_commit']=helper['push']()
    state['status']='complete';state['report']=str(report);state['completed_at']=time.time()
    save(state)


def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text())
        if state['status']=='complete':return
        state['monitor_pid']=__import__('os').getpid();save(state)
        errors=0
        while True:
            try:
                rows=json.loads(command(['condor_q',state['job'],'-json','-attributes','ClusterId,ProcId,JobStatus,RemoteHost,HoldReason']) or '[]')
                state['queue']=rows;state['last_poll']=time.time();save(state)
                if rows:
                    if rows[0]['JobStatus']==5:raise RuntimeError(rows[0].get('HoldReason','Job held'))
                else:
                    rows=json.loads(command(['condor_history',state['job'],'-match','1',
                        '-since',f'CompletionDate < {int(state["submitted_at"])-60}',
                        '-json','-attributes','JobStatus,ExitCode']) or '[]')
                    if rows:
                        state['history']=rows;save(state)
                        if rows[0].get('JobStatus')!=4 or rows[0].get('ExitCode')!=0:
                            state.update(status='needs_attention',error=f'Gate failed: {rows}');save(state);return
                        collect(state);return
                errors=0
            except Exception as exc:
                errors+=1;state.update(error=repr(exc),consecutive_errors=errors);save(state)
                if errors>=5:
                    state['status']='needs_attention';save(state);return
            time.sleep(30)

if __name__=='__main__':main()
