"""Collect the interim diagnostic and later repeat once on matched10B checkpoints."""
import fcntl
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
PARTITION=Path('/home/yzhong239/mhc-vocab-partition-20261031')


def cmd(args,cwd=ROOT):return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(s):
    with (ROOT/'state.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=ROOT/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(ROOT/'state.json')


def job(ident):
    attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RankDiagnosticKey,RankDiagnosticPhase,RemoteHost'
    rows=json.loads(cmd(['condor_q',ident,'-json','-attributes',attrs]) or '[]')
    if not rows:rows=json.loads(cmd(['condor_history',ident,'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
    if rows:assert rows[0]['RankDiagnosticKey']=='mhc-rank-diagnostics-20261031'
    return rows[0] if rows else None


def collect(s,phase):
    p=s['phases'][phase];archive=ROOT/'results'/f'{phase}-{p["job"]}.tar'
    with tarfile.open(archive) as a:
        result=json.load(a.extractfile('results/diagnostics.json'))
        assert result['complete'] and result['logging_backend']=='local' and result['plan']==p['plan']
        assert json.load(a.extractfile('results/self-test.json'))['passed']
        assert set(result['cases'])=={'partition','mhc'}
        assert not any(m.name.endswith(('.pt','.wandb')) or 'wandb' in Path(m.name).parts for m in a)
        destination=ROOT/'results'/phase;destination.mkdir(exist_ok=True)
        for member in a:
            if member.isfile() and member.name in ('results/diagnostics.json','results/spectra.npz','results/self-test.json'):
                (destination/Path(member.name).name).write_bytes(a.extractfile(member).read())
    for case in result['cases'].values():
        assert case['dev_tokens']==131072 and case['sampled_tokens']==8192 and math.isfinite(case['dev_ce'])
        assert len(case['sites'])==10
        assert case['head_probes']['original']['analytic_vs_cce_gradient_relative_l2']<.02
    if phase=='final':assert all(c['checkpoint_step']==305176 for c in result['cases'].values())
    p.update(status='complete',result=str(destination/'diagnostics.json'),completed_at=time.time())
    save(s)
    print(json.dumps(dict(phase=phase,complete=True,metrics={name:dict(dev_ce=c['dev_ce'],
        head_rank=c['trained_head']['block_diagonal']['rank_rel_1e6'],
        gradient_effective_rank=c['sites']['head_normalized']['gradient']['uncentered']['energy_entropy_rank'],
        stream_difference_energy=c['sites']['block8']['activation']['streams']['stream_difference_energy_fraction'])
        for name,c in result['cases'].items()})),flush=True)


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors=0
        while True:
            s=json.loads((ROOT/'state.json').read_text())
            s['monitor_pid']=os.getpid()
            if s['status'] in ('complete','needs_attention'):return
            try:
                for phase,p in list(s['phases'].items()):
                    if p['status']=='complete':continue
                    row=job(p['job']);p['last_job']=row
                    if row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0):
                        raise RuntimeError(row)
                    if row and row['JobStatus']==4:collect(s,phase)
                s.update(last_poll=time.time(),monitor_errors=0)
                if s['phases'].get('final',{}).get('status')=='complete':s['status']='complete'
                elif s['phases']['interim']['status']=='complete':s['status']='interim_complete_waiting_final_training'
                save(s);errors=0
                # The phase submitter has its own lock and records intent before mutation.
                if 'final' not in s['phases'] and s['phases']['interim']['status']=='complete' and (PARTITION/'results.json').exists():
                    print(cmd([sys.executable,str(REPO/'research/connections/rank-diagnostics/submit.py'),str(ROOT),'final']),flush=True)
                if s['status']=='complete':return
            except Exception as exc:
                errors+=1;s.update(last_error=repr(exc),monitor_errors=errors)
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):s['status']='needs_attention'
                save(s)
                if s['status']=='needs_attention':return
            time.sleep(30)


if __name__=='__main__':main()
