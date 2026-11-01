"""Collect one local diagnostic; never train, resubmit, or create W&B runs."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()


def save(s):
    p=ROOT/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(ROOT/'state.json')


def command(args):return subprocess.check_output(args,text=True,timeout=60).strip()


def collect(s):
    archive=ROOT/'results'/f'diagnostic-{s["job"]}.tar'
    with tarfile.open(archive) as tar:
        r=json.load(tar.extractfile('results/mixing-diagnostics.json'))
        assert r['complete'] and r['logging_backend']=='local' and r['plan']==s['plan']
        assert r['checkpoint_step']==s['plan']['cases'][0]['expected_step']
        assert r['dev_tokens']==131072 and r['sampled_tokens']==8192
        assert r['vjp']['max_relative_reconstruction_error']<.03
        checks=json.load(tar.extractfile('results/mixing-self-test.json'));assert checks['passed']
        assert not any(m.name.endswith(('.pt','.wandb')) or 'wandb' in Path(m.name).parts for m in tar)
        target=ROOT/'results/collected';target.mkdir(exist_ok=True)
        for name in ['mixing-diagnostics.json','mixing-self-test.json','mixing-spectra.npz','mixing-probe-losses.npz']:
            (target/name).write_bytes(tar.extractfile('results/'+name).read())
    s.update(status='complete',result=str(target/'mixing-diagnostics.json'),completed_at=time.time(),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    save(s)
    print(json.dumps(dict(complete=True,step=r['checkpoint_step'],dev_ce=r['dev_ce'],
        contrast_singular_values=r['matrix']['contrast_singular_values'],vjp={k:r['vjp'][k] for k in ['direct_norm_over_total','routing_norm_over_total']},
        disagreement_retention=r['activation_disagreement_energy_retention'],probes={k:{q:v[q] for q in ['ce','delta_ce']} for k,v in r['probes'].items()})),flush=True)


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);errors=0
        while True:
            s=json.loads((ROOT/'state.json').read_text());s['monitor_pid']=os.getpid()
            if s['status'] in ('complete','needs_attention','stopped_by_user'):return
            try:
                attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost,MixingDiagnosticKey'
                rows=json.loads(command(['condor_q',s['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',s['job'],'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
                if rows:
                    row=rows[0];s['last_job']=row
                    assert row['MixingDiagnosticKey']=='mhc-mixing-diagnostics-20261031'
                    if row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0:raise RuntimeError(row)
                    if row['JobStatus']==4:collect(s);return
                s.update(last_poll=time.time(),monitor_errors=0);save(s);errors=0
            except Exception as exc:
                errors+=1;s.update(last_error=repr(exc),monitor_errors=errors)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):s['status']='needs_attention'
                save(s);print(repr(exc),flush=True)
                if s['status']=='needs_attention':return
            time.sleep(30)


if __name__=='__main__':main()
