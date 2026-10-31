"""Collect bounded same-GPU profiling; never submit follow-up training.

Usage: python watch.py /home/yzhong239/mhc-profiling
"""
import fcntl,json,math,os,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
STATE=ROOT/'state.json'
def command(args):
    return subprocess.check_output(args,cwd=ROOT,text=True,timeout=60).strip()
def save(s):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2)+'\n');p.replace(STATE)
def job(ident):
    attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,ConnectionGateKey'
    rows=json.loads(command(['condor_q',ident,'-json','-attributes',attrs]) or '[]')
    if not rows:rows=json.loads(command(['condor_history',ident,'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
    return rows[0] if rows else None
def check(row):
    if row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0):
        raise RuntimeError(row)
def collect(s):
    with tarfile.open(ROOT/'results'/f'gate-{s["job"]}.tar') as tar:
        result=json.load(tar.extractfile('results/profile.json'))
    assert result['complete'] and result['source']==s['source']
    assert set(result['variants'])=={'baseline','mhc','mhc_no_manifold'}
    assert len({v['config']['gpu'] for v in result['variants'].values()})==1
    for name,v in result['variants'].items():
        assert v['complete'] and v['source']==s['source'] and v['variant']==name
        assert v['optimizer_updates']==66 and v['training_tokens']==66*32768
        assert v['cuda_profile']['kernel_count']>0
    result.update(local_verified=True,job=s['job'])
    for p in [ROOT/'profile.json',REPO/'research/connections/profile/results.json']:
        p.write_text(json.dumps(result,indent=2)+'\n')
    s.update(status='complete',push_status='pending_analysis_and_push');save(s)
def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors=0
        while True:
            s=json.loads(STATE.read_text());s['monitor_pid']=os.getpid()
            if s['status'] in ('complete','needs_attention'):return
            try:
                row=job(s['job']);s['last_job']=row
                check(row)
                if row and row['JobStatus']==4:
                    collect(s);return
                errors=0;s['monitor_errors']=0;s['last_poll']=time.time();save(s)
            except Exception as exc:
                errors+=1;s.update(monitor_errors=errors,last_error=repr(exc))
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    s['status']='needs_attention';save(s);return
                save(s)
            time.sleep(30)
if __name__=='__main__':main()
