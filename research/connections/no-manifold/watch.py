"""Collect one bounded three-variant ablation; never submit follow-up training.

Usage: python watch.py /home/yzhong239/mhc-no-manifold
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
        result=json.load(tar.extractfile('results/gate.json'))
        configs={v:json.load(tar.extractfile(f'results/{v}/config.json')) for v in result['smokes']}
        offline=[m for m in tar.getmembers() if '/offline-run-' in m.name and m.isfile()]
        if offline:
            dest=ROOT/'results'/('offline-'+s['job'])
            for m in offline:
                p=dest/m.name;assert p.resolve().is_relative_to(dest.resolve())
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
            for p in sorted((dest/'results/wandb').glob('offline-run-*')):command(['wandb','sync',str(p)])
    assert len({v['gpu'] for v in result['smokes'].values()})==1
    ref=configs['mhc'];other=configs['mhc_no_manifold']
    assert ref['parameters']==other['parameters']
    assert {k:v for k,v in ref['model'].items() if k!='variant'}=={k:v for k,v in other['model'].items() if k!='variant'}
    assert all(c['data_manifest']==ref['data_manifest'] and c['protocol']==ref['protocol'] for c in configs.values())
    assert result['passed'] and result['source']==s['source']
    assert result['checks']['failures']==result['checks']['errors']==0
    assert set(result['smokes'])=={'baseline','mhc','mhc_no_manifold'}
    import wandb
    api=wandb.Api(timeout=30)
    r=api.run('yibozhong657-none/nanochat/'+result['run_id'])
    assert r.state=='finished' and r.summary['passed']
    for name,smoke in result['smokes'].items():
        assert smoke['complete'] and smoke['variant']==name and smoke['steps']==50
        assert smoke['training_tokens']==1638400 and smoke['source']==s['source']
        assert smoke['final_dev_ce']<smoke['initial_dev_ce']-.02
        r=api.run('yibozhong657-none/nanochat/'+smoke['run_id'])
        assert r.state=='finished' and r.summary['complete']
        assert math.isclose(r.summary['final_dev_ce'],smoke['final_dev_ce'],rel_tol=1e-8)
    result.update(wandb_verified=True,job=s['job'])
    for p in [ROOT/'validation.json',REPO/'research/connections/no-manifold/validation.json']:p.write_text(json.dumps(result,indent=2)+'\n')
    s.update(status='complete',validation=result,push_status='pending_review_and_push');save(s)
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
