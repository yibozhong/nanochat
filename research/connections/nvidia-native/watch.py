"""Collect bounded same-GPU projection comparison; never submit follow-up training.

Usage: python watch.py /home/yzhong239/mhc-nvidia-native
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
        result=json.load(tar.extractfile('results/comparison.json'))
        offline=[m for m in tar.getmembers() if '/offline-run-' in m.name and m.isfile()]
        if offline:
            dest=ROOT/'results'/('offline-'+s['job'])
            for m in offline:
                p=dest/m.name;assert p.resolve().is_relative_to(dest.resolve())
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(m).read())
            for p in sorted((dest/'results/wandb').glob('offline-run-*')):command(['wandb','sync',str(p)])
    assert result['complete'] and result['source']==s['source']
    assert result['checks']['failures']==result['checks']['errors']==result['checks']['skipped']==0
    assert set(result['trials'])=={'baseline','mhc_official','mhc_torch_compile','mhc_nvidia_native','mhc_nvidia_fp32_dw','mhc_nvidia_precise'}
    assert len({v['config']['gpu_uuid'] for v in result['trials'].values()})==1
    import wandb
    api=wandb.Api(timeout=30)
    parent=api.run('yibozhong657-none/nanochat/'+result['run_id'])
    assert parent.state=='finished' and parent.summary['complete'] and parent.summary['checks_passed']
    assert parent.group=='mhc-nvidia-native-20261030'
    for name,v in result['trials'].items():
        assert v['complete'] and v['source']==s['source']
        assert v['optimizer_updates']==106 and v['training_tokens']==106*32768
        assert v['cuda_profile']['kernel_count']>0
        run=api.run('yibozhong657-none/nanochat/'+v['run_id'])
        assert run.state=='finished' and run.summary['complete']
        assert run.group==parent.group
        assert math.isclose(run.summary['normal_step_ms'],v['normal_ms']['mean'],rel_tol=1e-8)
    result.update(wandb_verified=True,job=s['job'])
    for p in [ROOT/'comparison.json',REPO/'research/connections/nvidia-native/results.json']:
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
