"""Collect the single authorized width sweep; never launch follow-up jobs."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(sys.argv[1]).resolve()
REPO = Path(__file__).resolve().parents[3]
STATE = ROOT/'state.json'


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=60).strip()


def save(state):
    path = STATE.with_suffix('.tmp'); path.write_text(json.dumps(state, indent=2)+'\n'); path.replace(STATE)


def job(ident):
    attrs = 'ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost,ConnectionGateKey'
    rows = json.loads(command(['condor_q', ident, '-json', '-attributes', attrs]) or '[]')
    if not rows:
        rows = json.loads(command(['condor_history', ident, '-limit', '1', '-scanlimit', '10000', '-json', '-attributes', attrs]) or '[]')
    return rows[0] if rows else None


def collect(state):
    with tarfile.open(ROOT/'results'/f'gate-{state["job"]}.tar') as tar:
        result = json.load(tar.extractfile('results/comparison.json'))
        offline = [m for m in tar.getmembers() if '/offline-run-' in m.name and m.isfile()]
        if offline:
            dest = ROOT/'results'/('offline-'+state['job'])
            for member in offline:
                path = dest/member.name; assert path.resolve().is_relative_to(dest.resolve())
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(tar.extractfile(member).read())
            for path in sorted((dest/'results/wandb').glob('offline-run-*')):
                command(['wandb', 'sync', str(path)])
    assert result['complete'] and result['source'] == state['source']
    import wandb
    api = wandb.Api(timeout=30)
    parent = api.run('yibozhong657-none/nanochat/'+result['run_id'])
    assert parent.state == 'finished' and parent.summary['complete'] and parent.summary['checks_passed']
    assert parent.group == 'mhc-scale-1b-2b-20261030'
    for group in ('trials', 'probes'):
        for value in result[group].values():
            run = api.run('yibozhong657-none/nanochat/'+value['run_id'])
            assert run.group == parent.group and run.summary['complete']
            expected_state = 'failed' if value['status'] == 'oom' else 'finished'
            assert run.state == expected_state, (value['run_id'], run.state, expected_state)
            assert run.summary['status'] == value['status']
            assert run.summary['training_tokens'] == value['training_tokens']
            if group == 'trials':
                assert math.isclose(run.summary['normal_step_ms'], value['normal_ms']['mean'], rel_tol=1e-8)
    result.update(wandb_verified=True, job=state['job'])
    for path in (ROOT/'comparison.json', REPO/'research/connections/scale/results.json'):
        path.write_text(json.dumps(result, indent=2)+'\n')
    analysis = command([sys.executable, str(REPO/'research/connections/scale/analyze.py'), str(ROOT)])
    (ROOT/'analysis-summary.json').write_text(analysis+'\n')
    state.update(status='complete', push_status='pending_review'); save(state)


def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = 0
        while True:
            state = json.loads(STATE.read_text()); state['monitor_pid'] = os.getpid()
            if state['status'] in ('complete', 'needs_attention'):
                return
            try:
                row = job(state['job']); state['last_job'] = row
                if row and (row['JobStatus'] in (3, 5) or row['JobStatus'] == 4 and row.get('ExitCode') != 0):
                    raise RuntimeError(row)
                if row and row['JobStatus'] == 4:
                    collect(state); return
                errors = 0; state.update(monitor_errors=0, last_poll=time.time()); save(state)
            except Exception as exc:
                errors += 1; state.update(monitor_errors=errors, last_error=repr(exc))
                print(repr(exc), flush=True)
                if errors >= 5 or isinstance(exc, (AssertionError, RuntimeError)):
                    state['status'] = 'needs_attention'; save(state); return
                save(state)
            time.sleep(30)


if __name__ == '__main__':
    main()
