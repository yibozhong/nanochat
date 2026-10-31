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
    assert result['complete'] and result['source'] == state['source']
    for group in ('trials', 'probes'):
        for value in result[group].values():
            assert value['complete'] and value['source'] == state['source']
            assert value['status'] in ('complete', 'fits', 'insufficient_headroom', 'oom')
            assert value['training_tokens'] == value['optimizer_updates'] * 32768
    result.update(local_verified=True, job=state['job'])
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
