"""Read-only job collector; never submits, trains, changes thresholds or publishes."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def command(args):
    return subprocess.check_output(args, text=True, timeout=45).strip()


def collect(root, state, ad):
    archive = root/'results'/('bench-'+state['job']+'.tar')
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not member.issym() and not member.islnk()
            assert path.parts[0] == 'results' and path.suffix not in ('.pt', '.pth', '.ckpt', '.wandb')
            if member.isfile():
                target = root/'collected'/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    report = root/'collected/results/hash-kernel-bench/report.json'
    state['status'] = 'needs_attention'
    if report.exists():
        value = json.loads(report.read_text())
        assert value['source'] == state['source'] and not value['formal_training']
        if ad.get('ExitCode') == 0 and value['complete']:
            assert len(value['checks']) == 10 and all(c['passed'] for c in value['checks'])
            assert value['finished_all_cases'] and len(value['results']) == 58 and not value['failures']
            state['status'] = 'complete'
        state['report'] = str(report)
    state['exit_code'] = ad.get('ExitCode')
    return state


def main(root):
    root = Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state = json.loads((root/'state.json').read_text())
            if state['status'] in ('complete', 'needs_attention'):
                return
            try:
                attrs = 'ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                rows = json.loads(command(['condor_q', state['job'], '-json', '-attributes', attrs]) or '[]')
                if not rows:
                    rows = json.loads(command(['condor_history', state['job'], '-limit', '1', '-scanlimit', '10000', '-json', '-attributes', attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None, last_poll=time.time())
                if rows and rows[0]['JobStatus'] == 4:
                    state = collect(root, state, rows[0])
                elif rows and rows[0]['JobStatus'] in (3, 5):
                    state['status'] = 'needs_attention'
            except Exception as error:
                state.update(status='needs_attention', error=repr(error))
            temporary = root/'state.tmp'
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(root/'state.json')
            if state['status'] in ('complete', 'needs_attention'):
                return
            time.sleep(300)


if __name__ == '__main__':
    main(sys.argv[1])
