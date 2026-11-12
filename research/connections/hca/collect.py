"""Read-only result collector; never retries jobs or submits training."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import tarfile
import time


def once(root):
    state = json.loads((root/'state.json').read_text())
    archive = root/'results'/('hca-'+state['job']+'.tar')
    if archive.exists():
        reports = root/'reports'; reports.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            # Only named small diagnostic artifacts, no arbitrary extraction.
            for name in ('results/hca/validation.json', 'results/hca/tests.xml',
                         'results/hca/test-counts.json', 'results/source.json',
                         'results/environment.txt', 'results/exit-code.txt'):
                try:
                    member = tar.getmember(name)
                except KeyError:
                    continue
                if not member.isfile() or member.size > 4*1024**2:
                    raise ValueError('Unexpected report member')
                (reports/Path(name).name).write_bytes(tar.extractfile(member).read())
        result_file = reports/'validation.json'
        complete = False
        if result_file.exists():
            result = json.loads(result_file.read_text())
            complete = (result.get('complete') is True and result.get('formal_training') is False
                        and result['source']['commit'] == state['source']['commit']
                        and result['tests']['tests'] > 0
                        and all(result['tests'][k] == 0 for k in ('failures','errors','skipped'))
                        and (reports/'exit-code.txt').read_text().strip() == '0')
        state['status'] = 'complete' if complete else 'needs_attention'
        state['reports'] = str(reports)
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
        return True
    query = json.loads(subprocess.check_output(['condor_q', state['job'], '-json', '-attributes',
        'ClusterId,ProcId,JobStatus,HoldReason,NumJobStarts'], text=True) or '[]')
    if query and query[0]['JobStatus'] == 5:
        state['status'] = 'needs_attention'; state['queue'] = query[0]
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
        return True
    return False


def main():
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--watch', action='store_true')
    a = p.parse_args(); root = Path(a.root).resolve()
    with (root/'collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Bound the collector as well. Expiry does not resubmit or alter a job.
        deadline = time.monotonic() + 24*3600
        while not (root/'STOPPED.json').exists():
            if once(root) or not a.watch or time.monotonic() > deadline:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
