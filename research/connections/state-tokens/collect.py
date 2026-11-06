"""Collect one existing gate every five minutes; never submit/train/publish."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main(root):
    root = Path(root).resolve()
    with (root/'collect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((root/'state.json').read_text())
        job = state['job']
        archive = root/'results'/('gate-'+job+'.tar')
        def save(status, **extra):
            state.update(status=status, **extra)
            temp = root/'state.tmp'
            temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(root/'state.json')
            print(status, extra, flush=True)
        try:
            for _ in range(288):
                if (root/'STOPPED.json').exists(): return
                queue = json.loads(subprocess.check_output(['condor_q', job, '-json'], text=True))
                if queue:
                    if queue[0]['JobStatus'] == 5:
                        raise RuntimeError(queue[0].get('HoldReason', 'Held gate'))
                else:
                    history = json.loads(subprocess.check_output(
                        ['condor_history', job, '-limit', '1', '-json'], text=True))
                    if history:
                        assert history[0]['JobStatus']==4 and history[0].get('ExitCode')==0, history
                        with tarfile.open(archive) as bundle:
                            result = json.load(bundle.extractfile('results/state-tokens/validation.json'))
                            assert result['complete'] and not result['formal_training']
                            assert result['checks']==dict(tests=9, failures=0, errors=0, skipped=0)
                            assert result['source']==state['source']
                            (root/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
                        save('validation_complete', collector_complete=True, formal_training=False)
                        return
                time.sleep(300)
            raise TimeoutError('24h collection window elapsed; no automatic resubmission')
        except Exception as exc:
            save('needs_attention', collector_complete=True, error=repr(exc), formal_training=False)
            raise


if __name__ == '__main__': main(sys.argv[1])
