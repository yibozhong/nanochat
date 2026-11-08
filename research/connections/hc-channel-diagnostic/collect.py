"""Collect a bounded frozen diagnostic; never train, resubmit or publish."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


def validate(value, state, tests):
    assert value['complete'] and not value['formal_training'] and value['optimizer_updates'] == 0
    assert value['source'] == state['source'] and value['dev_sequences'] == 64
    assert value['dev_tokens'] == 131072 and set(value['models']) == {'shared', 'scaled'}
    suites = list(tests.iter('testsuite'))
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    assert counts == dict(tests=5, failures=0, errors=0, skipped=0), counts
    for name, model in value['models'].items():
        assert model['config']['variant'] == ('mhc_no_sinkhorn' if name == 'shared' else 'scaled_hc')
        assert model['checkpoint_step'] == 305176 and len(model['geometry']) == 16
        assert all(model['checks'][k] for k in ('no_sinkhorn', 'observer_forward_exact', 'weights_and_buffers_unchanged'))
        assert max(model['checks']['decomposition_relative'].values()) < .01
        assert abs(model['historical_ce_difference']) < .002
        assert math.isfinite(model['original_ce']['mean'])
    assert len(value['models']['scaled']['interventions']) == 23
    for intervention in value['models']['scaled']['interventions'].values():
        assert math.isfinite(intervention['delta_ce']['mean'])


def main(root):
    root = Path(root).resolve()
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            s = json.loads((root/'state.json').read_text())
            if s['status'] in ('complete', 'needs_attention'): return
            s['monitor_pid'] = os.getpid()
            try:
                attrs = ['-json', '-attributes', 'ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason,NumJobStarts']
                def query(args): return json.loads(subprocess.check_output(args, text=True, timeout=45).strip() or '[]')
                rows = query(['condor_q', s['job'], *attrs])
                if not rows: rows = query(['condor_history', s['job'], '-limit', '1', *attrs])
                s.update(last_job=rows[0] if rows else None, last_poll=time.time())
                if rows and rows[0]['JobStatus'] == 4:
                    with tarfile.open(root/'results'/('diagnostic-'+s['job']+'.tar')) as tf:
                        for m in tf.getmembers():
                            p = Path(m.name)
                            assert not p.is_absolute() and '..' not in p.parts
                            assert not m.issym() and not m.islnk() and not m.name.endswith(('.pt', '.wandb'))
                            if m.isfile():
                                dest = root/'collected'/p; dest.parent.mkdir(parents=True, exist_ok=True)
                                dest.write_bytes(tf.extractfile(m).read())
                    assert rows[0].get('ExitCode') == 0, rows[0]
                    value = json.loads((root/'collected/results/hc-channel-diagnostic/diagnostic.json').read_text())
                    validate(value, s, ET.parse(root/'collected/results/unit-tests.xml').getroot())
                    for name, expected in json.loads((root/'retained-training.json').read_text()).items():
                        p = Path(name); st = p.stat()
                        assert dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=st.st_size,
                                    inode=st.st_ino, mtime_ns=st.st_mtime_ns) == expected, name
                    (root/'diagnostic.json').write_text(json.dumps(value, indent=2)+'\n')
                    s.update(status='complete', automatic_training=False)
                elif rows and rows[0]['JobStatus'] in (3, 5): s['status'] = 'needs_attention'
            except Exception as exc:
                s.update(status='needs_attention', error=repr(exc))
            p = root/'state.tmp'; p.write_text(json.dumps(s, indent=2)+'\n'); p.replace(root/'state.json')
            if s['status'] in ('complete', 'needs_attention'): return
            time.sleep(60)


if __name__ == '__main__': main(sys.argv[1])
