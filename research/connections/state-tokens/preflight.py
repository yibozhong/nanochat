"""Validate a previously untested GPU family before creating the formal run."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--backend', choices=('fa2','fa3'), required=True)
    a = p.parse_args()
    source = json.loads((a.source/'source.json').read_text())
    assert a.microbatch == source['microbatch']
    validated = source['profile_reference']['state_tokens']['actual_attention_backend']
    if a.backend == validated:
        print('Using GPU family already covered by full gate:', a.backend, flush=True)
        return
    root = a.results/('preflight-'+a.backend)
    marker = root/'passed.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['source'] == source and saved['microbatch'] == a.microbatch
        assert saved['backend'] == a.backend and saved['complete']
        return
    root.mkdir(parents=True, exist_ok=True)
    # The diagnostic subprocess has local logging and no credential environment.
    # Its working directory is inside source, away from the transferred .netrc.
    work = a.source/'preflight-work'; work.mkdir(exist_ok=True)
    (work/'source.json').write_text(json.dumps(source))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(a.source), WANDB_MODE='disabled', NETRC='/dev/null')
    env.pop('WANDB_API_KEY', None)
    def run(args, name):
        with (root/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,*args], cwd=work, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    tests = [str(a.source/'tests/test_state_tokens.py'), str(a.source/'tests/test_state_attention.py')]
    code = ('from nanochat.connection_experiment import configure_attention; '
            'configure_attention('+repr(a.backend)+');import pytest;raise SystemExit(pytest.main('
            +repr(['-v',*tests,'--junitxml='+str(root/'tests.xml')])+'))')
    run(['-c',code], 'tests')
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts == dict(tests=26,failures=0,errors=0,skipped=0)
    run(['-m','scripts.connection_context_profile','--variant','state_tokens',
         '--sequence-length','2048','--microbatch',str(a.microbatch),
         '--attention-backend',a.backend,'--state-attention-backend','triton',
         '--data-dir',str(a.source.parent/'data'),'--output',str(root/'profile.json')], 'training-check')
    profile = json.loads((root/'profile.json').read_text())
    assert profile['complete'] and profile['causality_passed'] and len(profile['losses']) == 60
    assert all(math.isfinite(x) for x in profile['losses'])
    assert profile['peak_reserved_bytes'] < .9*profile['device_total_bytes']
    marker.write_text(json.dumps(dict(complete=True,source=source,microbatch=a.microbatch,
                                     backend=a.backend,checks=counts),indent=2)+'\n')
    print('New GPU family validated before formal training:',a.backend,flush=True)


if __name__ == '__main__': main()
