"""Regression only after retiring statistics variants; no formal training."""
import json,os,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.connection_moment_gate import execute
from nanochat.packed_training import json_write


def main():
    assert not os.environ.get('WANDB_API_KEY')
    execute([sys.executable,'-m','pytest','tests/test_moment_attention.py','tests/test_jensen_attention.py',
        'tests/test_triton_moments.py','tests/test_triton_resume.py','-v',
        '--junitxml=results/cleanup-tests.xml'],'cleanup-tests')
    suites=ET.parse('results/cleanup-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=35,failures=0,errors=0,skipped=0),checks
    assert not list(Path('results').rglob('*.wandb'))
    json_write('results/cleanup-gate.json',dict(complete=True,checks=checks,
        source=json.loads(Path('source.json').read_text()),formal_training=False,weights_returned=False,
        supported_statistics=['none','variance','jensen_silu']))


if __name__=='__main__':main()
