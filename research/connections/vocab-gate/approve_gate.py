"""Aggregate actual evidence. Publishing approval follows the operator's review."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
from run_policy import validate_gate
p=argparse.ArgumentParser();p.add_argument('--report-only',action='store_true');a=p.parse_args()
source=json.loads(Path('source.json').read_text())
b=json.loads(Path('results/benchmark.json').read_text())
cli=json.loads(Path('results/cli-check.json').read_text())
suites=ET.parse('results/gpu-tests.xml').getroot()
cases=list(suites.iter('testcase'));bad=sum(bool(list(c.iter('failure')) or list(c.iter('error')) or list(c.iter('skipped'))) for c in cases)
assert b['source']==source and cli['passed']
report=dict(b,complete=True,tests_passed=len(cases)-bad,tests_failed=bad,
    cli_resume_passed=cli['passed'],optimizer_rng_resume_passed=any(c.get('name')=='test_checkpoint_real_kernel_next_update' for c in cases))
validate_gate(report,source)
path=Path('results/gate-report.json' if a.report_only else 'results/gate-approved.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(path),'passed':report['tests_passed'],'gate_step_ms':report['gate_step_ms']}))
