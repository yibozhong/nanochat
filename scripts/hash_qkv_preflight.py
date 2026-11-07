"""Validate the actual training/resume entrypoint before a fresh formal run.

Runs once per Hopper model family; diagnostic weights never enter checkpoints/.
The coordinator imports no torch while its children own the GPU.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def run(args, root, name, expected=0):
    print('PREFLIGHT_BEGIN '+name, flush=True)
    with (root/(name+'.log')).open('w') as log:
        process=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    assert code==expected,(name,code,expected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('H100','H200'),required=True)
    a=p.parse_args();root=Path('results')/('hash-qkv-preflight-'+a.family)
    root.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path('source.json').read_text())
    marker=root/'passed.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        assert previous['complete'] and previous['source']==source and previous['family']==a.family
        print('PREFLIGHT_REUSED '+a.family,flush=True);return
    checkpoints=Path('../work')/('hash-qkv-preflight-checkpoints-'+a.family)
    if checkpoints.exists():
        raise RuntimeError('Unexpected leftover diagnostic checkpoint; refusing to silently reuse it')
    run(['-m','pytest','-v','tests/test_hash_qkv.py','--junitxml='+str(root/'tests.xml')],root,'checks')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=12,failures=0,errors=0,skipped=0)
    common=['-m','scripts.hash_qkv_train','--steps','60','--schedule-steps','305176',
        '--microbatch','16','--seed','42','--attention-backend','hardware_auto',
        '--data-dir','../work/preflight-data/data','--output-dir',str(root/'restart'),
        '--checkpoint-dir',str(checkpoints),'--run-id','hash-qkv-preflight-'+a.family,
        '--segment-seconds','3600']
    run(common+['--segment-steps','25'],root,'first-25',85)
    progress=json.loads((root/'restart/progress.json').read_text())
    assert progress['step']==25 and progress['source']==source
    # The second process must restore this run's own optimizer/RNG/data position.
    run(common+['--resume','auto'],root,'resume-to-60')
    summary=json.loads((root/'restart/summary.json').read_text())
    config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==60 and summary['variant']=='hash_qkv'
    assert summary['source']==source and summary['training_tokens']==60*32768
    assert config['logging_backend']=='local' and config['parameters']==153092096
    assert config['model']['hash_qkv']['lookup_sha256']==source['lookup_sha256']
    assert [r['step'] for r in rows]==list(range(1,61))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']
    # Only diagnostic checkpoints in the explicitly separate work directory.
    shutil.rmtree(checkpoints)
    value=dict(complete=True,family=a.family,source=source,checks=counts,restart=summary,
        formal_weights_used=False,diagnostic_checkpoints_removed=True)
    temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(marker)
    print('PREFLIGHT_PASSED '+a.family,flush=True)


if __name__=='__main__':main()
