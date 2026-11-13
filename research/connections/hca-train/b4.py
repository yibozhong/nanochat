"""One separately authorized Fixed B4 10B run; no new paired-plan runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from policy import validate_probe
from submit import write, stopped


def validate(report, commit):
    assert report['complete'] and report['formal_training'] is False
    assert report['commit']==commit and report['attention_mode']=='fixed_hca' and report['block_size']==4
    assert report['tests']==dict(tests=3,failures=0,errors=0,skipped=0)
    assert report['microbatch']==16 and report['hca_backend']=='triton'
    validate_probe(report['probe'],'fixed_hca',16,commit,block_size=4)
    speed=report['speed'];assert speed['complete'] and speed['formal_training'] is False
    assert speed['source']['commit']==commit and speed['microbatch']==16
    assert set(speed['modes'])=={'full','fixed_hca'}
    assert speed['modes']['fixed_hca']['config']['hca_block_size']==4
    assert 0<speed['modes']['fixed_hca']['relative_to_baseline']<10
    return 16


def preflight():
    source=json.loads(Path('source.json').read_text())
    assert source['attention_mode']=='fixed_hca' and source['hca']['target_ratio']==4
    assert source['formal_training'] and source['hca_backend']=='triton'
    out=Path('results/hca-startup');out.mkdir(parents=True,exist_ok=True)
    marker=out/'passed.json'
    if marker.exists():
        ready=json.loads(marker.read_text());validate(ready,source['commit'])
        assert ready['run_id']==source['run_id'];return
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_hca_b4.py',
                    '--junitxml='+str(out/'b4.xml')],check=True)
    suites=ET.parse(out/'b4.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    probe=out/'b4-replay.json'
    subprocess.run([sys.executable,'research/connections/hca-train/replay_check.py','--mode','fixed_hca',
                    '--microbatch','16','--backend','triton','--block-size','4','--output',str(probe)],check=True)
    subprocess.run([sys.executable,'research/connections/hca-train/speed.py','--microbatch','16',
                    '--block-size','4','--modes','full','fixed_hca'],check=True)
    report=dict(complete=True,formal_training=False,commit=source['commit'],attention_mode='fixed_hca',
        run_id=source['run_id'],block_size=4,microbatch=16,hca_backend='triton',tests=counts,
        probe=json.loads(probe.read_text()),speed=json.loads((out/'speed.json').read_text()))
    validate(report,source['commit']);write(marker,report)
    print('B4_STARTUP_PASSED '+json.dumps(report,allow_nan=False),flush=True)


def tick(root):
    from bridge import refresh_log, handoff
    from controller import consume, collect
    if stopped(root):return False
    state=json.loads((root/'state.json').read_text())
    if state['status'] in ('needs_attention','complete'):return False
    attrs='ClusterId,ProcId,JobStatus,HoldReason,RemoteHost,NumJobStarts,ExitCode,ExitBySignal'
    ads=json.loads(subprocess.check_output(['condor_q',state['job'],'-json','-attributes',attrs],text=True) or '[]')
    if not ads:
        ads=json.loads(subprocess.check_output(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs],text=True) or '[]')
    assert len(ads)==1;ad=ads[0];state['scheduler']=ad
    if 'bridge' in state and ad['JobStatus']==2:refresh_log(root,state)
    starts=ad.get('NumJobStarts')
    if starts is not None and starts!=state.get('observed_starts'):
        state.update(observed_starts=starts,stdout_offset=0,inside_formal_log=False)
    path=root/f'logs/train.{state["job"]}.out'
    if not state.get('startup_passed') and path.exists():
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith('B4_STARTUP_PASSED '):
                report=json.loads(line.split(' ',1)[1]);validate(report,state['source']['commit'])
                assert report['run_id']==state['run_id']
                write(root/'validated.json',report)
                state.update(startup_passed=True,microbatch=16,
                    estimated_train_only_hours=report['speed']['modes']['fixed_hca']['estimated_train_only_hours'])
    consume(root,state,dict(commit=state['source']['commit']))
    if state['status']=='needs_attention':pass
    elif ad['JobStatus'] in (3,5):state.update(status='needs_attention',failure=ad.get('HoldReason','Removed'))
    elif ad['JobStatus']==4:
        assert ad.get('ExitCode')==0 and not ad.get('ExitBySignal',False)
        if 'bridge' in state:handoff(root,state,ad,validator=validate,hold_prefix=state['bridge']['hold_prefix'])
        else:collect(root,state)
    elif ad['JobStatus']==2:
        state['status']='formal_training' if state['formal_started'] else 'startup_validation'
    elif ad['JobStatus']==1:
        state['status']='queued_for_resume' if state['formal_started'] else 'queued_for_startup_validation'
    state['updated_at']=time.time();write(root/'state.json',state)
    return state['status'] not in ('complete','needs_attention')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','monitor'));p.add_argument('--root',type=Path)
    a=p.parse_args()
    if a.action=='preflight':preflight();return
    with (a.root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (a.root/'controller.pid').write_text(str(os.getpid())+'\n')
        errors=0
        while True:
            try:
                if not tick(a.root):return
                errors=0
            except Exception:
                errors+=1;traceback.print_exc()
                if errors>=3:
                    write(a.root/'controller-failure.json',dict(error=traceback.format_exc(),time=time.time()));return
            time.sleep(60)


if __name__=='__main__':main()
