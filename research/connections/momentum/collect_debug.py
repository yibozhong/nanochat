"""Bounded read-only debug result collector; no submissions, retries or W&B."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


def main():
    root=Path(sys.argv[1]).resolve()
    lock=(root/'collector.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((root/'state.json').read_text());job=state['job']
    deadline=time.time()+14400
    while time.time()<deadline:
        if any((root/n).exists() for n in ('STOPPED','STOPPED.json')):return
        ads=[]
        for args in (['condor_q',job,'-json'], ['condor_history',job,'-limit','1','-json']):
            p=subprocess.run(args,capture_output=True,text=True,timeout=30)
            if p.returncode==0:ads=json.loads(p.stdout or '[]')
            if ads:break
        ad=ads[0] if ads else {};status=ad.get('JobStatus')
        progress=dict(job=job,job_status=status,checked_at=time.time(),formal_training=False)
        (root/'collector-status.json').write_text(json.dumps(progress,indent=2)+'\n')
        if status not in (3,4,5):
            time.sleep(20);continue
        report=dict(progress,complete=False,exit_code=ad.get('ExitCode'),hold_reason=ad.get('HoldReason'))
        path=root/'results'/f'stream-{job}.tar'
        if path.is_file():
            with tarfile.open(path) as archive:
                for key,name in [('source','source.json'),('profile','momentum-profile/summary.json'),('resume','momentum-cli-resume.json')]:
                    try:
                        member=archive.getmember('results/'+name)
                        assert member.size<2_000_000
                        report[key]=json.load(archive.extractfile(member))
                    except KeyError:pass
                try:
                    suites=list(ET.parse(archive.extractfile('results/momentum-tests.xml')).getroot().iter('testsuite'))
                    report['tests']={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
                except KeyError:pass
        tests=report.get('tests',{})
        report['complete']=(status==4 and ad.get('ExitCode')==0 and tests.get('tests')==57
            and all(tests.get(k)==0 for k in ('failures','errors','skipped'))
            and report.get('profile',{}).get('complete',False) and report.get('resume',{}).get('complete',False)
            and report.get('source')==state['source'])
        (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status='validated' if report['complete'] else 'validation_stopped',validation_complete=report['complete'])
        (root/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        line=f"\nTransport+RMS 有限GPU验证回收（{job}）：complete={report['complete']}，JobStatus={status}，ExitCode={ad.get('ExitCode')}；报告 {root/'validation.json'}。"
        if report['complete']:
            profile=report['profile'];target=profile['models']['ngpt_heavy_ball']
            line+=f" {tests['tests']}项检查通过，128步finite，dev CE {target['initial_dev_ce']:.6f}→{target['final_dev_ce']:.6f}，同卡相对ambient耗时{profile['time_ratio_to_ambient']:.4f}；CLI严格恢复通过。"
        line+=' 无正式训练、无W&B、无自动重提。\n'
        with Path('/home/yzhong239/RESEARCH_HISTORY.md').open('a') as f:f.write(line)
        print(json.dumps({k:report[k] for k in ('job','job_status','complete','exit_code')}),flush=True)
        return
    (root/'collector-status.json').write_text(json.dumps(dict(status='collector_timeout',job=job,formal_training=False))+'\n')


if __name__=='__main__':main()
