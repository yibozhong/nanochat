"""Read-only result collection for the one bounded debug job; never resubmits."""
import fcntl
import json
from pathlib import Path,PurePosixPath
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET

ROOT=Path('/home/yzhong239/feedback-backprop-debug-20261114')


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def query(command):
    return json.loads(subprocess.check_output(command,text=True,timeout=30) or '[]')


def main():
    with (ROOT/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((ROOT/'state.json').read_text());job=state['job']
        for _ in range(5760):
            try:
                ads=query(['condor_q',job,'-json','-attributes','JobStatus,RemoteHost,NumJobStarts,HoldReason,ExitCode'])
                ad=ads[0] if ads else None
                archive=ROOT/'results'/f'stream-{job}.tar'
                if archive.exists() and (ad is None or ad.get('JobStatus') in (4,5)):
                    out=ROOT/'collected';out.mkdir(exist_ok=True)
                    with tarfile.open(archive) as tar:
                        for member in tar:
                            path=PurePosixPath(member.name)
                            assert not path.is_absolute() and '..' not in path.parts
                            if member.isfile() and path.parts[0]=='results':
                                target=out.joinpath(*path.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True)
                                target.write_bytes(tar.extractfile(member).read())
                    terminal=ad or (query(['condor_history',job,'-limit','1','-json','-attributes','ExitCode,JobStatus']) or [{}])[0]
                    state.update(terminal=terminal,finished_at=time.time(),status='debug_failed')
                    summary=out/'comparison/summary.json'
                    if summary.exists():
                        value=json.loads(summary.read_text());save(ROOT/'summary.json',value)
                        repair=out/'repair-success.json'
                        repaired=repair.exists() and json.loads(repair.read_text()).get('continuation_exit_code')==0
                        if value.get('complete') and (terminal.get('ExitCode')==0 or repaired):
                            tests=ET.parse(out/'tests.xml').getroot()
                            suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite'))
                            assert sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)==0
                            for source in ('embedding_qkv','per_layer_qkv'):
                                assert json.loads((out/f'cli-resume-{source}.json').read_text())['complete']
                            state.update(status='debug_passed_after_in_allocation_repair' if repaired else 'debug_complete',
                                tests=sum(int(s.attrib['tests']) for s in suites),comparisons=value['comparisons'])
                    save(ROOT/'state.json',state);print(json.dumps(state),flush=True);return
                if ad:
                    state.update(status={1:'queued',2:'running',5:'held'}.get(ad['JobStatus'],'transferring'),queue=ad,updated_at=time.time())
                    save(ROOT/'state.json',state)
                    if ad['JobStatus']==5:return
                time.sleep(30)
            except (subprocess.SubprocessError,OSError,tarfile.TarError,json.JSONDecodeError) as error:
                print(type(error).__name__+': '+str(error),flush=True);time.sleep(30)
        print('Collector deadline reached; no job mutation',flush=True)


if __name__=='__main__':main()
