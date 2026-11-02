"""Collect the single bounded context benchmark; never submit training."""
import fcntl
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(state):
    tmp=ROOT/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'state.json')


def main():
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (ROOT/'STOPPED.json').exists():
            state=json.loads((ROOT/'state.json').read_text())
            if state['status'] in ('complete','needs_attention','stopped_by_user'):return
            try:
                attrs='ClusterId,ProcId,JobStatus,NumJobStarts,RemoteHost,ExitCode,HoldReason'
                ads=json.loads(command(['condor_q',state['job'],'-json','-attributes',attrs]) or '[]')
                if not ads:ads=json.loads(command(['condor_history',state['job'],'-limit','1','-json','-attributes',attrs]) or '[]')
                state.update(last_poll=time.time(),last_job=ads[0] if ads else None)
                save(state)
                if ads and ads[0]['JobStatus'] in (3,4,5):
                    assert ads[0]['JobStatus']==4 and ads[0].get('ExitCode')==0,ads
                    arc=ROOT/'results'/f'context-{state["job"]}.tar'
                    with tarfile.open(arc) as tf:
                        assert not any(x.name.endswith(('.pt','.wandb')) for x in tf)
                        result=json.load(tf.extractfile('results/context-results.json'))
                    assert result['complete'] and result['source']==state['source'] and result['logging_backend']=='local'
                    assert result['lengths']==[2048,4096,8192,16384] and result['variants']==['baseline','mhc']
                    assert len(result['trials'])==4*len(result['selected_microbatches'])
                    for trial in result['trials']:
                        assert trial['gpu_uuid']==result['gpu_uuid'] and trial['causality_passed']
                        assert trial['batch_tokens']==32768 and trial['optimizer_updates']==60 and len(trial['step_seconds'])==40
                        assert trial['status']=='fits' and trial['peak_reserved_bytes']<int(.9*22000*1024**2)
                    # Protect concurrent user edits and a later cancellation.
                    assert not (ROOT/'STOPPED.json').exists()
                    assert command(['git','rev-parse','HEAD'],REPO)==state['source']['commit']
                    assert not command(['git','status','--porcelain'],REPO)
                    report=REPO/'research/connections/context/results.json'
                    report.write_text(json.dumps(result,indent=2)+'\n')
                    (ROOT/'results.json').write_bytes(report.read_bytes())
                    command(['git','add',str(report)],REPO)
                    command(['git','commit','-m','Record same-GPU 2K through16K context performance'],REPO)
                    state['report_commit']=command(['git','rev-parse','HEAD'],REPO);save(state)
                    state['remote_commit']=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
                    state['status']='complete';state['summary']=result['summary'];state['unsupported']=result['unsupported'];save(state)
                    p=Path('/home/yzhong239/AGENTS.md');s=p.read_text()
                    heading='## 新授权：2K/4K/8K/16K同卡吞吐与显存测试（2026-11-01）\n'
                    msg=f'\n最新：job{state["job"]}正常exit0，性能/显存/因果检查和同卡审计完成，报告及源码已推送{state["remote_commit"]}。结果{ROOT}/results.json，state complete；无W&B、无checkpoint、无正式训练。未满足22GB预算的长度见unsupported，不能称全部可行。\n'
                    if heading in s:p.write_text(s.replace(heading,heading+msg,1))
                    return
            except Exception as exc:
                state.update(status='needs_attention',error=repr(exc));save(state);raise
            time.sleep(30)


if __name__=='__main__':main()
