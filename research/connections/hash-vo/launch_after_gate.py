"""Collect once, audit the exact tested source, publish, launch one authorized VO10B."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

REPO=Path(__file__).resolve().parents[3]
TRAIN=Path('/home/yzhong239/hash-vo-10b-20261107')


def command(args):
    return subprocess.check_output(args,cwd=REPO,text=True,timeout=60).strip()


def main(root):
    root=Path(root).resolve()
    with (root/'automatic.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            runpy.run_path(str(REPO/'research/connections/hash-vo/collect.py'))['main'](root)
            if (root/'STOPPED.json').exists() or (TRAIN/'STOPPED.json').exists():return
            state=json.loads((root/'state.json').read_text())
            if state['status']!='complete':return
            formal=runpy.run_path(str(REPO/'research/connections/hash-vo-train/submit.py'))
            report=json.loads((root/'collected/results/hash-vo/report.json').read_text())
            formal['validate_gate'](report,state)
            assert not command(['git','status','--porcelain'])
            assert command(['git','rev-parse','HEAD'])==state['source']['commit']
            paths=command(['git','ls-files','nanochat','scripts','tests','research/connections/runtime',
                           'research/connections/toolchain','research/connections/hash-vo/runtime.sh',
                           'research/connections/hash-vo/worker.sh','research/connections/hash-qkv/lookup.npy']).splitlines()
            execution={}
            with tarfile.open(root/'bundles/source.tar') as tar:
                for name in paths:
                    raw=(REPO/name).read_bytes()
                    assert raw==tar.extractfile('nanochat-src/'+name).read(),name
                    execution[name]=hashlib.sha256(raw).hexdigest()
            before=json.loads((root/'preservation-before.json').read_text())
            for name,digest in before['files'].items():
                assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
            audit=dict(source=state['source'],job=state['job'],execution_sha256=execution,
                existing_production_files=before['files'],worker_restart=report['worker_restart'])
            dest=REPO/'research/connections/hash-vo'
            (dest/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
            (dest/'gpu-source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            summary={k:report[k] for k in ('baseline_ms','hash_vo_ms','overhead_percent','worker_restart')}
            summary.update(parameters=121634816,gate_job=state['job'],gpu_model_source=state['source']['commit'])
            (dest/'performance-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            command(['git','add',str(dest/'validation.json'),str(dest/'gpu-source-audit.json'),str(dest/'performance-summary.json')])
            command(['git','commit','-m','Validate VO16 math, ABBA speed and actual worker checkpoint restart'])
            saved_argv=sys.argv;sys.argv=['watch.py',str(root)]
            try:published=runpy.run_path(str(REPO/'research/connections/nvidia-only/watch.py'))['push']()
            finally:sys.argv=saved_argv
            (root/'published.json').write_text(json.dumps(dict(commit=published,time=time.time()),indent=2)+'\n')
            assert not (root/'STOPPED.json').exists() and not (TRAIN/'STOPPED.json').exists()
            formal['main'](TRAIN)
            with (TRAIN/'logs/watch-start.out').open('ab') as out, (TRAIN/'logs/watch-start.err').open('ab') as err:
                proc=subprocess.Popen([sys.executable,str(REPO/'research/connections/hash-vo-train/watch_start.py'),str(TRAIN)],
                    cwd=REPO,stdout=out,stderr=err,start_new_session=True)
            (TRAIN/'monitor.pid').write_text(str(proc.pid)+'\n')
            result=json.loads((TRAIN/'state.json').read_text())
            info=dict(status='training_submitted',job=result['job'],run_id=result['run_id'],watch_pid=proc.pid,
                      gate_job=state['job'],published=published)
            (root/'automatic-result.json').write_text(json.dumps(info,indent=2)+'\n')
            print(json.dumps(info),flush=True)
            agents=Path('/home/yzhong239/AGENTS.md')
            with agents.open('r+') as f:
                fcntl.flock(f,fcntl.LOCK_EX);text=f.read()
                entry=(f"VO gate {state['job']} 已exit0：15检查及重启后15检查、ABBA各60步、完整worker exit85/25→60恢复通过；"
                       f"baseline {report['baseline_ms']:.4f}ms/VO {report['hash_vo_ms']:.4f}ms。已push {published}，"
                       f"唯一正式 {result['job']}/W&B {result['run_id']} 已提交，root hash-vo-10b-20261107；"
                       f"首正式step watcher PID{proc.pid}/300秒，正式启动尚待观察。\n\n")
                text=text.replace('## 新授权：停止QKV16，改为VO16验证后单一10B（2026-11-07）\n\n',
                    '## 新授权：停止QKV16，改为VO16验证后单一10B（2026-11-07）\n\n'+entry,1)
                f.seek(0);f.write(text);f.truncate()
        except BaseException as error:
            (root/'automatic-result.json').write_text(json.dumps(dict(status='needs_attention',error=repr(error)),indent=2)+'\n')
            raise


if __name__=='__main__':main(sys.argv[1])
