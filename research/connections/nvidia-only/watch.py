"""Collect the reviewed cleanup gate and publish only after all GPU checks pass."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tarfile
import termios
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
STATE=ROOT/'state.json'
REPORT=REPO/'research/connections/nvidia-only/validation.json'


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(s):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)


def push():
    secret=Path('/home/yzhong239/.config/gradformer/github.token').read_text().strip()
    pid,fd=pty.fork()
    if pid==0:
        os.chdir(REPO)
        attrs=termios.tcgetattr(0);attrs[3]&=~termios.ECHO;termios.tcsetattr(0,termios.TCSANOW,attrs)
        os.environ['GIT_ASKPASS']='';os.environ['SSH_ASKPASS']=''
        os.execvp('git',['git','-c','credential.helper=','-c','credential.username=yibozhong','push','origin','main'])
    buf=b'';sent=False;deadline=time.time()+55;status=None
    try:
        while time.time()<deadline:
            ready,_,_=select.select([fd],[],[],1)
            if ready:
                try:data=os.read(fd,65536)
                except OSError:break
                if not data:break
                buf+=data
                if b'Password' in buf and not sent:
                    os.write(fd,(secret+'\n').encode());sent=True
            else:
                done,status=os.waitpid(pid,os.WNOHANG)
                if done:break
        if status is None:
            done,status=os.waitpid(pid,os.WNOHANG)
            if not done:
                os.kill(pid,15);os.waitpid(pid,0);raise RuntimeError('Git push timeout')
        assert os.waitstatus_to_exitcode(status)==0,buf.decode(errors='replace').replace(secret,'[redacted]')[-1000:]
    finally:os.close(fd)
    head=command(['git','rev-parse','HEAD'],REPO)
    remote=command(['git','ls-remote','origin','refs/heads/main'],REPO).split()[0]
    assert head==remote
    return head


def collect(s):
    archive=ROOT/'results'/f'gate-{s["job"]}.tar'
    with tarfile.open(archive) as tar:
        value=json.load(tar.extractfile('results/cleanup-gate.json'))
        assert value['complete'] and value['source']==s['source']
        assert value['checks']==dict(tests=25,failures=0,errors=0,skipped=0)
        assert value['retired_packages_absent'] and value['logging_backend']=='local'
        assert not any('wandb' in Path(m.name).parts or m.name.endswith('.wandb') for m in tar)
        parity={name:json.load(tar.extractfile(name)) for name in (
            'results/controlled-parity/baseline.json','results/controlled-parity/mhc.json','results/cleanup-checks/mhc.json')}
        assert all(v['initial_weights_bitwise_equal'] and max(v['gradient_relative_errors'].values())<3e-4 for v in parity.values())
    value.update(job=s['job'],parity=parity,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/'validation.json').write_text(json.dumps(value,indent=2)+'\n')
    # Do not publish over unrelated edits or a new user-directed code revision.
    assert command(['git','rev-parse','HEAD'],REPO)==s['reviewed_head']
    assert not command(['git','status','--porcelain'],REPO)
    REPORT.write_text(json.dumps(value,indent=2)+'\n')
    command(['git','add',str(REPORT)],REPO)
    command(['git','commit','-m','Verify NVIDIA-only cleanup against the running training revision'],REPO)
    s.update(status='validated_pending_push',validation=value,reviewed_head=command(['git','rev-parse','HEAD'],REPO));save(s)
    s.update(pushed_commit=push(),status='complete');save(s)
    agents=Path('/home/yzhong239/AGENTS.md')
    with agents.open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        text=f.read();heading='## 最新代码清理授权：仅保留NVIDIA mHC（2026-10-31）\n'
        note=f'\n清理已完成并推送main {s["pushed_commit"]}：GPU gate{s["job"]}全部25项通过，无W&B；D512/8层/2K的旧新版初始化逐位相同，loss/全参数梯度对照通过。报告research/connections/nvidia-only/validation.json。旧DeepSeek/eager/compile/projection-only/no-manifold代码、入口、依赖已删除，NVIDIA类已独立。两个10B训练继续固定699dcf2。此段覆盖本节所有尚未推送/排队描述，勿重启清理gate。\n'
        assert heading in text;f.seek(0);f.write(text.replace(heading,heading+note,1));f.truncate()


def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);errors=0
        while True:
            s=json.loads(STATE.read_text());s['monitor_pid']=os.getpid()
            if s['status'] in ('complete','needs_attention'):return
            try:
                if s['status']=='validated_pending_push':
                    assert command(['git','rev-parse','HEAD'],REPO)==s['reviewed_head']
                    assert not command(['git','status','--porcelain'],REPO)
                    s.update(pushed_commit=push(),status='complete');save(s);return
                attrs='ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost'
                rows=json.loads(command(['condor_q',s['job'],'-json','-attributes',attrs]) or '[]')
                if not rows:
                    rows=json.loads(command(['condor_history',s['job'],'-limit','1','-scanlimit','10000','-json','-attributes',attrs]) or '[]')
                row=rows[0] if rows else None;s['last_job']=row
                if row and (row['JobStatus'] in (3,5) or row['JobStatus']==4 and row.get('ExitCode')!=0):raise RuntimeError(row)
                if row and row['JobStatus']==4:collect(s);return
                errors=0;s.update(last_poll=time.time(),monitor_errors=0);save(s)
            except Exception as exc:
                errors+=1;s.update(last_error=repr(exc),monitor_errors=errors)
                print(repr(exc),flush=True)
                if errors>=5 or isinstance(exc,(AssertionError,RuntimeError)):
                    s['status']='needs_attention';save(s);return
                save(s)
            time.sleep(30)


if __name__=='__main__':main()
