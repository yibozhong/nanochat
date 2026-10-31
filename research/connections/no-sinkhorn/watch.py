"""Audit the local gate, publish the reviewed change, release and track one run."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tarfile
import termios
import time

ROOT = Path(sys.argv[1]).resolve()
REPO = Path(__file__).resolve().parents[3]
STATE = ROOT/'state.json'


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def save(state):
    p = STATE.with_suffix('.tmp')
    p.write_text(json.dumps(state, indent=2)+'\n')
    p.replace(STATE)


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
                done,code=os.waitpid(pid,os.WNOHANG)
                if done:
                    status=code;break
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


def job(ident):
    attrs = 'ClusterId,ProcId,JobStatus,ExitCode,HoldReason,HoldReasonCode,NumJobStarts,RemoteHost,NoSinkhornKey,NoSinkhornRole'
    rows = json.loads(command(['condor_q', ident, '-json', '-attributes', attrs]) or '[]')
    if not rows:
        rows = json.loads(command(['condor_history', ident, '-limit', '1', '-scanlimit', '10000',
                                   '-json', '-attributes', attrs]) or '[]')
    if rows:
        assert rows[0]['NoSinkhornKey'] == 'mhc-no-sinkhorn-20261031'
    return rows[0] if rows else None


def immutable_reference_inputs():
    values = json.loads((ROOT/'immutable-reference-inputs.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in values.items())


def verify_gate(s):
    path = ROOT/'results'/f'gate-{s["gate_job"]}.tar'
    with tarfile.open(path) as archive:
        value = json.load(archive.extractfile('results/no-sinkhorn-gate.json'))
        assert value['complete'] and value['logging_backend'] == 'local'
        assert value['source'] == s['source']
        assert value['checks'] == dict(tests=9, failures=0, errors=0, skipped=0)
        trial = value['trial']
        assert trial['steps'] == value['unique_training_steps'] == 50
        assert trial['complete'] and trial['training_tokens'] == 50*32768
        assert math.isfinite(trial['final_dev_ce']) and trial['final_dev_ce'] < trial['initial_dev_ce']
        assert value['config']['applied_sinkhorn_iterations'] == 0
        assert value['config']['residual_mixing'] == 'unconstrained_raw'
        assert value['config']['protocol']['schedule_steps'] == 305176
        assert not any('wandb' in Path(m.name).parts or m.name.endswith('.wandb') for m in archive)
        value['unit_results'] = {Path(m.name).name: json.load(archive.extractfile(m))
            for m in archive.getmembers() if m.isfile() and m.name.startswith('results/no-sinkhorn-checks/') and m.name.endswith('.json')}
        assert value['unit_results']['initialization.json']['all_parameters_bitwise_equal']
        assert value['unit_results']['initialization.json']['optimizer_groups_equal']
    immutable_reference_inputs()
    value['archive_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    value['gate_job'] = s['gate_job']
    (ROOT/'gate.json').write_text(json.dumps(value, indent=2)+'\n')
    s.update(gate_verified=True, status='gate_passed_training_held')
    save(s)


def publish_and_release(s):
    if not s.get('report_committed'):
        assert command(['git', 'rev-parse', 'HEAD'], REPO) == s['reviewed_head']
        assert not command(['git', 'status', '--porcelain'], REPO), 'Unexpected working-tree edits; review before publishing'
        report = REPO/'research/connections/no-sinkhorn/gate-results.json'
        report.write_text((ROOT/'gate.json').read_text())
        command(['git', 'add', str(report)], REPO)
        command(['git', 'commit', '-m', 'Validate NVIDIA no-Sinkhorn ablation and training restart'], REPO)
        s.update(report_committed=True, reviewed_head=command(['git', 'rev-parse', 'HEAD'], REPO))
        save(s)
    if not s.get('pushed_commit'):
        assert command(['git', 'rev-parse', 'HEAD'], REPO) == s['reviewed_head']
        assert not command(['git', 'status', '--porcelain'], REPO)
        s['pushed_commit'] = push()
        save(s)
    if not s['training_released']:
        immutable_reference_inputs()
        row = job(s['train_job'])
        assert row is not None and row['NoSinkhornRole'] == 'formal'
        if row['JobStatus'] == 5:
            assert row['HoldReasonCode'] == 15, row
            command(['condor_release', s['train_job']])
        else:
            assert row['JobStatus'] in (1, 2, 4), row
        s.update(training_released=True, status='training_released', released_at=time.time())
        save(s)
        p = Path('/home/yzhong239/AGENTS.md')
        with p.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            txt = f.read()
            heading = '## 新授权：NVIDIA mHC 去掉 Sinkhorn 的10B消融（2026-10-31）\n'
            note = f'\n最新：gate{s["gate_job"]}已exit0，9检查及25→50真实恢复通过，报告/实现已推送main {s["pushed_commit"]}。正式{s["train_job"]}已自动release，W&B预分配{s["run_id"]}，305176steps/10B；release不等于已经执行训练。监控继续跟踪同一job，勿重复提交。\n'
            assert heading in txt
            f.seek(0);f.write(txt.replace(heading, heading+note, 1));f.truncate()


def collect_training(s):
    path = ROOT/'results'/f'train-{s["train_job"]}.tar'
    with tarfile.open(path) as archive:
        result = json.load(archive.extractfile('results/train/summary.json'))
        config = json.load(archive.extractfile('results/train/config.json'))
        offline = [m for m in archive.getmembers() if '/offline-run-' in m.name and m.isfile()]
        if offline:
            dest = ROOT/'results'/'formal-offline'
            for member in offline:
                target = dest/member.name
                assert target.resolve().is_relative_to(dest.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
            for run in sorted((dest/'results/wandb').glob('offline-run-*')):
                command(['wandb', 'sync', str(run)])
    assert result['complete'] and result['source'] == s['source']
    assert result['variant'] == 'mhc_no_sinkhorn' and result['run_id'] == s['run_id']
    assert result['steps'] == s['expected_steps'] and result['training_tokens'] == s['expected_tokens']
    assert result['test_evaluated'] and result['test_tokens'] == 2097152
    assert config['logging_backend'] == 'wandb' and config['applied_sinkhorn_iterations'] == 0
    checkpoint = ROOT/'results'/f'train-{s["train_job"]}-checkpoints.tar'
    with tarfile.open(checkpoint) as archive:
        for name in ('latest.pt', 'best.pt'):
            assert archive.getmember('checkpoints/'+name).size == result['checkpoint_files'][name]['bytes']
    import wandb
    run = wandb.Api(timeout=30).run('yibozhong657-none/nanochat/'+s['run_id'])
    if run.state != 'finished' or not math.isclose(run.summary.get('final_test_ce', float('nan')), result['final_test_ce'], rel_tol=1e-8):
        raise ValueError('Waiting for formal W&B final summary synchronization')
    result.update(wandb_verified=True, checkpoint_archive=str(checkpoint))
    (ROOT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    s.update(status='complete', result=result)
    save(s)


def main():
    with (ROOT/'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors = 0
        while True:
            s = json.loads(STATE.read_text())
            s['monitor_pid'] = os.getpid()
            if s['status'] in ('complete', 'needs_attention'):
                return
            try:
                if not s['gate_verified']:
                    row = job(s['gate_job'])
                    s['last_gate_job'] = row
                    if row and (row['JobStatus'] in (3, 5) or row['JobStatus'] == 4 and row.get('ExitCode') != 0):
                        raise RuntimeError(row)
                    if row and row['JobStatus'] == 4:
                        verify_gate(s)
                if s['gate_verified']:
                    publish_and_release(s)
                row = job(s['train_job'])
                s['last_training_job'] = row
                if row and (row['JobStatus'] == 3 or row['JobStatus'] == 4 and row.get('ExitCode') != 0):
                    raise RuntimeError(row)
                if row and row['JobStatus'] == 5 and (s['training_released'] or row.get('HoldReasonCode') != 15):
                    raise RuntimeError(row)
                if s['training_released'] and row and row['JobStatus'] == 4:
                    collect_training(s)
                    return
                errors = 0
                s.update(last_poll=time.time(), monitor_errors=0)
                save(s)
            except Exception as exc:
                errors += 1
                s.update(last_error=repr(exc), monitor_errors=errors)
                print(repr(exc), flush=True)
                if errors >= 5 or isinstance(exc, (AssertionError, RuntimeError)):
                    s['status'] = 'needs_attention'
                    save(s)
                    return
                save(s)
            time.sleep(30)


if __name__ == '__main__':
    main()
