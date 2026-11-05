"""Audit the GPU gate, publish it, then submit the two authorized 10B runs."""
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / 'research/connections/static-mixing'
VARIANTS = {'mhc_static_linear': 'static-mixing-linear-10b-20261105',
            'mhc_static_mlp': 'static-mixing-mlp-10b-20261105'}


def cmd(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def save(root, state):
    tmp = root / 'state.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(root / 'state.json')


def collect(root, state, ad):
    with tarfile.open(root / 'results' / ('gate-' + state['job'] + '.tar')) as tf:
        for member in tf.getmembers():
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert not member.name.endswith(('.pt', '.wandb'))
            if member.isfile():
                dest = root / 'collected' / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
    assert ad.get('ExitCode') == 0, ad
    value = json.loads((root / 'collected/results/static-mixing/validation.json').read_text())
    assert value['complete'] and not value['formal_training'] and value['source'] == state['source']
    assert value['checks'] == dict(tests=20, failures=0, errors=0, skipped=0)
    for variant in VARIANTS:
        assert value['restarts'][variant]['complete'] and value['restarts'][variant]['steps'] == 128
        assert value['profiles'][variant]['status'] == 'fits'
    assert cmd(['git', 'rev-parse', 'HEAD'], REPO) == state['source']['commit']
    assert not cmd(['git', 'status', '--porcelain'], REPO)
    report = WORKFLOW / 'validation.json'
    report.write_text(json.dumps(dict(value, job=state['job']), indent=2) + '\n')
    cmd(['git', 'add', str(report)], REPO)
    cmd(['git', 'commit', '-m', 'Validate static mHC residual mixers on GPU'], REPO)
    state.update(status='validated_pending_push', reviewed_head=cmd(['git', 'rev-parse', 'HEAD'], REPO))
    save(root, state)


def submit_training(root, state):
    state.setdefault('training', {})
    for variant, name in VARIANTS.items():
        train_root = root.parent / name
        subprocess.run([sys.executable, str(WORKFLOW / 'submit_train.py'), str(train_root), variant, str(root)],
                       check=True, cwd=REPO, timeout=120)
        training = json.loads((train_root / 'state.json').read_text())
        pid_path = train_root / 'monitor.pid'
        pid = int(pid_path.read_text()) if pid_path.exists() else None
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive and training['status'] not in ('training_started', 'needs_attention'):
            with (train_root / 'logs/watch-start.out').open('a') as out, (train_root / 'logs/watch-start.err').open('a') as err:
                proc = subprocess.Popen([sys.executable, str(WORKFLOW / 'watch_start.py'), str(train_root)],
                    cwd=REPO, stdout=out, stderr=err, start_new_session=True)
            pid = proc.pid
            pid_path.write_text(str(pid) + '\n')
        state['training'][variant] = dict(root=str(train_root), job=training['job'],
                                         run_id=training['run_id'], monitor_pid=pid)
        save(root, state)
    agents = Path('/home/yzhong239/AGENTS.md')
    with agents.open('r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        text = f.read()
        heading = '## 新授权：静态 dense / bottleneck mHC mixing 各10B（2026-11-05）\n'
        assert heading in text
        runs = '；'.join(f'{v}={s["job"]}/W&B{s["run_id"]}/首步watch PID{s["monitor_pid"]}' for v, s in state['training'].items())
        note = f'\nGPU gate{state["job"]}已exit0，20检查、同卡60step profile、两组25→128真实恢复通过；main {state["pushed_commit"]}已push核验。两正式10B已提交：{runs}。提交不等于已开跑；各watch每300秒检查至首finite step退出。短检查不证明长期效果/稳定性。\n'
        f.seek(0)
        f.write(text.replace(heading, heading + note, 1))
        f.truncate()
    state['status'] = 'complete'
    save(root, state)


def main(root):
    root = Path(root).resolve()
    with (root / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root / 'STOPPED.json').exists():
            state = json.loads((root / 'state.json').read_text())
            if state['status'] in ('complete', 'needs_attention'):
                return
            state['monitor_pid'] = os.getpid()
            try:
                if state['status'] == 'validated_pending_push':
                    assert cmd(['git', 'rev-parse', 'HEAD'], REPO) == state['reviewed_head']
                    assert not cmd(['git', 'status', '--porcelain'], REPO)
                    state['pushed_commit'] = runpy.run_path(str(REPO / 'research/connections/nvidia-only/watch.py'))['push']()
                    state['status'] = 'published'
                    save(root, state)
                if state['status'] == 'published':
                    submit_training(root, state)
                    return
                attrs = 'ClusterId,ProcId,JobStatus,ExitCode,RemoteHost,HoldReason'
                rows = json.loads(cmd(['condor_q', state['job'], '-json', '-attributes', attrs]) or '[]')
                if not rows:
                    rows = json.loads(cmd(['condor_history', state['job'], '-limit', '1', '-json', '-attributes', attrs]) or '[]')
                state.update(last_job=rows[0] if rows else None, last_poll=time.time())
                if rows and rows[0]['JobStatus'] == 4:
                    collect(root, state, rows[0])
                    continue
                if rows and rows[0]['JobStatus'] in (3, 5):
                    state['status'] = 'needs_attention'
                save(root, state)
            except Exception as exc:
                state.update(status='needs_attention', error=repr(exc))
                save(root, state)
                return
            if state['status'] == 'needs_attention':
                return
            time.sleep(300)


if __name__ == '__main__':
    main(sys.argv[1])
