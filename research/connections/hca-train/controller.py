"""Submit Dynamic only after both startup checks pass; monitor without retries."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

from policy import MODES, STEPS, TOKENS, validate_shared
from submit import ROOT, enqueue, write, stopped


def consume(root, state, plan):
    path = root/f'logs/train.{state["job"]}.out'
    if not path.exists():
        return
    with path.open('rb') as f:
        f.seek(state.get('stdout_offset', 0))
        while True:
            offset = f.tell(); line = f.readline()
            if not line or not line.endswith(b'\n'):
                state['stdout_offset'] = offset
                break
            text = line.decode('utf-8', errors='replace').strip()
            if text.startswith('HCA_PREFLIGHT_FAILED '):
                state.update(status='needs_attention', failure=text)
            if text.startswith('FORMAL_TRAINING_BEGIN '):
                state['inside_formal_log'] = text == 'FORMAL_TRAINING_BEGIN '+state['source']['attention_mode']+' '+state['run_id']
            if not text.startswith('{'):
                continue
            try:
                row = json.loads(text)
            except ValueError:
                continue
            if row.get('event') == 'HCA_STARTUP_PASSED':
                mode = state['source']['attention_mode']
                assert row['run_id'] == state['run_id'] and row['attention_mode'] == mode
                shared = row['report']; micro = validate_shared(shared, plan['commit'])
                assert shared['run_id'] == state['run_id'] and shared['attention_mode'] == mode
                state.update(startup_passed=True, microbatch=micro,
                    estimated_train_only_hours=shared['modes'][mode]['estimated_train_only_hours'])
                if mode == 'fixed_hca':
                    write(ROOT/'validated-shared.json', shared)
            if (state.get('inside_formal_log') and isinstance(row.get('step'), int) and row['step'] >= 1
                and isinstance(row.get('loss'), (int, float)) and math.isfinite(row['loss'])):
                state['formal_started'] = True
                state.setdefault('first_formal_step', row)
                state['latest_logged_step'] = row


def collect(root, state):
    archive = root/f'results/train-{state["job"]}.tar'
    checkpoints = root/f'results/train-{state["job"]}-checkpoints.tar'
    assert archive.exists() and checkpoints.exists()
    reports = root/'reports'; reports.mkdir(exist_ok=True)
    wanted = ('results/train/summary.json', 'results/train/config.json', 'results/train/progress.json',
              'results/source.json', 'results/hca/validation.json', 'results/hca-startup/passed.json',
              'results/hca-startup/own-allocation.json', 'results/environment.txt')
    values = {}
    with tarfile.open(archive) as tf:
        for name in wanted:
            try:
                member = tf.getmember(name)
            except KeyError:
                continue
            assert member.isfile() and member.size < 32*1024**2
            raw = tf.extractfile(member).read()
            target = reports/name.removeprefix('results/'); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            if name.endswith('.json'):
                values[name] = json.loads(raw)
    summary = values['results/train/summary.json']
    config = values['results/train/config.json']
    assert summary['complete'] and summary['steps'] == STEPS and summary['training_tokens'] == TOKENS
    assert summary['run_id'] == state['run_id'] and summary['source'] == state['source']
    assert config['model']['attention_mode'] == state['source']['attention_mode']
    assert config['protocol']['microbatch'] == state['microbatch'] and config['protocol']['seed'] == 42
    assert summary['test_evaluated'] and all(math.isfinite(summary[k]) for k in ('final_dev_ce', 'best_dev_ce', 'final_test_ce', 'best_selected_test_ce'))
    with tarfile.open(checkpoints) as tf:
        for name, info in summary['checkpoint_files'].items():
            assert name in ('latest.pt', 'best.pt')
            assert tf.getmember('checkpoints/'+name).size == info['bytes']
    state.update(status='complete', summary=summary, completed_at=time.time(), reports=str(reports))


def tick():
    plan = json.loads((ROOT/'plan.json').read_text())
    if stopped(ROOT) or plan.get('automatic_training_blocked', False):
        return False
    states = {}
    for mode in MODES:
        root = Path(plan['runs'][mode]['root']); path = root/'state.json'
        if not path.exists():
            continue
        state = json.loads(path.read_text()); states[mode] = state
        if state['status'] in ('complete', 'needs_attention'):
            continue
        if stopped(root):
            state.update(status='needs_attention', failure='Stopped marker; no new actions')
            write(path, state); continue
        consume(root, state, plan)
        ads = json.loads(subprocess.check_output(['condor_q', state['job'], '-json', '-attributes',
            'ClusterId,ProcId,JobStatus,HoldReason,RemoteHost,JobCurrentStartDate,QDate,NumJobStarts'], text=True) or '[]')
        if not ads:
            ads = json.loads(subprocess.check_output(['condor_history', state['job'], '-limit', '1', '-json',
                '-attributes', 'ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,HoldReason'], text=True) or '[]')
        assert len(ads) == 1, 'Missing/ambiguous scheduler state; inspect instead of resubmitting'
        ad = ads[0]; state['scheduler'] = ad
        if state['status'] == 'needs_attention':
            pass
        elif ad['JobStatus'] in (3, 5, 6):
            # Output transfer (6) is normal; do not classify it as failure.
            if ad['JobStatus'] != 6:
                state.update(status='needs_attention', failure=ad.get('HoldReason', 'Removed job'))
        elif ad['JobStatus'] == 4:
            assert ad.get('ExitCode') == 0 and not ad.get('ExitBySignal', False), ad
            collect(root, state)
        elif ad['JobStatus'] == 2:
            state['status'] = 'formal_training' if state['formal_started'] else ('formal_initialization' if state.get('startup_passed') else 'startup_validation')
        elif ad['JobStatus'] == 1:
            state['status'] = 'queued_for_resume' if state['formal_started'] else 'queued_for_startup_validation'
            if time.time() - ad.get('QDate', time.time()) >= 3600:
                state['gpu_policy_note'] = 'Hopper-only: other architectures are not validated; no resource-only expansion'
        state['updated_at'] = time.time(); write(path, state)
    failed = any(v['status'] == 'needs_attention' for v in states.values())
    if not failed and (ROOT/'validated-shared.json').exists() and 'dynamic_hca' not in states:
        states['dynamic_hca'] = enqueue('dynamic_hca')
        print('DYNAMIC_HCA_SUBMITTED '+states['dynamic_hca']['job'], flush=True)
    complete = len(states) == 2 and all(v['status'] == 'complete' for v in states.values())
    plan.update(status='complete' if complete else 'needs_attention' if failed else 'active',
                updated_at=time.time(), states={k: dict(status=v['status'], job=v['job'], formal_started=v['formal_started']) for k,v in states.items()})
    write(ROOT/'plan.json', plan)
    # A failed job is never released/replaced; keep collecting an already
    # submitted independent partner until that job reaches a terminal state.
    return not complete and any(v['status'] not in ('needs_attention', 'complete') for v in states.values())


def main():
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
        errors = 0
        while True:
            try:
                if not tick():
                    return
                errors = 0
            except Exception:
                errors += 1; traceback.print_exc()
                if errors >= 3:
                    write(ROOT/'controller-failure.json', dict(status='needs_attention', time=time.time(),
                        error=traceback.format_exc(), note='No duplicate submission or automatic release attempted'))
                    return
            time.sleep(60)


if __name__ == '__main__':
    main()
