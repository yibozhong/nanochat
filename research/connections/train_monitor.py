"""Submit-node monitors for startup, GPU fallback and feedback result collection."""
import fcntl
from functools import partial
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import time

from research.connections.gpu_policy import desired_fallback, feedback_gpus, scale_gpus


def _save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def _job_ad(job, ad_key=None):
    attrs = 'ClusterId,ProcId,JobStatus,ExitCode,HoldReason,RemoteHost,NumJobStarts'
    if ad_key:
        attrs = ('ClusterId,ProcId,JobStatus,EnteredCurrentStatus,ExitCode,HoldReason,'
                 'RemoteHost,NumJobStarts,RequireGPUs,' + ad_key)
    args = ['-json', '-attributes', attrs]
    timeout = 45 if ad_key else 60
    for query in (['condor_q', job], ['condor_history', job, '-limit', '1']):
        output = subprocess.check_output(query + args, text=True, timeout=timeout)
        rows = json.loads(output.strip() or '[]')
        if rows:
            return rows[0]
    return None


def _change_fallback(job, ad, require_gpus):
    wanted = desired_fallback(ad, time.time())
    if wanted is None or wanted == ('A100' in ad.get('RequireGPUs', '')):
        return None
    cluster, proc = map(int, job.split('.'))
    # Match the observed status epoch so a new queue wait cannot inherit an older edit.
    condition = (f'ClusterId=={cluster} && ProcId=={proc} && JobStatus=={ad["JobStatus"]}'
                 f' && EnteredCurrentStatus=={ad["EnteredCurrentStatus"]}')
    if wanted:
        condition += ' && time()-EnteredCurrentStatus>=3600'
    result = subprocess.check_output(['condor_qedit', '-constraint', condition, 'RequireGPUs',
                                      require_gpus(wanted)], text=True, timeout=45).strip()
    return dict(time=time.time(), fallback=wanted, result=result)


def _finite_step(row):
    return 'step' in row and 'loss' in row and row['step'] >= 1 and math.isfinite(row['loss'])


def _log_row(line):
    if line.startswith('{'):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return {}


def marked_steps(lines, run_id, variant):
    if run_id is None:
        raise KeyError('run_id')
    marker = f'FORMAL_TRAINING_BEGIN {variant} {run_id}'
    formal = False
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN '):
            formal = line == marker
        elif formal:
            row = _log_row(line)
            if _finite_step(row):
                yield row


def first_marked_step(lines, run_id, *, variant):
    return next(marked_steps(lines, run_id, variant), None)


def last_marked_step(lines, run_id, *, variant):
    latest = None
    for row in marked_steps(lines, run_id, variant):
        latest = row
    return latest


def first_unmarked_step(lines, run_id=None):
    for line in lines:
        row = _log_row(line)
        if _finite_step(row):
            return row
    return None


def first_step_after_preflight(lines, run_id=None):
    preflight = False
    for line in lines:
        if line.startswith('BEGIN '):
            preflight = True
        row = _log_row(line)
        if row.get('preflight') is True and row.get('complete') is True:
            preflight = False
        elif not preflight and _finite_step(row):
            return row
    return None


def _observed_step(root, job, run_id, read_step):
    log = root / 'logs' / f'train.{job}.out'
    if log.exists():
        return read_step(log.read_text(errors='replace').splitlines(), run_id)
    return None


def watch_start(root, read_step):
    root = Path(root)
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state = json.loads((root/'state.json').read_text())
            if state['status'] in ('training_started', 'needs_attention'):
                return
            state.update(monitor_pid=os.getpid(), last_poll=time.time())
            job = state['job']
            try:
                ad = _job_ad(job)
                state['last_job'] = ad
                value = _observed_step(root, job, state.get('run_id'), read_step)
                if value is not None:
                    state.update(status='training_started', first_observed_step=value, monitor_complete=True)
                elif ad and ad['JobStatus'] in (3, 4, 5):
                    state.update(status='needs_attention', monitor_complete=True)
            except Exception as exc:
                state.update(poll_errors=state.get('poll_errors', 0) + 1, last_error=repr(exc))
                if state['poll_errors'] >= 3:
                    state['status'] = 'needs_attention'
            _save(root/'state.json', state)
            if state['status'] in ('training_started', 'needs_attention'):
                return
            time.sleep(300)


def collect_feedback(root, training):
    archive = root/'results'/f'train-{training["job"]}.tar'
    if not archive.exists():
        return False
    with tarfile.open(archive) as tar:
        for filename in ('summary.json', 'progress.json', 'config.json'):
            try:
                member = tar.getmember('results/train/' + filename)
            except KeyError:
                continue
            assert member.isfile() and member.size < 4*1024*1024
            value = json.load(tar.extractfile(member))
            assert value['source'] == training['source']
            if filename != 'config.json':
                assert value['run_id'] == training['run_id']
            _save(root/filename, value)
    summary = root/'summary.json'
    if not summary.exists():
        return False
    value = json.loads(summary.read_text())
    assert value['complete'] and value['steps'] == training['source']['steps']
    assert value['training_tokens'] == training['source']['training_tokens']
    return True


def _feedback_poll(root, state, training, ad_key, read_step):
    job = training['job']
    ad = _job_ad(job, ad_key)
    if ad is None:
        raise RuntimeError('Job absent from queue and history')
    state['last_job'] = ad
    assert ad[ad_key] == root.name
    transition = _change_fallback(job, ad, feedback_gpus)
    if transition:
        state['transitions'].append(transition)
    latest = _observed_step(root, job, training['run_id'], read_step)
    status = ad['JobStatus']
    if latest:
        state['latest_formal_step'] = latest
        if 'first_formal_step' not in state:
            state['first_formal_step'] = latest
            _save(root/'startup-observation.json', dict(job=job, run_id=training['run_id'],
                  first_formal_step=latest, observed_at=time.time()))
        training.update(status='resume_queued' if status == 1 else 'training_started', latest_observed_step=latest)
    else:
        training['status'] = 'training_submitted' if status == 1 else 'preparing'
    state['status'] = 'monitoring'
    if status in (3, 5) or status == 4 and ad.get('ExitCode') != 0:
        training.update(status='needs_attention', last_job=ad)
        state['status'] = 'needs_attention'
        collect_feedback(root, training)
    elif status == 4:
        if not collect_feedback(root, training):
            raise RuntimeError('Completed job missing final results')
        state['status'] = training['status'] = 'completed'
    _save(root/'state.json', training)


def _scale_poll(root, state, training, ad_key, expected_key, read_step):
    job = training['job']
    ad = _job_ad(job, ad_key)
    state['last_job'] = ad
    if ad:
        assert ad[ad_key] == expected_key
        transition = _change_fallback(job, ad, scale_gpus)
        if transition:
            transition.update(status=ad['JobStatus'], idle_since=ad['EnteredCurrentStatus'])
            state['transitions'].append(transition)
        status = ad['JobStatus']
        if status in (3, 5) or status == 4 and ad.get('ExitCode') != 0:
            state['status'] = 'needs_attention'
            training.update(status='needs_attention', last_job=ad)
            _save(root/'state.json', training)
        elif status == 4:
            state['status'] = 'completed_job'
        else:
            state['status'] = 'monitoring'
    if 'first_formal_step' not in state:
        value = _observed_step(root, job, training['run_id'], read_step)
        if value:
            state['first_formal_step'] = value
            if state.get('status') != 'needs_attention':
                training.update(status='training_started', first_observed_step=value)
                _save(root/'state.json', training)
            _save(root/'startup-observation.json', dict(job=job, run_id=training['run_id'],
                  first_formal_step=value, observed_at=time.time(), last_job=ad))


def _watch_training(root, poll, completion_status, expected_variant=None):
    root = Path(root).resolve()
    with (root/'scheduling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'scheduling.json'
        state = json.loads(path.read_text()) if path.exists() else dict(transitions=[], poll_errors=0)
        while not any((root/name).exists() for name in ('STOPPED', 'STOPPED.json')):
            training = json.loads((root/'state.json').read_text())
            if expected_variant:
                assert training['source']['model_variant'] == expected_variant
            state.update(pid=os.getpid(), job=training['job'], last_poll=time.time())
            try:
                poll(root, state, training)
                state['poll_errors'] = 0
                _save(path, state)
            except Exception as exc:
                state.update(poll_errors=state.get('poll_errors', 0) + 1, last_error=repr(exc))
                _save(path, state)
                if state['poll_errors'] >= 3:
                    state['status'] = 'needs_attention'
                    _save(path, state)
                    return
            if state.get('status') in ('needs_attention', completion_status):
                return
            time.sleep(60)


def watch_feedback(root, ad_key, read_step):
    poll = partial(_feedback_poll, ad_key=ad_key, read_step=read_step)
    _watch_training(root, poll, 'completed')


def watch_scale(root, variant, ad_key, expected_key, read_step):
    poll = partial(_scale_poll, ad_key=ad_key, expected_key=expected_key, read_step=read_step)
    _watch_training(root, poll, 'completed_job', expected_variant=variant)
