"""Two newly authorized B4 10B runs, independent of the frozen older jobs.

AP-only preparation/submission/monitoring. No automatic replacement/release of
failed jobs. Native exit85 checkpoints keep the same job and W&B run ID.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time
import traceback
import uuid

from submit import REPO, HOME_ROOT, ASSETS, WHEEL, command, write, stopped, submission_text
from policy import STEPS, TOKENS
from optimized_preflight import PLAN, VERIFIED, MODES, validate_source, validate_ready

ROOT = HOME_ROOT/PLAN


def launcher():
    original = (REPO/'research/connections/hca-train/train.sh').read_text()
    prefix, tail = original.split('# Explicit investigation payloads', 1)
    assert 'hca_root_cause_v1' in tail
    return prefix + '''# Frozen B4 policy, shared allocation for diagnostic gate and formal run.
TASK_GRADIENT=$(python - "$TASK_MODE" "$TASK_RUN_ID" <<'PYCONFIG'
import json,sys
s=json.load(open('source.json'))
assert (s['attention_mode'],s['run_id']) == tuple(sys.argv[1:])
print(s['hca']['dynamic_gradient'])
PYCONFIG
)
if timeout --signal=TERM --kill-after=30 3600 python research/connections/hca-train/optimized_preflight.py; then
    echo "OPTIMIZED_B4_PREFLIGHT_COMPLETE $TASK_MODE $TASK_RUN_ID"
else
    failure=$?
    echo "HCA_PREFLIGHT_FAILED $TASK_MODE $TASK_RUN_ID exit=$failure; debug allocation retained 600 seconds; no automatic retry."
    sleep 600
    exit "$failure"
fi
export WANDB_MODE=online
echo "FORMAL_TRAINING_BEGIN $TASK_MODE $TASK_RUN_ID"
set +e
python -m scripts.connection_train --architecture vanilla --variant baseline --streams 1 \\
    --sequence-length 2048 --attention-kernel original --attention-mode "$TASK_MODE" \\
    --hca-block-size 4 --hca-window 256 --hca-ratio-loss-weight 1 --hca-temperature 1 --no-hca-noise \\
    --hca-backend triton --hca-dynamic-gradient "$TASK_GRADIENT" \\
    --steps 305176 --schedule-steps 305176 --microbatch 16 --seed 42 \\
    --no-activation-checkpoint --no-fused-connection-grad --evaluate-test \\
    --data-dir ../data --output-dir results/train --checkpoint-dir ../checkpoints \\
    --formal-training --wandb-group hca-optimized-b4-10b-20261113 --run-id "$TASK_RUN_ID" \\
    --resume auto --segment-seconds 1800
status=$?
set -e
if [ "$status" -ne 0 ] && [ "$status" -ne 85 ]; then
    echo "HCA_FORMAL_FAILED $TASK_MODE $TASK_RUN_ID exit=$status; debug allocation retained 600 seconds."
    sleep 600
fi
exit "$status"
'''


def description(mode, run_id, data_uri):
    text = submission_text(mode, run_id, data_uri)
    # Reuse the audited transfer/checkpoint/container contract. No legacy
    # cross-run gate: each independently authorized job has its own preflight.
    text = text.replace('bundles/core-evidence.json,', '')
    text = text.replace(',bundles/validated-shared.json', '')
    text = text.replace('request_memory = 32GB', 'request_memory = 48GB')
    text = text.replace('request_disk = 80GB', 'request_disk = 100GB')
    text = text.replace('nanochat-hca-10b', PLAN).replace('hca-10b-20261112', PLAN)
    return text.replace('hold = true', '+HCAFormalTraining = true\n+HCABlockSize = 4\nhold = true')


def initialize():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not stopped(ROOT)
        if (ROOT/'plan.json').exists():
            return json.loads((ROOT/'plan.json').read_text())
        assert not command(['git', 'status', '--porcelain']), 'Commit and push before freezing source'
        commit = command(['git', 'rev-parse', 'HEAD'])
        assert not command(['git', 'diff', '--name-only', VERIFIED, commit, '--',
                            'nanochat', 'tests', 'research/connections/runtime', 'research/connections/toolchain'])
        assert not json.loads(command(['condor_q', 'yzhong239', '-constraint',
            f'HCAPlanKey == "{PLAN}"', '-json', '-attributes', 'ClusterId,ProcId,JobStatus']) or '[]')
        data = json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length'] == 2048
        assert data['manifest']['training_steps'] == STEPS and data['manifest']['training_tokens'] == TOKENS
        assert data['manifest']['tokenizer']['vocab_size'] == 32768
        assert stat.S_IMODE((HOME_ROOT/'.netrc').stat().st_mode) == 0o600
        transport = json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        (ROOT/'bundles').mkdir(exist_ok=True)
        subprocess.run(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o',
                        str(ROOT/'bundles/source.tar'), commit], cwd=REPO, check=True)
        plan = dict(status='prepared', commit=commit, plan=PLAN, authorized_formal_runs=2,
                    created_at=time.time(), steps_per_run=STEPS, tokens_per_run=TOKENS,
                    initial_gpu_policy='Complete H100/H200; no architecture expansion without compatible validation', runs={})
        for mode in MODES:
            root = ROOT/mode
            for name in ('bundles', 'logs', 'results'):
                (root/name).mkdir(parents=True, exist_ok=True)
            run_id = uuid.uuid4().hex[:8]
            source = dict(plan=PLAN, commit=commit, formal_training=True, attention_mode=mode, run_id=run_id,
                steps=STEPS, training_tokens=TOKENS, seed=42, microbatch=16, batch_tokens=32768,
                initialization='seed42_from_zero_no_validation_weights', hca_backend='triton',
                verified_runtime_commit=VERIFIED, prior_validation='research/connections/hca-opt/results.json',
                data_uri=data['data_uri'], data_archive_bytes=data['archive']['bytes'],
                dataset_revision=data['manifest']['dataset_revision'], tokenizer=data['manifest']['tokenizer']['tokenizer'],
                hca=dict(target_ratio=4, window=256, ratio_loss_weight=1, temperature=1, noise=False,
                         dynamic_gradient='confidence' if mode == 'dynamic_hca' else 'dmc'))
            validate_source(source); write(root/'bundles/source.json', source)
            (root/'bundles/source.tar').symlink_to(ROOT/'bundles/source.tar')
            (root/'bundles'/WHEEL).symlink_to(ASSETS/'bundles'/WHEEL)
            (root/'train.sh').write_text(launcher()); (root/'train.sh').chmod(0o755)
            subprocess.run(['bash', '-n', str(root/'train.sh')], check=True)
            (root/'train.sub').write_text(description(mode, run_id, data['data_uri']))
            plan['runs'][mode] = dict(root=str(root), run_id=run_id,
                wandb_url='https://wandb.ai/yibozhong657-none/nanochat/runs/'+run_id)
        write(ROOT/'plan.json', plan)
        return plan


def enqueue(mode):
    plan = json.loads((ROOT/'plan.json').read_text()); info = plan['runs'][mode]; root = Path(info['root'])
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not stopped(ROOT) and not stopped(root)
        if (root/'state.json').exists():
            return json.loads((root/'state.json').read_text())
        assert not (root/'submission-intent.json').exists(), 'Inspect prior submission intent; never duplicate'
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint',
            f'HCAPlanKey == "{PLAN}" && HCATrainMode == "{mode}"', '-json', '-attributes', 'ClusterId,ProcId,JobStatus']) or '[]')
        assert not active
        source = json.loads((root/'bundles/source.json').read_text()); validate_source(source)
        assert source['commit'] == plan['commit'] and source['run_id'] == info['run_id'] and source['attention_mode'] == mode
        subprocess.run(['condor_submit', '-dry-run', str(root/'dry-run.ads'), 'train.sub'], cwd=root, check=True, capture_output=True)
        write(root/'submission-intent.json', dict(source=source, time=time.time()))
        job = command(['condor_submit', '-terse', 'train.sub'], cwd=root).split()[0]
        state = dict(status='held_for_audit', job=job, run_id=info['run_id'], source=source,
                     wandb_url=info['wandb_url'], formal_started=False, microbatch=16, submitted_at=time.time())
        write(root/'state.json', state)
        ad = json.loads(command(['condor_q', job, '-json']))[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert (ad['RequestGPUs'], ad['RequestCpus'], ad['RequestMemory'], ad['RequestDisk']) == (1, 2, 49152, 100*1024**2)
        assert ad['Args'] == mode+' '+info['run_id'] and ad['HCABlockSize'] == 4
        assert ad['SuccessCheckpointExitCode'] == 85 and set(ad['TransferCheckpoint'].split(',')) == {'checkpoints.tar', 'results.tar'}
        assert '.netrc' in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and 'source.tar' in ad['TransferInput']
        assert 'validated-shared' not in ad['TransferInput']
        assert 'H100|H200' in ad['RequireGPUs'] and '78000' in ad['RequireGPUs']
        assert ad['HCAPlanKey'] == PLAN and ad['HCATrainMode'] == mode and ad['HCARunId'] == info['run_id']
        assert ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        write(root/'submission-verified.json', ad)
        subprocess.run(['condor_release', job], check=True, capture_output=True)
        state['status'] = 'queued_for_startup_validation'; write(root/'state.json', state)
        return state


def tick():
    from controller import consume, collect
    if stopped(ROOT):
        return False
    plan = json.loads((ROOT/'plan.json').read_text()); states = {}
    attrs = 'ClusterId,ProcId,JobStatus,HoldReason,RemoteHost,NumJobStarts,ExitCode,ExitBySignal,QDate'
    for mode in MODES:
        root = Path(plan['runs'][mode]['root']); path = root/'state.json'
        if not path.exists():
            continue  # No automatic new submissions from the monitor.
        state = json.loads(path.read_text()); states[mode] = state
        if state['status'] in ('needs_attention', 'complete'):
            continue
        if stopped(root):
            state.update(status='needs_attention', failure='Stopped marker'); write(path, state); continue
        ads = json.loads(command(['condor_q', state['job'], '-json', '-attributes', attrs]) or '[]')
        if not ads:
            ads = json.loads(command(['condor_history', state['job'], '-limit', '1', '-json', '-attributes', attrs]) or '[]')
        assert len(ads) == 1, 'Missing scheduler state; inspect rather than resubmit'
        ad = ads[0]; state['scheduler'] = ad
        starts = ad.get('NumJobStarts')
        if starts is not None and starts != state.get('observed_starts'):
            state.update(observed_starts=starts, stdout_offset=0, inside_formal_log=False)
        log = root/f'logs/train.{state["job"]}.out'
        if not state.get('startup_passed') and log.exists():
            for line in log.read_text(errors='replace').splitlines():
                if line.startswith('OPTIMIZED_B4_STARTUP_PASSED '):
                    report = json.loads(line.split(' ', 1)[1]); validate_ready(report, state['source'])
                    write(root/'validated.json', report)
                    state.update(startup_passed=True, estimated_train_only_hours=report['estimated_train_only_hours'])
        consume(root, state, plan)
        if state['status'] == 'needs_attention':
            pass
        elif ad['JobStatus'] in (3, 5):
            state.update(status='needs_attention', failure=ad.get('HoldReason', 'Removed'))
        elif ad['JobStatus'] == 4:
            assert ad.get('ExitCode') == 0 and not ad.get('ExitBySignal', False)
            collect(root, state)
            c = json.loads((root/'reports/train/config.json').read_text())['model']
            assert c['hca_block_size'] == 4 and c['hca_noise'] is False and c['hca_dynamic_gradient'] == state['source']['hca']['dynamic_gradient']
        elif ad['JobStatus'] == 2:
            state['status'] = 'formal_training' if state['formal_started'] else ('formal_initialization' if state.get('startup_passed') else 'startup_validation')
        elif ad['JobStatus'] == 1:
            state['status'] = 'queued_for_resume' if state['formal_started'] else 'queued_for_startup_validation'
        state['updated_at'] = time.time(); write(path, state)
    plan.update(updated_at=time.time(), states={k:dict(job=v['job'], status=v['status'], formal_started=v['formal_started']) for k,v in states.items()})
    plan['status'] = ('complete' if len(states) == 2 and all(v['status'] == 'complete' for v in states.values())
                      else 'needs_attention' if any(v['status'] == 'needs_attention' for v in states.values()) else 'active')
    write(ROOT/'plan.json', plan)
    return any(v['status'] not in ('needs_attention', 'complete') for v in states.values())


def monitor():
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
                    write(ROOT/'controller-failure.json', dict(error=traceback.format_exc(), time=time.time()))
                    return
            time.sleep(60)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('initialize', 'monitor', 'tick', *MODES)); a = p.parse_args()
    if a.action == 'monitor':
        monitor()
    else:
        print(json.dumps(initialize() if a.action == 'initialize' else tick() if a.action == 'tick' else enqueue(a.action), indent=2))
