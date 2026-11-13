"""Exactly two authorized runs; frozen sources, held audit, no retry on failure."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import stat
import subprocess
import time
import uuid

from policy import MODES, STEPS, TOKENS, CORE_SOURCE, CORE_JOB, TEST_COUNTS, validate_shared

REPO = Path(__file__).resolve().parents[3]
HOME_ROOT = Path('/home/yzhong239')
ROOT = HOME_ROOT/'hca-10b-20261112'
ASSETS = HOME_ROOT/'nanochat-runtime-assets'
WHEEL = 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'


def command(args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def stopped(root):
    return any((root/name).exists() for name in ('STOPPED', 'STOPPED.json'))


def submission_text(mode, run_id, data_uri):
    inputs = ['bundles/source.tar', 'bundles/source.json', 'bundles/core-evidence.json',
              'bundles/'+WHEEL, str(HOME_ROOT/'.netrc'), data_uri,
              'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
              'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
    if mode == 'dynamic_hca':
        inputs.append('bundles/validated-shared.json')
    return f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = train.sh
arguments = {mode} {run_id}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 32GB
request_disk = 80GB
request_gpus = 1
gpus_minimum_memory = 78000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && regexp("H100|H200",DeviceName) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 78000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "long"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
encrypt_input_files = /home/yzhong239/.netrc
transfer_input_files = {','.join(inputs)}
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+JobBatchName = "nanochat-hca-10b"
+HCAPlanKey = "{ROOT.name}"
+HCATrainMode = "{mode}"
+HCARunId = "{run_id}"
hold = true
on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))
queue 1
'''


def initialize():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not stopped(ROOT)
        if (ROOT/'plan.json').exists():
            return json.loads((ROOT/'plan.json').read_text())
        assert not command(['git', 'status', '--porcelain']), 'Commit reviewed source before freezing the plan'
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint',
            'HCAPlanKey =!= undefined', '-json', '-attributes', 'ClusterId,ProcId,JobStatus']) or '[]')
        assert not active, active
        commit = command(['git', 'rev-parse', 'HEAD'])
        # Normal version diff only. No content hash scans or remote HEAD audit.
        assert not command(['git', 'diff', '--name-only', CORE_SOURCE, commit, '--',
                            'nanochat', 'tests/test_hca.py', 'research/connections/runtime', 'research/connections/toolchain'])
        core_root = HOME_ROOT/'hca-gate-20261112-r2'
        counts = json.loads((core_root/'reports/test-counts.json').read_text())
        prior = json.loads((core_root/'reports/source.json').read_text())
        assert counts == TEST_COUNTS and prior['commit'] == CORE_SOURCE
        prior_log = (core_root/f'logs/hca.{CORE_JOB}.log').read_text()
        assert 'DeviceName = "NVIDIA H200"' in prior_log and 'Capability = 9.0' in prior_log
        transport = json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        data = json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length'] == 2048
        assert data['manifest']['training_steps'] == STEPS and data['manifest']['training_tokens'] == TOKENS
        assert data['manifest']['tokenizer']['vocab_size'] == 32768
        assert stat.S_IMODE((HOME_ROOT/'.netrc').stat().st_mode) == 0o600
        evidence = dict(source_commit=CORE_SOURCE, job=CORE_JOB, tests=counts, capability=[9, 0],
                        model_code_unchanged=True, test_code_unchanged=True, validated_bundle_commit=commit,
                        scope='18 core tests passed; prior overall gate failed in baseline masked FA2 cache; remaining comparison reruns at startup')
        (ROOT/'bundles').mkdir(exist_ok=True)
        archive = ROOT/'bundles/source.tar'
        subprocess.run(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o', str(archive), commit], cwd=REPO, check=True)
        write(ROOT/'bundles/core-evidence.json', evidence)
        plan = dict(status='prepared', commit=commit, created_at=time.time(), authorized_formal_runs=2,
                    steps_per_run=STEPS, tokens_per_run=TOKENS, data_uri=data['data_uri'],
                    initial_gpu_policy='Complete H100/H200 only; other backends not yet validated', runs={})
        for mode in MODES:
            root = ROOT/mode
            for name in ('bundles', 'logs', 'results'):
                (root/name).mkdir(parents=True, exist_ok=True)
            run_id = uuid.uuid4().hex[:8]
            source = dict(commit=commit, formal_training=True, attention_mode=mode, run_id=run_id,
                          steps=STEPS, training_tokens=TOKENS, seed=42, batch_tokens=32768,
                          initialization='seed42_from_zero_no_validation_weights',
                          data_uri=data['data_uri'], data_archive_bytes=data['archive']['bytes'],
                          dataset_revision=data['manifest']['dataset_revision'],
                          tokenizer=data['manifest']['tokenizer']['tokenizer'],
                          hca=dict(target_ratio=16, window=256, ratio_loss_weight=1, temperature=1, noise=True),
                          microbatch_policy='largest common of 16/8/4/2/1 passing full-shape checks within 90 percent of 78000 MiB',
                          core_validation=dict(source=CORE_SOURCE, job=CORE_JOB, tests=counts),
                          references='research/connections/hca-train/README.md')
            write(root/'bundles/source.json', source)
            for name in ('source.tar', 'core-evidence.json'):
                (root/'bundles'/name).symlink_to(ROOT/'bundles'/name)
            (root/'bundles'/WHEEL).symlink_to(ASSETS/'bundles'/WHEEL)
            shutil.copy2(REPO/'research/connections/hca-train/train.sh', root/'train.sh')
            (root/'train.sh').chmod(0o755)
            (root/'train.sub').write_text(submission_text(mode, run_id, data['data_uri']))
            plan['runs'][mode] = dict(root=str(root), run_id=run_id,
                wandb_url='https://wandb.ai/yibozhong657-none/nanochat/runs/'+run_id)
        write(ROOT/'plan.json', plan)
        return plan


def enqueue(mode):
    plan = json.loads((ROOT/'plan.json').read_text())
    info = plan['runs'][mode]; root = Path(info['root'])
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not stopped(ROOT) and not stopped(root)
        assert not plan.get('automatic_training_blocked', False), 'Training plan blocked for repair'
        if (root/'state.json').exists():
            return json.loads((root/'state.json').read_text())
        assert not (root/'submission-intent.json').exists(), 'Recover prior submission intent; never duplicate'
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint',
            f'HCAPlanKey == "{ROOT.name}" && HCATrainMode == "{mode}"', '-json', '-attributes', 'ClusterId,ProcId,JobStatus']) or '[]')
        assert not active, active
        source = json.loads((root/'bundles/source.json').read_text())
        assert source['commit'] == plan['commit'] and source['attention_mode'] == mode and source['run_id'] == info['run_id']
        if mode == 'dynamic_hca':
            shared = json.loads((ROOT/'validated-shared.json').read_text())
            validate_shared(shared, plan['commit'])
            assert shared['attention_mode'] == 'fixed_hca' and shared['run_id'] == plan['runs']['fixed_hca']['run_id']
            write(root/'bundles/validated-shared.json', shared)
        subprocess.run(['condor_submit', '-dry-run', str(root/'dry-run.ads'), 'train.sub'], cwd=root, check=True, capture_output=True)
        write(root/'submission-intent.json', dict(source=source, time=time.time()))
        job = command(['condor_submit', '-terse', 'train.sub'], cwd=root).split()[0]
        state = dict(status='held_for_audit', job=job, run_id=info['run_id'], source=source,
                     wandb_url=info['wandb_url'], formal_started=False, submitted_at=time.time())
        write(root/'state.json', state)
        ad = json.loads(command(['condor_q', job, '-json']))[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestCpus'] == 2
        assert ad['RequestMemory'] == 32768 and ad['RequestDisk'] == 80*1024**2
        assert ad['SuccessCheckpointExitCode'] == 85 and set(ad['TransferCheckpoint'].split(',')) == {'checkpoints.tar', 'results.tar'}
        assert '.netrc' in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'], 'Preserve the automatic container input URI'
        assert 'H100|H200' in ad['RequireGPUs'] and '78000' in ad['RequireGPUs']
        assert ad['HCATrainMode'] == mode and ad['HCARunId'] == info['run_id']
        assert ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        write(root/'submission-verified.json', ad)
        subprocess.run(['condor_release', job], check=True, capture_output=True)
        state['status'] = 'queued_for_startup_validation'; write(root/'state.json', state)
        return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('initialize', *MODES)); a = p.parse_args()
    print(json.dumps(initialize() if a.action == 'initialize' else enqueue(a.action), indent=2))
