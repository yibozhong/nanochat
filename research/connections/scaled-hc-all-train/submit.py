"""Submit one fresh 10B run of the GPU-validated NVIDIA scaled HC implementation."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[3]
HOME_ROOT = Path('/home/yzhong239')
ASSETS = HOME_ROOT / 'nanochat-runtime-assets'
KEY = 'scaled-hc-all-10b-20261107'
STEPS = 305176
GATE_SOURCE = '9f211a4af09c53c107be79c896815c7bc310f455'
REPORT_COMMIT = 'eafbadd9a81ccda11718b367c63eaef3894627dc'
REPORT_DIR = Path('research/connections/scaled-hc-all')


def command(args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def validate_gate(gate):
    assert gate['complete'] and not gate['formal_training'] and not gate['preflight']
    assert gate['source']['commit'] == GATE_SOURCE
    assert gate['checks'] == dict(tests=52, failures=0, errors=0, skipped=0)
    assert gate['capability'] == [8, 0] and gate['microbatch'] == 16
    restart = gate['restart']
    assert restart['complete'] and restart['variant'] == 'scaled_hc_all' and restart['steps'] == 128
    assert math.isfinite(restart['final_dev_ce']) and restart['final_dev_ce'] < restart['initial_dev_ce']
    profiles = list(gate['profiles'].values())
    assert sorted(p['variant'] for p in profiles) == ['mhc_no_sinkhorn']*2 + ['scaled_hc']*2 + ['scaled_hc_all']*2
    assert len({p['gpu_uuid'] for p in profiles}) == len({p['input_sha256'] for p in profiles}) == 1
    for p in profiles:
        assert p['source'] == gate['source'] and p['complete'] and p['status'] == 'fits'
        assert p['optimizer_updates'] == 60 and p['causality_passed']
        assert p['microbatch'] == 16 and p['batch_tokens'] == 32768 and p['sequence_length'] == 2048
        assert p['actual_attention_backend'] == 'fa2'
        assert len(p['losses']) == 60 and all(math.isfinite(x) for x in p['losses'])
        assert p['peak_reserved_bytes'] < .9 * 22000 * 1024**2
        if p['variant'] == 'scaled_hc_all':
            assert p['parameters'] == 59615669
    micro = gate['micro']
    assert micro['complete'] and micro['shape'] == [16, 2048, 512, 4]
    assert micro['gpu_uuid'] == profiles[0]['gpu_uuid']
    assert set(micro['results']) == {'read_shared','read_scaled','write_mix_only','write_all'}
    assert len(gate['trained_scale_statistics']) == 49
    assert all(v['mean_abs_change'] > 1e-6 for v in gate['trained_scale_statistics'].values())


def execution_files():
    """Assert every retained model/runtime/training file is the tested version."""
    paths = ['nanochat', 'scripts/connection_train.py', 'research/connections/runtime',
             'research/connections/toolchain', 'research/connections/attnres/device_probe.py',
             'tests/test_scaled_hc.py', 'tests/test_scaled_hc_all.py',
             'tests/test_connection_training_resume.py', 'scripts/connection_scaled_hc_all_gate.py']
    assert not command(['git', 'diff', '--name-only', GATE_SOURCE, 'HEAD', '--', *paths])
    # The previous H200 run passed every check except the corrected saved-tensor
    # unpack. Its kernel math is identical; only that regression and a short
    # real training/restart check are repeated on Hopper in the formal worker.
    assert not command(['git','diff','--name-only','ecf4535357c24267346b253e72ab4bd14a59b3b4','HEAD','--',
        'nanochat/scaled_aggregate.py','nanochat/scaled_nvidia_kernels.py','nanochat/scaled_reduction.py'])
    files = command(['git', 'ls-files', '--', *paths]).splitlines()
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in files}


def submission_text(run_id, microbatch, inputs):
    template = (ASSETS / 'gate.sub').read_text()
    template = template.replace('executable = validate.sh',
                                f'executable = train.sh\narguments = {run_id} {microbatch}')
    template = template.replace('request_disk = 40GB', 'request_disk = 80GB')
    template = template.replace('when_to_transfer_output = ON_EXIT',
        'when_to_transfer_output = ON_EXIT\ncheckpoint_exit_code = 85\n'
        'transfer_checkpoint_files = checkpoints.tar,results.tar\n'
        'encrypt_input_files = /home/yzhong239/.netrc')
    lines = []
    for line in template.splitlines():
        if line.startswith('transfer_input_files ='):
            line = 'transfer_input_files = ' + ','.join(inputs)
        if line.startswith('transfer_output_files ='):
            line = 'transfer_output_files = results.tar,checkpoints.tar'
        if line.startswith('transfer_output_remaps ='):
            line = ('transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;'
                    'checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"')
        line = line.replace('logs/stream.', 'logs/train.').replace('nanochat-depth-diagnostic', 'nanochat-scaled-hc-all-10b')
        line = line.replace('DepthDiagnosticKey', 'ScaledHCAllTrainKey').replace('depth-diagnostic-20261103', KEY)
        lines.append(line)
    template = '\n'.join(lines) + '\n'
    template = template.replace('(Capability == 9.0)', '(Capability == 9.0) && regexp("H100|H200",DeviceName)')
    assert 'gpus_minimum_capability = 9.0' in template
    return template.replace('queue 1', 'hold = true\nqueue 1')


def main(root):
    root = Path(root).resolve()
    assert root == HOME_ROOT / KEY
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((root / x).exists() for x in ('STOPPED', 'STOPPED.json'))
        if (root / 'state.json').exists():
            state = json.loads((root / 'state.json').read_text())
            print(json.dumps({k: state[k] for k in ('status', 'job', 'run_id')}, indent=2))
            return
        assert not (root / 'submission-intent.json').exists(), 'Recover prior intent; never duplicate a submission'
        assert not any((root / x).exists() for x in ('checkpoints', 'checkpoints.tar', 'results.tar'))
        assert not (root / 'results').exists() or not any((root / 'results').iterdir())
        assert not command(['git', 'status', '--porcelain'])
        head = command(['git', 'rev-parse', 'HEAD'])
        assert head == command(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint', f'ScaledHCAllTrainKey == "{KEY}"',
                                    '-json', '-attributes', 'ClusterId,ProcId']) or '[]')
        assert not active, 'Recover existing marked job instead of duplicating'
        for name in ('validation.json', 'performance-summary.json', 'gpu-source-audit.json'):
            path = REPORT_DIR / name
            published = subprocess.check_output(['git', 'show', f'{REPORT_COMMIT}:{path}'], cwd=REPO)
            assert (REPO / path).read_bytes() == published, name
        gate = json.loads((REPO / REPORT_DIR / 'validation.json').read_text())
        validate_gate(gate)
        history = json.loads(command(['condor_history', '8025768.0', '-limit', '1', '-json',
                                      '-attributes', 'ClusterId,ProcId,JobStatus,ExitCode']) or '[]')
        assert len(history) == 1 and history[0]['JobStatus'] == 4 and history[0]['ExitCode'] == 0
        retained = execution_files()
        data = json.loads((HOME_ROOT / 'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length'] == 2048
        assert data['manifest']['training_tokens'] == STEPS * 32768
        runtime = json.loads((REPO / 'research/checkpoint-safety/validation.json').read_text())
        assert runtime['complete'] and runtime['actual_eviction']
        assert (HOME_ROOT / '.netrc').stat().st_mode & 0o777 == 0o600
        for name in ('bundles', 'logs', 'results'):
            (root / name).mkdir(exist_ok=True)
        bundle = root / 'bundles'
        wheel = 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        command(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o', str(bundle / 'source.tar'), head])
        (bundle / wheel).symlink_to(ASSETS / 'bundles' / wheel)
        shutil.copy2(REPO / 'research/connections/scaled-hc-all-train/train.sh', root / 'train.sh')
        hashes = json.loads((ASSETS / 'state.json').read_text())['source']['hashes'].copy()
        hashes.pop('gate-data.tar')
        hashes['source.tar'] = hashlib.sha256((bundle / 'source.tar').read_bytes()).hexdigest()
        data_name = data['data_uri'].rsplit('/', 1)[-1]
        hashes[data_name] = data['archive']['sha256']
        source = dict(commit=head, launcher_commit=head, formal_training=True, gate_job='8025768.0',
            validated_model_commit=GATE_SOURCE, validated_execution_files=retained,
            purpose='NVIDIA static read/mix/post/head channel-stream scales10B; fresh seed42 vanilla8layers D512 original ReLU-squared MLP',
            model_variant='scaled_hc_all', scale='Independent FP32 channel-stream scales for read, residual H, post write and final head',
            initialization='seed42_from_zero_no_validation_weights', steps=STEPS, training_tokens=STEPS*32768,
            data_uri=data['data_uri'], wandb_group=KEY, checkpoint_upload_seconds=1800,
            launcher_sha256=hashlib.sha256((root / 'train.sh').read_bytes()).hexdigest(),
            passed_families=[8], microbatch=16, hopper_prior_gate='8025401.0',
            hopper_preflight='one corrected recomputation test plus25-to60 NTP restart; separate weights deleted',
            profile_reference={}, validation_a100_step_ms=json.loads((REPO / REPORT_DIR / 'performance-summary.json').read_text())['full_step_ms'],
            preflight_sha256=hashlib.sha256((REPO/'research/connections/scaled-hc-all-train/preflight.py').read_bytes()).hexdigest())
        (bundle / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
        hashes['source.json'] = hashlib.sha256((bundle / 'source.json').read_bytes()).hexdigest()
        (bundle / 'inputs.sha256').write_text(''.join(v + '  ' + k + '\n' for k, v in hashes.items()))
        run_id = uuid.uuid4().hex[:8]
        inputs = [str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs += [str(HOME_ROOT / '.netrc'), data['data_uri'],
                   'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
                   'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        (root / 'train.sub').write_text(submission_text(run_id, 16, inputs))
        command(['condor_submit', '-dry-run', 'train-dryrun.ad', 'train.sub'], root)
        state = dict(status='submitting', run_id=run_id, source=source, hashes=hashes, formal_training=True,
                     expected_steps=STEPS, expected_tokens=STEPS*32768, submitted_at=time.time())
        (root / 'submission-intent.json').write_text(json.dumps(state, indent=2) + '\n')
        output = command(['condor_submit', '-terse', 'train.sub'], root)
        job = output.split()[0]
        assert len(job.split('.')) == 2 and all(x.isdigit() for x in job.split('.')), output
        state.update(status='training_submitted', job=job,
                     wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        (root / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        rows = json.loads(command(['condor_q', job, '-json'], root))
        assert len(rows) == 1
        ad = rows[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestCpus'] == 2 and ad['RequestMemory'] == 16384
        assert ad['RequestDisk'] == 80*1024**2 and ad['GPUsMinCapability'] == 9.0
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' not in ad['RequireGPUs']
        assert ad['SuccessCheckpointExitCode'] == 85
        assert set(ad['TransferCheckpoint'].split(',')) == {'checkpoints.tar', 'results.tar'}
        assert set(ad['TransferOutput'].split(',')) == {'checkpoints.tar', 'results.tar'}
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert data_name in ad['TransferInput'] and 'gate-data.tar' not in ad['TransferInput']
        assert ad['ScaledHCAllTrainKey'] == KEY and ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        keys = ['ClusterId', 'ProcId', 'JobStatus', 'NumJobStarts', 'RequestGPUs', 'RequestCpus',
                'RequestMemory', 'RequestDisk', 'RequireGPUs', 'GPUsMinCapability', 'Rank', 'WantGPULab',
                'SuccessCheckpointExitCode', 'TransferCheckpoint', 'TransferOutput', 'TransferInput',
                'EncryptInputFiles', 'ContainerImage', 'ScaledHCAllTrainKey']
        (root / 'submission-verified.json').write_text(json.dumps({k: ad.get(k) for k in keys}, indent=2) + '\n')
        command(['condor_release', job], root)
        print(json.dumps({k: state[k] for k in ('status', 'job', 'run_id', 'wandb_url', 'expected_steps', 'expected_tokens')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
