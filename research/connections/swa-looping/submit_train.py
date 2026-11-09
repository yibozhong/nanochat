"""Submit one fresh 10B run of the GPU-validated vanilla swa-looping implementation."""
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
from collect import validate

REPO = Path(__file__).resolve().parents[3]
HOME_ROOT = Path('/home/yzhong239')
ASSETS = HOME_ROOT / 'nanochat-runtime-assets'
KEY = 'swa-looping-10b-20261108'
STEPS = 305176
REPORT_DIR = Path('research/connections/swa-looping')
GATE_ROOT = HOME_ROOT / 'swa-looping-gate-20261108'


def command(args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def execution_files(gate_source):
    paths = ['nanochat', 'scripts/connection_train.py', 'scripts/connection_context_profile.py',
             'research/connections/runtime', 'research/connections/toolchain',
             'research/connections/attnres/device_probe.py', 'tests/test_swa_looping.py',
             'tests/test_connection_training_resume.py', 'scripts/connection_swa_looping_gate.py']
    assert not command(['git','diff','--name-only',gate_source,'HEAD','--',*paths])
    files=command(['git','ls-files','--',*paths]).splitlines()
    return {name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in files}


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
        line = line.replace('logs/stream.', 'logs/train.').replace('nanochat-depth-diagnostic', 'nanochat-swa-looping-10b')
        line = line.replace('DepthDiagnosticKey', 'SWALoopingTrainKey').replace('depth-diagnostic-20261103', KEY)
        lines.append(line)
    template = '\n'.join(lines) + '\n'
    template=template.replace('(Capability == 9.0)', '(Capability == 9.0 && regexp("H100|H200",DeviceName))')
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
        active = json.loads(command(['condor_q', 'yzhong239', '-constraint', f'SWALoopingTrainKey == "{KEY}"',
                                    '-json', '-attributes', 'ClusterId,ProcId']) or '[]')
        assert not active, 'Recover existing marked job instead of duplicating'
        gate_state=json.loads((GATE_ROOT/'state.json').read_text())
        assert gate_state['status']=='validated' and gate_state['published_commit']
        assert not any((GATE_ROOT/name).exists() for name in ('STOPPED','STOPPED.json'))
        path=REPORT_DIR/'validation.json'
        assert (REPO/path).read_bytes()==subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=REPO)
        gate=json.loads((REPO/path).read_text());validate(gate,gate_state)
        assert gate['job']==gate_state['job']
        history=json.loads(command(['condor_history',gate['job'],'-limit','1','-json','-attributes','JobStatus,ExitCode']) or '[]')
        assert len(history)==1 and history[0]['JobStatus']==4 and history[0]['ExitCode']==0
        retained=execution_files(gate['source']['commit']);microbatch=gate['microbatch']
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
        shutil.copy2(REPO / 'research/connections/swa-looping/train.sh', root / 'train.sh')
        hashes = json.loads((ASSETS / 'state.json').read_text())['source']['hashes'].copy()
        hashes.pop('gate-data.tar')
        hashes['source.tar'] = hashlib.sha256((bundle / 'source.tar').read_bytes()).hexdigest()
        data_name = data['data_uri'].rsplit('/', 1)[-1]
        hashes[data_name] = data['archive']['sha256']
        source = dict(commit=head, launcher_commit=head, formal_training=True, gate_job=gate['job'],
            validated_model_commit=gate['source']['commit'], validated_execution_files=retained,
            purpose='Vanilla SSSL three SWA-only attention passes per S layer10B; fresh seed42,8layers D512 original ReLU-squared MLP',
            model_variant='baseline', experiment='sssl_swa_loop3',
            model_options=dict(architecture='vanilla',variant='baseline',streams=1,window_pattern='SSSL',swa_looping=True,swa_loops=3,swa_window=512),
            initialization='seed42_from_zero_no_validation_weights', steps=STEPS, training_tokens=STEPS*32768,
            data_uri=data['data_uri'], wandb_group=KEY, checkpoint_upload_seconds=1800,
            launcher_sha256=hashlib.sha256((root/'train.sh').read_bytes()).hexdigest(),
            passed_families=[[9,0]], microbatch=microbatch,
            family_preflight='Reuse21checks and25-to128 restart on SM90; reject unsupported GPU families',
            scheduling='H100/H200 only: native-window FA3 required; the current frozen implementation does not support A100/SM120 sliding-window training',
            profile_reference={}, validation_step_seconds=gate['mean_step_seconds'],
            preflight_sha256=hashlib.sha256((REPO/'research/connections/swa-looping/preflight.py').read_bytes()).hexdigest())
        (bundle / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
        hashes['source.json'] = hashlib.sha256((bundle / 'source.json').read_bytes()).hexdigest()
        (bundle / 'inputs.sha256').write_text(''.join(v + '  ' + k + '\n' for k, v in hashes.items()))
        run_id = uuid.uuid4().hex[:8]
        inputs = [str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs += [str(HOME_ROOT / '.netrc'), data['data_uri'],
                   'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
                   'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        (root / 'train.sub').write_text(submission_text(run_id, microbatch, inputs))
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
        assert ad['SWALoopingTrainKey'] == KEY and ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        assert 'RTX PRO 6000' not in ad['RequireGPUs']
        keys = ['ClusterId', 'ProcId', 'JobStatus', 'NumJobStarts', 'RequestGPUs', 'RequestCpus',
                'RequestMemory', 'RequestDisk', 'RequireGPUs', 'GPUsMinCapability', 'Rank', 'WantGPULab',
                'SuccessCheckpointExitCode', 'TransferCheckpoint', 'TransferOutput', 'TransferInput',
                'EncryptInputFiles', 'ContainerImage', 'SWALoopingTrainKey']
        (root / 'submission-verified.json').write_text(json.dumps({k: ad.get(k) for k in keys}, indent=2) + '\n')
        command(['condor_release', job], root)
        print(json.dumps({k: state[k] for k in ('status', 'job', 'run_id', 'wandb_url', 'expected_steps', 'expected_tokens')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
