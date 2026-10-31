"""Submit the authorized ablation gate and a held, immutable10B formal run."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(sys.argv[1]).resolve()
REPO = Path(__file__).resolve().parents[3]
PRETRAIN = Path('/home/yzhong239/mhc-pretrain-20261030')
KEY = 'mhc-no-sinkhorn-20261031'
DATA = 'osdf:///chtc/staging/y/yzhong239/nanochat-mhc-10b-data-7560536.tar'
GATE_DATA = 'osdf:///chtc/staging/y/yzhong239/nanochat-ebt-20261029/data-7444406.tar'


def command(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def save(state):
    path = ROOT/'state.tmp'
    path.write_text(json.dumps(state, indent=2)+'\n')
    path.replace(ROOT/'state.json')


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text())
            return
        assert not command(['git', 'status', '--porcelain'], REPO)
        jobs = json.loads(command(['condor_q', 'yzhong239', '-constraint', f'NoSinkhornKey == "{KEY}"',
                                   '-json', '-attributes', 'ClusterId,ProcId']) or '[]')
        assert not jobs, 'Recover marked jobs instead of duplicating them'
        assert not (ROOT/'submission-intent.json').exists(), 'Recover the existing submission intent'
        prior = json.loads((PRETRAIN/'state.json').read_text())
        assert prior['data_ready'] and prior['training_steps'] == 305176 and prior['data_uri'] == DATA
        for folder in ('bundles', 'logs', 'results'):
            (ROOT/folder).mkdir(exist_ok=True)
        immutable_paths = [PRETRAIN/'train.sh', PRETRAIN/'train.sub',
            *[(PRETRAIN/'training-bundles'/name) for name in ('source.tar', 'source.json', 'inputs.sha256')]]
        immutable = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable_paths}
        (ROOT/'immutable-reference-inputs.json').write_text(json.dumps(immutable, indent=2)+'\n')
        commit = command(['git', 'rev-parse', 'HEAD'], REPO)
        run_id = uuid.uuid4().hex[:8]
        intent = dict(commit=commit, run_id=run_id, tokens=10000007168, steps=305176, key=KEY)
        (ROOT/'submission-intent.json').write_text(json.dumps(intent, indent=2)+'\n')
        bundle = ROOT/'bundles'
        wheel = 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(PRETRAIN/'training-bundles'/wheel, bundle/wheel)
        command(['git', 'archive', '--format=tar', '--prefix=nanochat-src/',
                 f'--output={bundle/"source.tar"}', commit], REPO)
        hashes = {name: digest for name, digest in prior['training_source']['hashes'].items()
                  if name in (wheel, 'mhc-official-runtime-7546767.tar.gz', 'cuda-compat-13-1_590.48.01-1_amd64.deb')}
        assert len(hashes) == 3
        hashes['source.tar'] = hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        source = dict(commit=commit, reference_training_commit=prior['training_source']['commit'],
            purpose='User-authorized NVIDIA mHC no-Sinkhorn10B ablation', hashes=hashes,
            variant='mhc_no_sinkhorn', steps=305176, training_tokens=10000007168,
            data_uri=DATA, gate_data_uri=GATE_DATA, reference_jobs=prior['training_jobs'])
        for name in ('validate.sh', 'train.sh'):
            shutil.copy2(REPO/'research/connections/no-sinkhorn'/name, ROOT/name)
        source['launcher_sha256'] = hashlib.sha256((ROOT/'train.sh').read_bytes()).hexdigest()
        source['gate_launcher_sha256'] = hashlib.sha256((ROOT/'validate.sh').read_bytes()).hexdigest()
        (bundle/'source.json').write_text(json.dumps(source, indent=2)+'\n')
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in hashes.items()))
        common = '''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
'''+f'+NoSinkhornKey = "{KEY}"\n'
        inputs = ','.join([f'bundles/{wheel}', 'bundles/source.tar', 'bundles/source.json', 'bundles/inputs.sha256',
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb'])
        gate = common+f'''executable = validate.sh
request_disk = 40GB
transfer_input_files = {inputs},{GATE_DATA}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/gate-$(Cluster).$(Process).tar"
log = logs/gate.$(Cluster).$(Process).log
output = logs/gate.$(Cluster).$(Process).out
error = logs/gate.$(Cluster).$(Process).err
+JobBatchName = "nanochat-mhc-no-sinkhorn-gate"
+NoSinkhornRole = "gate"
queue 1
'''
        train = common+f'''executable = train.sh
arguments = mhc_no_sinkhorn {run_id} 305176
environment = "NANOCHAT_DATA_ARCHIVE=nanochat-mhc-10b-data-7560536.tar"
encrypt_input_files = /home/yzhong239/.netrc
hold = True
request_disk = 80GB
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
transfer_input_files = {inputs},{DATA},/home/yzhong239/.netrc
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"
log = logs/train.$(Cluster).$(Process).log
output = logs/train.$(Cluster).$(Process).out
error = logs/train.$(Cluster).$(Process).err
+JobBatchName = "nanochat-mhc-no-sinkhorn-pretrain"
+NoSinkhornRole = "formal"
+MHCVariant = "mhc_no_sinkhorn"
+MHCRunID = "{run_id}"
queue 1
'''
        for name, text in [('gate', gate), ('train', train)]:
            (ROOT/(name+'.sub')).write_text(text)
            command(['condor_submit', '-dry-run', name+'-dryrun.ad', name+'.sub'])
        state = dict(status='submitting', source=source, reviewed_head=commit, run_id=run_id,
            expected_steps=305176, expected_tokens=10000007168, gate_verified=False,
            training_released=False, submitted_at=time.time())
        save(state)
        # Persist each returned ID before the next mutation. If interrupted,
        # do not submit again: recover the marked job and recorded intent.
        for role in ('train', 'gate'):
            output = command(['condor_submit', '-terse', role+'.sub'])
            job = output.split()[0]
            assert len(job.split('.')) == 2 and all(x.isdigit() for x in job.split('.')), output
            state[role+'_job'] = job
            save(state)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in immutable.items())
        state['status'] = 'gate_pending_training_held'
        save(state)
        print(json.dumps({k: state[k] for k in ('status', 'train_job', 'gate_job', 'run_id')}, indent=2))


if __name__ == '__main__':
    main()
