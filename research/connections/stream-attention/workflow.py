"""Submit/collect only a bounded independent-attention implementation gate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

REPO = Path(__file__).resolve().parents[3]
KEY = 'mhc-stream-attention-20261103'


def command(args, cwd):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def write(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)+'\n'); tmp.replace(path)


def submit(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists(): print((root/'state.json').read_text()); return
        assert not (root/'submission-intent.json').exists(), 'Recover existing submission intent; do not duplicate'
        jobs = json.loads(command(['condor_q', 'yzhong239', '-constraint', f'StreamAttentionKey == "{KEY}"',
            '-json', '-attributes', 'ClusterId,ProcId'], root) or '[]')
        assert not jobs
        assert not command(['git', 'status', '--porcelain'], REPO), 'Commit reviewed source first'
        commit = command(['git', 'rev-parse', 'HEAD'], REPO)
        for name in ('bundles', 'logs', 'results'): (root/name).mkdir(exist_ok=True)
        bundle = root/'bundles'
        previous = Path('/home/yzhong239/mhc-no-sinkhorn-20261031/bundles')
        wheel = 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(previous/wheel, bundle/wheel)
        command(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', f'--output={bundle/"source.tar"}', commit], REPO)
        old = json.loads((previous/'source.json').read_text())
        hashes = {k:v for k,v in old['hashes'].items() if k in (wheel,
            'mhc-official-runtime-7546767.tar.gz', 'cuda-compat-13-1_590.48.01-1_amd64.deb')}
        hashes['source.tar'] = hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        source = dict(commit=commit, hashes=hashes, formal_training=False,
            scope='Per-stream mHC attention correctness, synthetic optimizer smoke and same-GPU profile')
        write(bundle/'source.json', source)
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/stream-attention/validate.sh', root/'validate.sh')
        inputs = [str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs += ['osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        text = f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = validate.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_disk = 40GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
transfer_input_files = {','.join(inputs)}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/stream-$(Cluster).$(Process).tar"
log = logs/stream.$(Cluster).$(Process).log
output = logs/stream.$(Cluster).$(Process).out
error = logs/stream.$(Cluster).$(Process).err
+JobBatchName = "nanochat-stream-attention-gate"
+StreamAttentionKey = "{KEY}"
queue 1
'''
        (root/'gate.sub').write_text(text)
        command(['condor_submit', '-dry-run', 'dryrun.ad', 'gate.sub'], root)
        write(root/'submission-intent.json', dict(source=source, time=time.time()))
        job = command(['condor_submit', '-terse', 'gate.sub'], root).split()[0]
        state = dict(status='submitted', job=job, source=source, formal_training=False, submitted_at=time.time())
        write(root/'state.json', state); print(json.dumps(state, indent=2))


def monitor(root):
    with (root/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (root/'STOPPED.json').exists():
            state = json.loads((root/'state.json').read_text())
            if state['status'] in ('complete', 'needs_attention', 'stopped_by_user'): return
            try:
                attrs = 'ClusterId,ProcId,JobStatus,NumJobStarts,RemoteHost,ExitCode,HoldReason'
                ads = json.loads(command(['condor_q', state['job'], '-json', '-attributes', attrs], root) or '[]')
                if not ads:
                    ads = json.loads(command(['condor_history', state['job'], '-limit', '1', '-json', '-attributes', attrs], root) or '[]')
                state.update(last_poll=time.time(), last_job=ads[0] if ads else None); write(root/'state.json', state)
                if ads and ads[0]['JobStatus'] in (3, 4, 5):
                    assert ads[0]['JobStatus']==4 and ads[0].get('ExitCode')==0, ads
                    archive = root/'results'/f'stream-{state["job"]}.tar'
                    with tarfile.open(archive) as tf:
                        assert not any(m.name.endswith(('.pt', '.wandb')) for m in tf)
                        result = json.load(tf.extractfile('results/stream-results.json'))
                    assert result['complete'] and result['source']==state['source']
                    assert result['checks']['failures']==result['checks']['errors']==result['checks']['skipped']==0
                    assert len(result['trials'])==6 and result['logging_backend']=='local' and not result['formal_training']
                    write(root/'results.json', result)
                    state.update(status='complete', summary=result['summary'], completed_at=time.time()); write(root/'state.json', state)
                    return  # no automatic training or mutation of concurrently edited repository
            except Exception as exc:
                state.update(status='needs_attention', error=repr(exc)); write(root/'state.json', state); raise
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('submit', 'monitor'))
    parser.add_argument('root', type=Path); args = parser.parse_args()
    globals()[args.action](args.root.resolve())
