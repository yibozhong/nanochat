"""One frozen-checkpoint diagnostic, held/audited before release; no training."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path('/home/yzhong239/nanochat-runtime-assets')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def main(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((root/n).exists() for n in ('state.json', 'submission-intent.json', 'STOPPED.json'))
        def command(args): return subprocess.check_output(args, cwd=REPO, text=True).strip()
        assert not command(['git', 'status', '--porcelain'])
        for n in ('bundles', 'logs', 'results'): (root/n).mkdir(exist_ok=True)
        retained = {}
        for folder in ('scaled-hc-all-10b-20261107', 'dynamic-hc-all-10b-20261108'):
            for name in ('train.sh', 'bundles/source.tar', 'bundles/source.json', 'bundles/inputs.sha256'):
                p = Path('/home/yzhong239')/folder/name; st = p.stat()
                retained[str(p)] = dict(sha256=digest(p), size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
        (root/'retained-training.json').write_text(json.dumps(retained, indent=2)+'\n')
        source = dict(commit=command(['git', 'rev-parse', 'HEAD']), formal_training=False,
                      purpose='Frozen matched-10B no-Sinkhorn shared HC versus static channel-scaled HC; geometry and channel interventions')
        (root/'bundles/source.json').write_text(json.dumps(source, indent=2)+'\n')
        subprocess.run(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o', str(root/'bundles/source.tar'), 'HEAD'], cwd=REPO, check=True)
        inputs = {}
        for name, folder, stem in [('shared', 'mhc-no-sinkhorn-20261031', 'train-7581136.0'),
                                   ('scaled', 'scaled-hc-10b-20261107', 'train-8015499.0')]:
            parent = Path('/home/yzhong239')/folder/'results'
            with tarfile.open(parent/(stem+'.tar')) as tf:
                summary = json.load(tf.extractfile('results/train/summary.json'))
            assert summary['complete'] and summary['steps'] == 305176
            archive = stem+'-checkpoints.tar'
            inputs[name] = dict(checkpoint='../checkpoints/'+name+'.pt', archive=archive,
                variant=summary['variant'], run_id=summary['run_id'],
                checkpoint_sha256=summary['checkpoint_files']['latest.pt']['sha256'],
                historical_dev_ce=summary['final_dev_ce'])
            (root/'bundles'/archive).symlink_to(parent/archive)
        (root/'bundles/diagnostic-inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
        for name in ('gate-data.tar', 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (root/'bundles'/name).symlink_to(ASSETS/'bundles'/name)
        hashes = json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        for name in ('source.tar', 'source.json', 'diagnostic-inputs.json'):
            hashes[name] = digest(root/'bundles'/name)
        (root/'bundles/inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/hc-channel-diagnostic/run.sh', root/'run.sh')
        (root/'run.sh').chmod(0o755)
        template = (ASSETS/'gate.sub').read_text().replace('executable = validate.sh', 'executable = run.sh')
        template = template.replace('stream-$(Cluster)', 'diagnostic-$(Cluster)').replace('logs/stream.', 'logs/diagnostic.')
        template = template.replace('nanochat-depth-diagnostic', 'nanochat-hc-channel-diagnostic').replace('DepthDiagnosticKey', 'HCChannelDiagnosticKey').replace('depth-diagnostic-20261103', root.name)
        template = template.replace('gpus_minimum_capability = 9.0', 'gpus_minimum_capability = 8.0')
        template = template.replace('(Capability == 9.0)', '((Capability == 9.0 && regexp("H100|H200",DeviceName)) || (Capability == 8.0 && regexp("A100",DeviceName)))')
        template = template.replace('!regexp("MIG",DeviceName)', '!regexp("MIG",DeviceName) && (DeviceUuid =!= "7067f38b-b4c2-5568-ec82-f698473ddd07")')
        transfer = [str(p.relative_to(root)) for p in sorted((root/'bundles').iterdir())]
        transfer += ['osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
                     'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        template = '\n'.join('transfer_input_files = '+','.join(transfer) if l.startswith('transfer_input_files =') else l for l in template.splitlines())+'\n'
        template = template.replace('queue 1', 'hold = true\nqueue 1')
        (root/'diagnostic.sub').write_text(template)
        subprocess.run(['condor_submit', '-dry-run', str(root/'dry-run.ads'), 'diagnostic.sub'], cwd=root, check=True, capture_output=True)
        (root/'submission-intent.json').write_text(json.dumps(source, indent=2)+'\n')
        job = subprocess.check_output(['condor_submit', '-terse', 'diagnostic.sub'], cwd=root, text=True).split()[0]
        state = dict(status='submitted', job=job, source=source, hashes=hashes,
                     formal_training=False, automatic_training=False)
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        ad = json.loads(subprocess.check_output(['condor_q', job, '-json'], text=True))[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestCpus'] == 2 and ad['RequestMemory'] == 16384
        assert ad['RequestDisk'] == 40*1024**2 and ad['GPUsMinCapability'] == 8.0
        assert ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        assert ad['TransferOutput'] == 'results.tar' and '.netrc' not in ad['TransferInput']
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' in ad['RequireGPUs']
        assert ad['HCChannelDiagnosticKey'] == root.name
        (root/'submission-verified.json').write_text(json.dumps(ad, indent=2)+'\n')
        subprocess.run(['condor_release', job], check=True, capture_output=True)
        print(json.dumps(state, indent=2))


if __name__ == '__main__': main(sys.argv[1])
