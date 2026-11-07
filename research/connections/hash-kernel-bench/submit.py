"""Submit exactly one isolated, credential-free kernel comparison."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path('/home/yzhong239/nanochat-runtime-assets')
REFERENCES = Path('/home/yzhong239/hash-weight-kernel-references-20261107')


def main(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any((root/n).exists() for n in ('state.json', 'submission-intent.json', 'STOPPED.json'))
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
        for name in ('bundles', 'logs', 'results'):
            (root/name).mkdir(exist_ok=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        dependencies = json.loads((REFERENCES/'dependency-manifest.json').read_text())
        assert dependencies[0]['commit'] == '9b680f428484625f4f35dc3617f134187c6bcd4a'
        source = dict(commit=commit, formal_training=False, purpose='Fixed-hash BF16 FFN kernel comparison: DeepGEMM and TE GroupedLinear', dependencies=dependencies)
        (root/'bundles/source.json').write_text(json.dumps(source, indent=2)+'\n')
        (root/'bundles/dependencies.json').write_text(json.dumps(dependencies, indent=2)+'\n')
        subprocess.run(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o', str(root/'bundles/source.tar'), 'HEAD'], cwd=REPO, check=True)
        for name in ('gate-data.tar', 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (root/'bundles'/name).symlink_to(ASSETS/'bundles'/name)
        hashes = json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        for dep in dependencies:
            origin = REFERENCES/dep.get('source_filename', dep['name']+'-'+dep['commit']+'.tar.gz')
            assert hashlib.sha256(origin.read_bytes()).hexdigest() == dep['sha256']
            destination = root/'bundles'/dep.get('archive_name', dep['name']+'.tar.gz')
            shutil.copy2(origin, destination)
            hashes[destination.name] = dep['sha256']
        for name in ('source.json', 'source.tar', 'dependencies.json'):
            hashes[name] = hashlib.sha256((root/'bundles'/name).read_bytes()).hexdigest()
        (root/'bundles/inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/hash-kernel-bench/run.sh', root/'run.sh')
        template = (ASSETS/'gate.sub').read_text().replace('executable = validate.sh', 'executable = run.sh')
        template = template.replace('stream-$(Cluster)', 'bench-$(Cluster)').replace('logs/stream.', 'logs/bench.')
        template = template.replace('nanochat-depth-diagnostic', 'nanochat-hash-kernel-benchmark').replace('DepthDiagnosticKey', 'HashKernelBenchKey').replace('depth-diagnostic-20261103', root.name)
        template = template.replace('22000', '40000').replace('(Capability == 9.0)', '(Capability == 9.0) && regexp("H100|H200",DeviceName)')
        template = template.replace('transfer_input_files = ', 'transfer_input_files = bundles/deepgemm.tar.gz,bundles/cutlass.tar.gz,bundles/fmt.tar.gz,bundles/cuobjdump.tar.xz,bundles/dependencies.json,')
        template = template.replace('queue 1', 'hold = true\nqueue 1')
        (root/'bench.sub').write_text(template)
        subprocess.run(['condor_submit', '-dry-run', str(root/'dry-run.ads'), 'bench.sub'], cwd=root, check=True, capture_output=True)
        (root/'submission-intent.json').write_text(json.dumps(source, indent=2)+'\n')
        job = subprocess.check_output(['condor_submit', '-terse', 'bench.sub'], cwd=root, text=True).split()[0]
        state = dict(status='submitted', job=job, source=source, hashes=hashes, formal_training=False, automatic_training=False)
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        ad = json.loads(subprocess.check_output(['condor_q', job, '-json'], text=True))[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestMemory'] == 16384 and ad['RequestCpus'] == 2
        assert '.netrc' not in ad['TransferInput'] and ad['TransferOutput'] == 'results.tar'
        assert 'H100|H200' in ad['RequireGPUs'] and '40000' in ad['RequireGPUs']
        assert ad['HashKernelBenchKey'] == root.name
        (root/'submission-verified.json').write_text(json.dumps(ad, indent=2)+'\n')
        subprocess.run(['condor_release', job], check=True, capture_output=True)
        print(json.dumps(dict(job=job, source=commit, root=str(root))))


if __name__ == '__main__':
    main(sys.argv[1])
