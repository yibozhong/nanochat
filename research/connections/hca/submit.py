"""Single bounded GPU gate, held audit then release. Never submits training."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path('/home/yzhong239/nanochat-runtime-assets')


def main(root):
    root = Path(root).resolve()
    if root.parent != Path('/home/yzhong239') or not root.name.startswith('hca-gate-'):
        raise ValueError('Use a dedicated home/hca-gate-* directory')
    root.mkdir(exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('state.json', 'STOPPED.json', 'submission-intent.json'):
            assert not (root/name).exists(), 'Inspect existing job/state instead of duplicating'
        active = json.loads(subprocess.check_output(['condor_q', 'yzhong239', '-constraint',
            'HCAGateKey =!= undefined', '-json', '-attributes', 'ClusterId,ProcId,JobStatus'], text=True) or '[]')
        assert not active, active
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
        for name in ('bundles', 'logs', 'results'):
            (root/name).mkdir(exist_ok=True)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        source = dict(commit=commit, formal_training=False,
                      purpose='Fixed HCA and free Dynamic HCA with global DMC-style compression loss; target R16, no hard quota, no formal run',
                      budget=dict(validation_timeout_seconds=2400, max_job_seconds=3600),
                      references='research/connections/hca/README.md')
        (root/'bundles/source.json').write_text(json.dumps(source, indent=2)+'\n')
        subprocess.run(['git', 'archive', '--format=tar', '--prefix=nanochat-src/', '-o',
                        str(root/'bundles/source.tar'), 'HEAD'], cwd=REPO, check=True)
        for name in ('gate-data.tar', 'transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (root/'bundles'/name).symlink_to(ASSETS/'bundles'/name)
        shutil.copy2(REPO/'research/connections/hca/run.sh', root/'run.sh')
        (root/'run.sh').chmod(0o755)
        template = (ASSETS/'gate.sub').read_text()
        template = template.replace('executable = validate.sh', 'executable = run.sh')
        template = template.replace('bundles/inputs.sha256,', '')
        template = template.replace('stream-', 'hca-').replace('logs/stream.', 'logs/hca.')
        template = template.replace('nanochat-depth-diagnostic', 'nanochat-hca-validation')
        template = template.replace('DepthDiagnosticKey', 'HCAGateKey').replace('depth-diagnostic-20261103', root.name)
        template = template.replace('(Capability == 9.0)', '(Capability == 9.0) && regexp("H100|H200",DeviceName)')
        template = template.replace('queue 1', '''hold = true
on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)
periodic_hold = (JobStatus == 2) && (CurrentTime - JobCurrentStartDate > 3600)
queue 1''')
        (root/'gate.sub').write_text(template)
        subprocess.run(['condor_submit', '-dry-run', str(root/'dry-run.ads'), 'gate.sub'], cwd=root, check=True, capture_output=True)
        (root/'submission-intent.json').write_text(json.dumps(source, indent=2)+'\n')
        job = subprocess.check_output(['condor_submit', '-terse', 'gate.sub'], cwd=root, text=True).split()[0]
        state = dict(status='held_for_audit', job=job, source=source, formal_training=False)
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        ad = json.loads(subprocess.check_output(['condor_q', job, '-json'], text=True))[0]
        assert ad['JobStatus'] == 5 and ad.get('NumJobStarts', 0) == 0
        assert ad['RequestGPUs'] == 1 and ad['RequestCpus'] == 2 and ad['RequestMemory'] == 16384 and ad['RequestDisk'] == 40*1024**2
        assert '.netrc' not in ad['TransferInput'] and 'inputs.sha256' not in ad['TransferInput']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'], 'Preserve automatically added container URI'
        assert ad['TransferOutput'] == 'results.tar' and ad['HCAGateKey'] == root.name
        assert 'H100|H200' in ad['RequireGPUs'] and ad['WantGPULab'] is True and 'backfill' in ad['Rank']
        (root/'submission-verified.json').write_text(json.dumps(ad, indent=2)+'\n')
        subprocess.run(['condor_release', job], check=True, capture_output=True)
        state['status'] = 'released_for_validation'
        (root/'state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
