"""Restore the original 2K cache. CPU data preparation only; never train or log W&B."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

REPO = Path(__file__).resolve().parents[3]
ASSETS = Path('/home/yzhong239/datasets/fineweb-edu-10bt/prepare-assets')
KEY = 'nanochat-data-2k-20261102'
CURRENT = Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json')


def command(args, root):
    return subprocess.check_output(args, cwd=root, text=True, timeout=60).strip()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(root, packing='native', profile_report=None):
    with (root / 'state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'STOPPED.json').exists()
        assert not (root / 'state.json').exists(), 'Existing workflow must not be resubmitted'
        key = KEY if packing == 'native' else root.name
        threads = 8 if packing == 'indexed8' else 4
        profile = None
        if packing != 'native':
            assert profile_report is not None, 'Validate performance and exact outputs before switching'
            profile = json.loads(profile_report.read_text())
            assert profile['complete'] and profile['byte_and_source_state_parity']
            assert profile['speedup_vs_native'][packing] > 1.1
            assert len(profile['trials']) == 8
            tested_hash = profile['input_sha256']['fast_pack.py']
            assert sha(REPO / 'research/connections/data2k/fast_pack.py') == tested_hash
        ads = json.loads(command(['condor_q', '-constraint', f'NanochatDataKey == "{key}"', '-json'], root) or '[]')
        assert not ads, ads
        assert not (root / 'submission-intent.json').exists(), 'Inspect previous submission before retrying'
        for name in ('bundles', 'logs', 'results'):
            (root / name).mkdir(exist_ok=True)
        bundle = root / 'bundles'
        for name in ('nanochat-145da40.tar', 'tokenizer.tar', 'dataset-manifest.json'):
            shutil.copy2(ASSETS / name, bundle / name)
        for name in ('prepare_data.py', 'prepare_data.sh'):
            shutil.copy2(REPO / 'research/connections/pretrain' / name, bundle / name)
        if packing != 'native':
            (bundle / 'prepare_data.py').rename(bundle / 'prepare_native.py')
            shutil.copy2(REPO / 'research/connections/data2k/prepare_indexed.py', bundle / 'prepare_data.py')
            shutil.copy2(REPO / 'research/connections/data2k/fast_pack.py', bundle / 'fast_pack.py')
        for name in ('run.sh', 'finish_data.py'):
            shutil.copy2(REPO / 'research/connections/data2k' / name, bundle / name)
        shutil.copy2('/home/yzhong239/mhc-pretrain-20261030/data.json', bundle / 'expected-data.json')
        hashes = {p.name: sha(p) for p in sorted(bundle.iterdir())}
        (bundle / 'prepare-inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k, v in hashes.items()))
        raw = json.loads((bundle / 'dataset-manifest.json').read_text())
        inputs = [str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs += [item['osdf_uri'] for item in raw['files']]
        preferred_host = None if profile is None else profile['trials'][0]['hostname']
        rank = '0' if preferred_host is None else f'ifThenElse(TARGET.Machine == {json.dumps(preferred_host)}, 1, 0)'
        text = f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = bundles/run.sh
arguments = {threads}
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
rank = {rank}
request_cpus = {threads}
request_memory = 16GB
request_disk = 90GB
transfer_input_files = {','.join(inputs)}
transfer_output_files = results.tar,data.tar,gate-data.tar
transfer_output_remaps = "results.tar=results/data-$(Cluster).$(Process).tar;data.tar=osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-$(Cluster).tar;gate-data.tar=results/gate-data-$(Cluster).$(Process).tar"
log = logs/data.$(Cluster).$(Process).log
output = logs/data.$(Cluster).$(Process).out
error = logs/data.$(Cluster).$(Process).err
+JobBatchName = "nanochat-fineweb-2k-cache"
+NanochatDataKey = "{key}"
queue 1
'''
        (root / 'prepare.sub').write_text(text)
        command(['condor_submit', '-dry-run', 'dryrun.ad', 'prepare.sub'], root)
        source = command(['git', 'rev-parse', 'HEAD'], REPO)
        assert not command(['git', 'status', '--porcelain'], REPO), 'Commit reviewed code before submitting'
        write(root / 'submission-intent.json', dict(source_commit=source, input_sha256=hashes, created_at=time.time()))
        output = command(['condor_submit', '-terse', 'prepare.sub'], root)
        cluster = output.split('.')[0]
        assert cluster.isdigit(), output
        state = dict(status='submitted', job=cluster + '.0', source_commit=source,
                     input_sha256=hashes, sequence_length=2048, training_tokens=10000007168,
                     data_uri=f'osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-{cluster}.tar',
                     formal_training=False, automatic_training=False, submitted_at=time.time(),
                     packing_backend=packing, tokenizer_threads=threads,
                     preferred_host=preferred_host,
                     profile_reference=None if profile is None else str(profile_report.resolve()))
        write(root / 'state.json', state)
        attrs = 'ClusterId,ProcId,JobStatus,NumJobStarts,TransferInput,TransferOutputRemaps,RequestCpus,RequestMemory,RequestDisk,RequestGpus,NanochatDataKey'
        ads = json.loads(command(['condor_q', state['job'], '-json', '-attributes', attrs], root) or '[]')
        assert len(ads) == 1 and ads[0]['NanochatDataKey'] == key
        assert ads[0].get('RequestGpus', 0) == 0
        assert '.netrc' not in ads[0]['TransferInput'] and 'nanochat-uv-v4.sif' in ads[0]['TransferInput']
        write(root / 'submission-verified.json', ads[0])
        print(json.dumps(state), flush=True)


def collect(root, state):
    with tarfile.open(root / 'results' / f'data-{state["job"]}.tar') as archive:
        data = json.load(archive.extractfile('results/data.json'))
        info = json.load(archive.extractfile('results/data-archive.json'))
        check = json.load(archive.extractfile('results/restoration.json'))
        hashes = {name: digest for digest, name in
                  (line.split() for line in archive.extractfile('results/prepare-inputs.sha256').read().decode().splitlines())}
    assert hashes == state['input_sha256']
    assert check['complete'] and check['original_split_hashes_matched'] and check['logging_backend'] == 'local'
    assert data['sequence_length'] == 2048 and data['training_steps'] == 305176
    assert data['training_tokens'] == state['training_tokens'] == 10000007168
    expected = json.loads((root / 'bundles/expected-data.json').read_text())
    for split in ('train', 'dev', 'test'):
        for key in ('sha256', 'bytes', 'rows', 'tokens'):
            assert data['files'][split][key] == expected['files'][split][key]
    staging = Path('/staging/y/yzhong239') / state['data_uri'].rsplit('/', 1)[-1]
    assert staging.stat().st_size == info['bytes']
    # Seek through headers and read the manifest, without hashing 20GB on the login node.
    with tarfile.open(staging, mode='r:') as archive:
        assert json.load(archive.extractfile('data/manifest.json')) == data
        for split in ('train', 'dev', 'test'):
            assert archive.getmember('data/' + split + '.bin').size == data['files'][split]['bytes']
    gate = root / 'results' / f'gate-data-{state["job"]}.tar'
    assert gate.stat().st_size == check['gate_archive']['bytes'] and sha(gate) == check['gate_archive']['sha256']
    with tarfile.open(gate) as archive:
        assert json.load(archive.extractfile('data/manifest.json')) == check['gate_manifest']
        for split, record in check['gate_manifest']['files'].items():
            assert hashlib.sha256(archive.extractfile('data/' + split + '.bin').read()).hexdigest() == record['sha256']
    ready = dict(ready=True, sequence_length=2048, data_uri=state['data_uri'], archive=info,
                 manifest=data, gate_archive=dict(path=str(gate), **check['gate_archive']),
                 job=state['job'], source_commit=state['source_commit'], split_hashes_match_previous_2k=True,
                 verification='Full split/archive hashes on CPU worker; staging size, tar members and manifest checked after transfer; small prefix fully rehashed on receipt.',
                 formal_training=False, completed_at=time.time())
    write(root / 'data.json', data)
    write(root / 'ready.json', ready)
    write(CURRENT, ready)
    state.update(status='complete', ready_manifest=str(CURRENT), archive=info, completed_at=time.time())
    write(root / 'state.json', state)
    p = Path('/home/yzhong239/AGENTS.md')
    text = p.read_text()
    heading = '## 当前数据：恢复默认2K缓存，清理16K缓存（2026-11-02）\n'
    message = f'\n完成：CPU数据job {state["job"]} exit0；2K train/dev/test与原缓存SHA逐一一致，完整10B缓存与128step检查前缀已核验。当前入口 {CURRENT}；URI {state["data_uri"]}；archive SHA {info["sha256"]}。无训练任务、无W&B。\n'
    if heading in text and message not in text:
        p.write_text(text.replace(heading, heading + message, 1))


def watch(root):
    with (root / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        errors = 0
        while not (root / 'STOPPED.json').exists():
            state = json.loads((root / 'state.json').read_text())
            if state['status'] in ('complete', 'needs_attention', 'stopped_by_user'):
                return
            try:
                attrs = 'ClusterId,ProcId,JobStatus,ExitCode,HoldReason,NumJobStarts,RemoteHost'
                ads = json.loads(command(['condor_q', state['job'], '-json', '-attributes', attrs], root) or '[]')
                if not ads:
                    ads = json.loads(command(['condor_history', state['job'], '-limit', '1', '-json', '-attributes', attrs], root) or '[]')
                state.update(last_poll=time.time(), monitor_pid=os.getpid(), last_job=ads[0] if ads else None)
                write(root / 'state.json', state)
                if ads and ads[0]['JobStatus'] in (3, 4, 5):
                    assert ads[0]['JobStatus'] == 4 and ads[0].get('ExitCode') == 0, ads
                    assert not (root / 'STOPPED.json').exists()
                    collect(root, state)
                    return
                errors = 0
            except Exception as exc:
                errors += 1
                state.update(monitor_errors=errors, last_error=repr(exc))
                print(repr(exc), flush=True)
                if isinstance(exc, (AssertionError, RuntimeError)) or errors >= 5:
                    state['status'] = 'needs_attention'
                    write(root / 'state.json', state)
                    return
            state['monitor_errors'] = errors
            write(root / 'state.json', state)
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('action', choices=('submit', 'watch'))
    parser.add_argument('--packing', choices=('native', 'indexed4', 'indexed8'), default='native')
    parser.add_argument('--profile-report', type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(exist_ok=True)
    if args.action == 'submit':
        submit(args.root, args.packing, args.profile_report)
    else:
        watch(args.root)
