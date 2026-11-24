"""Submit the paired native three-scalar Chord recipe once, on a full Hopper."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HOME = Path.home()
ASSETS = HOME / 'nanochat-runtime-assets'
KEY = 'native-chord-three-scalar-token-bf16-10b-20261123'
ROOT = HOME / KEY


def run(*args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=90).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def submit():
    if (ROOT/'state.json').exists():
        print((ROOT/'state.json').read_text())
        return
    assert not any((ROOT/name).exists() for name in ('submission-intent.json','STOPPED','STOPPED.json'))
    assert not json.loads(run('condor_q', '-constraint', f'NativeChordKey == "{KEY}"', '-json') or '[]')
    assert not run('git', 'status', '--porcelain'), 'Commit the source before creating the archive'
    for d in ('bundles','logs','results'):
        (ROOT/d).mkdir(exist_ok=True)
    bundles = ROOT/'bundles'
    commit = run('git','rev-parse','HEAD')
    run('git','archive','--prefix=nanochat-src/','-o',str(bundles/'source.tar'),commit)
    shutil.copy2(HERE/'train.sh', ROOT/'train.sh')
    wheel, = (ASSETS/'bundles').glob('transformer_engine_torch-*.whl')
    (bundles/wheel.name).symlink_to(wheel)
    data = json.loads((HOME/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
    assert data['ready'] and data['manifest']['training_steps'] == 305176
    run_id = uuid.uuid4().hex[:8]
    source = dict(commit=commit, run_id=run_id, plan=KEY, formal_training=True, seed=42,
        initialization='fresh', architecture='native', native_chord_write='three_scalar',
        steps=305176, schedule_steps=305176, batch_tokens=32768, training_tokens=305176*32768,
        microbatch=16, attention_backend='fa3', window_pattern='SSSL', velocity_init='token',
        model_args=(HERE/'train.args').read_text().splitlines(), data_uri=data['data_uri'],
        dataset_revision=data['manifest']['dataset_revision'], tokenizer=data['manifest']['tokenizer'],
        runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock', checkpoint_seconds=1800)
    save(bundles/'source.json', source)
    # Use the current site's container, transfer plugins, encrypted auth and pool settings.
    lines = []
    for line in (ASSETS/'gate.sub').read_text().splitlines():
        if line.startswith('queue '):
            continue
        line = line.replace('validate.sh', 'train.sh').replace('logs/stream.', 'logs/train.')
        line = line.replace('DepthDiagnosticKey', 'NativeChordKey').replace('depth-diagnostic-20261103', KEY)
        line = line.replace('nanochat-depth-diagnostic', KEY)
        if line.startswith('transfer_input_files'):
            line = line.replace('bundles/gate-data.tar,bundles/inputs.sha256', data['data_uri']+','+str(HOME/'.netrc'))
        elif line.startswith('request_disk'):
            line = 'request_disk = 80GB'
        elif line.startswith('require_gpus'):
            line = line.replace('(Capability == 9.0)', '(Capability == 9.0 && regexp("H100|H200",DeviceName))')
        elif line.startswith('transfer_output_files'):
            line = 'transfer_output_files = results.tar,checkpoints.tar'
        elif line.startswith('transfer_output_remaps'):
            line = 'transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
        lines.append(line)
    lines += [f'arguments = {run_id}', 'checkpoint_exit_code = 85',
              'transfer_checkpoint_files = checkpoints.tar,results.tar', f'encrypt_input_files = {HOME}/.netrc',
              '+InteractiveJob = true', 'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
              'hold = true', 'queue 1']
    (ROOT/'train.sub').write_text('\n'.join(lines)+'\n')
    run('condor_submit','-dry-run','dry-run.ad','train.sub',cwd=ROOT)
    state = dict(run_id=run_id, source=source, status='intent',
                 wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
    save(ROOT/'submission-intent.json', state)
    job = run('condor_submit','-terse','train.sub',cwd=ROOT).split()[0]
    state.update(job=job,status='held_for_audit')
    save(ROOT/'state.json',state)
    ad, = json.loads(run('condor_q',job,'-json'))
    assert ad['JobStatus']==5 and not ad.get('NumJobStarts',0) and ad['Args']==run_id
    assert ad['NativeChordKey']==KEY and ad['RequestGPUs']==1 and ad['RequestDisk']==80*1024**2
    assert ad['RequestMemory']==16384 and ad['RequestCpus']==2 and ad['InteractiveJob']
    assert 'H100|H200' in ad['RequireGPUs'] and ad['GPUsMinCapability']==9
    assert data['data_uri'] in ad['TransferInput'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
    assert 'source.tar' in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
    assert ad['SuccessCheckpointExitCode']==85
    assert set(ad['TransferCheckpoint'].split(','))==set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
    save(ROOT/'submission-verified.json',{k:ad[k] for k in ('ClusterId','ProcId','Args','RequireGPUs','TransferInput','RequestMemory','RequestDisk','TransferCheckpoint','EncryptInputFiles')})
    run('condor_release',job)
    state['status']='submitted'
    save(ROOT/'state.json',state)
    print(job, state['wandb_url'])


if __name__ == '__main__':
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit()
