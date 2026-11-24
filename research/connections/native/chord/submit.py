"""Submit one native Chord or matched mHC run on a full Hopper."""
import argparse
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
KEYS = {'chord': 'native-chord-three-scalar-token-bf16-10b-20261123',
        'mhc': 'native-mhc-normal-init-10b-20261123',
        'token_vnorm': 'native-chord-token-controls-vnorm-10b-20261124',
        'token_bounded': 'native-chord-token-controls-bounded-10b-20261124',
        'token_h': 'native-chord-token-h-only-native-lr-10b-20261124'}


def run(*args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=90).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def submit(recipe, root):
    key = root.name
    ad_key = 'NativeMhcKey' if recipe == 'mhc' else 'NativeChordKey'
    if (root/'state.json').exists():
        print((root/'state.json').read_text())
        return
    assert not any((root/name).exists() for name in ('submission-intent.json','STOPPED','STOPPED.json'))
    assert not json.loads(run('condor_q', '-constraint', f'{ad_key} == "{key}"', '-json') or '[]')
    assert not run('git', 'status', '--porcelain'), 'Commit the source before creating the archive'
    for d in ('bundles','logs','results'):
        (root/d).mkdir(exist_ok=True)
    bundles = root/'bundles'
    commit = run('git','rev-parse','HEAD')
    run('git','archive','--prefix=nanochat-src/','-o',str(bundles/'source.tar'),commit)
    shutil.copy2(HERE/'train.sh', root/'train.sh')
    wheel, = (ASSETS/'bundles').glob('transformer_engine_torch-*.whl')
    (bundles/wheel.name).symlink_to(wheel)
    data = json.loads((HOME/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
    assert data['ready'] and data['manifest']['training_steps'] == 305176
    run_id = uuid.uuid4().hex[:8]
    source = dict(commit=commit, run_id=run_id, plan=key, formal_training=True, seed=42,
        initialization='fresh', architecture='native', hopper_only=True,
        steps=305176, schedule_steps=305176, batch_tokens=32768, training_tokens=305176*32768,
        microbatch=16, attention_backend='fa3', window_pattern='SSSL',
        model_args=(HERE/('mhc.args' if recipe == 'mhc' else 'train.args')).read_text().splitlines(),
        data_uri=data['data_uri'],
        dataset_revision=data['manifest']['dataset_revision'], tokenizer=data['manifest']['tokenizer'],
        runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock', checkpoint_seconds=1800)
    if recipe == 'mhc':
        source.update(connection_variant='mhc', streams=4, sinkhorn_iterations=20,
                      residual_projection_init='normal', output_projection_actual_std=.005)
    elif recipe in ('token_vnorm', 'token_bounded', 'token_h'):
        flags = source['model_args']
        flags[flags.index('--native-chord-write=three_scalar')] = f'--native-chord-write={recipe}'
        source.update(native_chord_write=recipe, velocity_init='token', gate_weight_init=0., extra_parameters=95232)
        if recipe == 'token_h':
            source.update(gate_input='current_sublayer_h', control_optimizer_reference='nanochat_native_mhc_router',
                          extra_parameters=47616, control_maps=['linear','softplus','softplus','sigmoid','linear','linear'])
        elif recipe == 'token_bounded':
            source.update(gate_reference='lululu39/NextTransformer@c31127b', gate_input_norm='joint_h_v',
                          gate_joint_rms_eps=1e-6, control_ranges={'mu': [-3, 3], 'alpha': [0, 1],
                                                                 'beta': [0, 1], 'gamma': [0, 2]})
        else:
            source.update(gate_reference='lululu39/NextTransformer@753b013', gate_velocity_rms_eps=1e-6)
    else:
        source.update(native_chord_write='three_scalar', velocity_init='token')
    job_args = f'{run_id} {recipe}'
    save(bundles/'source.json', source)
    # Use the current site's container, transfer plugins, encrypted auth and pool settings.
    lines = []
    for line in (ASSETS/'gate.sub').read_text().splitlines():
        if line.startswith('queue '):
            continue
        line = line.replace('validate.sh', 'train.sh').replace('logs/stream.', 'logs/train.')
        line = line.replace('DepthDiagnosticKey', ad_key).replace('depth-diagnostic-20261103', key)
        line = line.replace('nanochat-depth-diagnostic', key)
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
    lines += [f'arguments = {job_args}', 'checkpoint_exit_code = 85',
              'transfer_checkpoint_files = checkpoints.tar,results.tar', f'encrypt_input_files = {HOME}/.netrc',
              '+InteractiveJob = true', 'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
              'hold = true', 'queue 1']
    (root/'train.sub').write_text('\n'.join(lines)+'\n')
    run('condor_submit','-dry-run','dry-run.ad','train.sub',cwd=root)
    state = dict(run_id=run_id, source=source, status='intent',
                 wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
    save(root/'submission-intent.json', state)
    job = run('condor_submit','-terse','train.sub',cwd=root).split()[0]
    state.update(job=job,status='held_for_audit')
    save(root/'state.json',state)
    ad, = json.loads(run('condor_q',job,'-json'))
    assert ad['JobStatus']==5 and not ad.get('NumJobStarts',0) and ad['Args']==job_args
    assert ad[ad_key]==key and ad['RequestGPUs']==1 and ad['RequestDisk']==80*1024**2
    assert ad['RequestMemory']==16384 and ad['RequestCpus']==2 and ad['InteractiveJob']
    assert 'H100|H200' in ad['RequireGPUs'] and ad['GPUsMinCapability']==9
    assert data['data_uri'] in ad['TransferInput'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
    assert 'source.tar' in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
    assert ad['SuccessCheckpointExitCode']==85
    assert set(ad['TransferCheckpoint'].split(','))==set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
    save(root/'submission-verified.json',{k:ad[k] for k in ('ClusterId','ProcId','Args','RequireGPUs','TransferInput','RequestMemory','RequestDisk','TransferCheckpoint','EncryptInputFiles')})
    run('condor_release',job)
    state['status']='submitted'
    save(root/'state.json',state)
    print(job, state['wandb_url'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=KEYS, default='chord')
    recipe = parser.parse_args().recipe
    root = HOME / 'nanochat-experiments' / KEYS[recipe]
    root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(recipe, root)
