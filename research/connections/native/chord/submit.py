"""Submit a nanochat-1b baseline, Chord, or native mHC experiment on Hopper."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
HOME = Path.home()
ASSETS = HOME / 'nanochat-runtime-assets'
sys.path.insert(0, str(REPO))
from nanochat.experiment_logging import ENTITY, PROJECT, STEPS
from nanochat.research_protocol import validate_identity

CHORD_ARGS = {
    'chord_scalar_bf16': HERE.parents[1] / 'ngpt/chord.args',
    'chord_next_bf16': HERE.parents[1] / 'ngpt/chord_next.args',
}
RECIPES = ('vanilla_baseline', 'vanilla_baseline_zero', *CHORD_ARGS, 'mhc')


def run(*args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=90).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def submit(recipe, root, steps, group, name, baseline):
    if recipe not in RECIPES:
        raise ValueError(f'Unknown active recipe: {recipe}')
    key = root.name
    ad_key = 'NativeMhcKey' if recipe == 'mhc' else 'VanillaChordKey'
    if (root/'state.json').exists():
        state = json.loads((root/'state.json').read_text())
        pinned = state['source']
        if (pinned['recipe'], pinned['group'], pinned['run_name'], pinned['baseline_run']) != (recipe, group, name, baseline):
            raise ValueError('This job directory belongs to a different configuration; choose a distinct --name')
        print(state['job'], state['wandb_url'])
        return
    assert not any((root/marker).exists() for marker in ('submission-intent.json','STOPPED','STOPPED.json'))
    assert not json.loads(run('condor_q', '-constraint', f'{ad_key} == "{key}"', '-json') or '[]')
    assert not run('git', 'status', '--porcelain', '--', '.', ':(exclude)research/RESULTS.csv',
                   ':(exclude)research/OVERVIEW.md', ':(exclude)research/*/REPORT.md',
                   ':(exclude)research/*/curves*.png', ':(exclude)research/*/curves.csv', ':(exclude)research/*/configs/*.json',
                   ':(exclude)research/LEADERBOARD.md', ':(exclude)research/index.html'), 'Commit training code before creating the archive'
    for d in ('bundles','logs','results'):
        (root/d).mkdir(exist_ok=True)
    bundles = root/'bundles'
    commit = run('git','rev-parse','HEAD')
    run('git','archive','--prefix=nanochat-src/','-o',str(bundles/'source.tar'),commit)
    shutil.copy2(HERE/'train.sh', root/'train.sh')
    wheel, = (ASSETS/'bundles').glob('transformer_engine_torch-*.whl')
    (bundles/wheel.name).symlink_to(wheel)
    data = json.loads((HOME/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
    assert data['ready'] and data['manifest']['training_steps'] >= STEPS
    run_id = uuid.uuid4().hex[:8]
    source = dict(commit=commit, run_id=run_id, plan=key, formal_training=bool(steps), seed=42,
        wandb_project=PROJECT, group=group, run_name=name, baseline_run=baseline, recipe=recipe,
        initialization='fresh', hopper_only=True,
        steps=steps, schedule_steps=STEPS, batch_tokens=32768, training_tokens=steps*32768,
        microbatch=16,
        data_uri=data['data_uri'],
        dataset_revision=data['manifest']['dataset_revision'], tokenizer=data['manifest']['tokenizer'],
        runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock', checkpoint_seconds=1800)
    if recipe == 'mhc':
        source.update(architecture='native', attention_backend='fa3', window_pattern='SSSL',
                      model_args=(HERE/'mhc.args').read_text().splitlines(),
                      connection_variant='mhc', streams=4, sinkhorn_iterations=20,
                      residual_projection_init='normal', output_projection_actual_std=.005)
    else:
        source.update(architecture='vanilla', attention_backend='fa2', window_pattern='L',
                      control_optimizer_reference='vanilla_chord')
        if recipe in CHORD_ARGS:
            flags = CHORD_ARGS[recipe].read_text().splitlines()
            options = dict(flag[2:].split('=', 1) for flag in flags if '=' in flag)
            source.update(model_args=flags, velocity_init=options['velocity-init'],
                          model_variant=options.get('chord-six-controls', options['ngpt-velocity-mode']),
                          residual_projection_init=options['residual-projection-init'],
                          control_peak_lr=.008*.25*(768/512)**.5)
            if recipe == 'chord_next_bf16':
                source.update(next_run='2574b7dc', next_source='lululu39/NextTransformer@b8bbfa41d8ab4b31eb5447cbde6271fe84074b78',
                              residual_update='split(force / sqrt(mean(force^2) + 0.05^2))',
                              modulation_scale_init=.01, paired_preflight='chord_scalar_bf16',
                              velocity_projection_weight_std=.02/512**.5, velocity_projection_optimizer='muon',
                              velocity_projection_peak_lr=.02*.25)
        else:
            init = 'zero' if recipe == 'vanilla_baseline_zero' else 'normal'
            source.update(model_variant='vanilla', velocity_init='zero', residual_projection_init=init,
                          model_args=['--variant=baseline', '--architecture=vanilla', '--streams=1',
                                      '--attention-backend=fa2', '--residual-projection-init='+init])
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
                 wandb_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}' if steps else None)
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
    parser.add_argument('--recipe', choices=RECIPES, required=True)
    parser.add_argument('--group', required=True, help='Research question slug, shared by its configurations')
    parser.add_argument('--name', help='Configuration name (defaults to recipe, seed and budget)')
    parser.add_argument('--baseline-run', required=True, help='nanochat-1b baseline ID or self')
    parser.add_argument('--debug-only', action='store_true', help='Local checks only, without a W&B run')
    args = parser.parse_args()
    steps = 0 if args.debug_only else STEPS
    name = args.name or args.recipe.replace('_', '-') + '-s42-1b'
    group, name, baseline = validate_identity(args.group, name, args.baseline_run)
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        parser.error('The configuration name needs an ASCII letter or digit for its job directory')
    key = f'{PROJECT}-{group}--{slug}' + ('-debug' if args.debug_only else '')
    root = HOME / 'nanochat-experiments' / key
    root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(args.recipe, root, steps, group, name, baseline)
