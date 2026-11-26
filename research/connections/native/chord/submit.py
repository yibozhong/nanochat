"""Submit nanochat-1b experiments in one sequential job on Hopper."""
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
from nanochat.experiment_logging import BATCH_TOKENS, ENTITY, PROJECT, STEPS
from nanochat.procedural import DYCK, TASKS, ProceduralData
from nanochat.research_protocol import validate_identity

CHORD_ARGS = {
    'chord_scalar_bf16': HERE.parents[1] / 'ngpt/chord.args',
    'chord_next_bf16': HERE.parents[1] / 'ngpt/chord_next.args',
}
RESIDUAL_STUDY = ('vanilla_baseline', 'native_baseline', 'vanilla_attnres', 'vanilla_mhc', *CHORD_ARGS)
RECIPES = (*RESIDUAL_STUDY, 'vanilla_baseline_zero', 'mhc')
SUITE = 'residual_suite'
ABLATION_SUITE = 'scalar_ablation_suite'
ABLATION_ARGS = {
    'chord_scalar_from_rest': CHORD_ARGS['chord_scalar_bf16'],
    'chord_scalar_sphere_only': HERE.parents[1] / 'ngpt/scalar_sphere_only.args',
    'chord_scalar_lookahead_only': HERE.parents[1] / 'ngpt/scalar_lookahead_only.args',
}
SUITES = {SUITE: RESIDUAL_STUDY, ABLATION_SUITE: tuple(ABLATION_ARGS)}


def run(*args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=90).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def chord_args(recipe):
    flags = (CHORD_ARGS | ABLATION_ARGS)[recipe].read_text().splitlines()
    if recipe == 'chord_scalar_from_rest':
        flags[flags.index('--velocity-init=token')] = '--velocity-init=zero'
    return flags


def make_source(recipe, key, steps, group, name, baseline, commit, data):
    run_id = uuid.uuid4().hex[:8]
    source = dict(commit=commit, run_id=run_id, plan=key, formal_training=bool(steps), seed=42,
        wandb_project=PROJECT, group=group, run_name=name, baseline_run=baseline, recipe=recipe,
        initialization='fresh', hopper_only=True,
        steps=steps, schedule_steps=STEPS, batch_tokens=32768, training_tokens=steps*32768,
        microbatch=16,
        data_uri=data['data_uri'],
        dataset_revision=data['manifest']['dataset_revision'], tokenizer=data['manifest']['tokenizer'],
        runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock', checkpoint_seconds=1800)
    native = recipe in ('native_baseline', 'mhc')
    variant = {'vanilla_attnres': 'attnres', 'vanilla_mhc': 'mhc', 'mhc': 'mhc'}.get(recipe, 'baseline')
    streams = 4 if variant == 'mhc' else 1
    arch, backend, window = ('native', 'fa3', 'SSSL') if native else ('vanilla', 'fa2', 'L')
    source.update(architecture=arch, attention_backend=backend, window_pattern=window,
                  connection_variant=variant, streams=streams, matched_backbone_init=True,
                  residual_projection_init='zero', output_projection_actual_std=0.)
    if recipe in CHORD_ARGS or recipe in ABLATION_ARGS:
        flags = chord_args(recipe)
        options = dict(flag[2:].split('=', 1) for flag in flags if '=' in flag)
        source.update(model_args=flags, velocity_init=options['velocity-init'],
                      model_variant=options.get('chord-six-controls', options.get('ngpt-velocity-mode', options['residual-dynamics'])),
                      residual_projection_init=options['residual-projection-init'],
                      control_peak_lr=.008*.25*(768/512)**.5)
        if recipe == 'chord_next_bf16':
            source.update(next_run='2574b7dc', next_source='lululu39/NextTransformer@b8bbfa41d8ab4b31eb5447cbde6271fe84074b78',
                          residual_update='split(force / sqrt(mean(force^2) + 0.05^2))',
                          modulation_scale_init=.01, paired_preflight='chord_scalar_bf16',
                          velocity_projection_weight_std=.02/512**.5, velocity_projection_optimizer='muon',
                          velocity_projection_peak_lr=.02*.25)
    else:
        flags = [f'--variant={variant}', f'--architecture={arch}', f'--streams={streams}',
                 f'--attention-backend={backend}', f'--window-pattern={window}', '--residual-projection-init=zero']
        if variant == 'attnres':
            flags += ['--attnres-block-size=2']
        if variant == 'mhc':
            flags += ['--connection-backend=nvidia', '--sinkhorn-iterations=20']
        if recipe == 'mhc':
            flags = (HERE/'mhc.args').read_text().splitlines()
        source.update(model_args=flags, model_variant=variant)
    return source


def submit(recipe, root, steps, group, name, baseline, procedural=None):
    if recipe not in (*RECIPES, *SUITES):
        raise ValueError(f'Unknown active recipe: {recipe}')
    if recipe == SUITE and baseline != 'self':
        raise ValueError('The residual suite pairs its own baseline runs')
    if recipe == ABLATION_SUITE and baseline == 'self':
        raise ValueError('Select the paired vanilla baseline run for Scalar Chord ablations')
    if procedural and recipe in SUITES:
        raise ValueError('Select the single model recipe for the procedural comparison')
    key = root.name
    ad_key = 'NativeMhcKey' if recipe == 'mhc' else 'VanillaChordKey'
    if (root/'state.json').exists():
        state = json.loads((root/'state.json').read_text())
        pinned = state['source']
        if pinned.get('procedural') != procedural:
            raise ValueError('This job directory has a different procedural configuration')
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
    if recipe in SUITES:
        sources = []
        for member in SUITES[recipe]:
            suffix = '-zero-s42-1b' if recipe == SUITE else '-v0zero-s42-1b'
            member_name = member.replace('_', '-') + suffix
            reference = baseline
            if (recipe == SUITE and member not in RESIDUAL_STUDY[:2]) or (recipe == ABLATION_SUITE and sources):
                reference = sources[0]['run_id']
            sources.append(make_source(member, key, steps, group, member_name, reference, commit, data))
        run_id = sources[0]['run_id']
        source = dict(recipe=recipe, run_id=run_id, group=group, run_name=name, baseline_run=baseline, runs=sources)
    else:
        source = make_source(recipe, key, steps, group, name, baseline, commit, data)
        if procedural:
            source['procedural'] = procedural
        run_id = source['run_id']
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
            line = 'request_disk = 100GB' if recipe in SUITES else 'request_disk = 80GB'
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
    assert ad[ad_key]==key and ad['RequestGPUs']==1 and ad['RequestDisk']==(100 if recipe in SUITES else 80)*1024**2
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
    parser.add_argument('--recipe', choices=(*RECIPES, *SUITES), required=True)
    parser.add_argument('--group', required=True, help='Research question slug, shared by its configurations')
    parser.add_argument('--name', help='Configuration name (defaults to recipe, seed and budget)')
    parser.add_argument('--baseline-run', required=True, help='nanochat-1b baseline ID or self')
    parser.add_argument('--debug-only', action='store_true', help='Local checks only, without a W&B run')
    parser.add_argument('--procedural-task', choices=TASKS)
    parser.add_argument('--procedural-tokens', type=int, default=0)
    parser.add_argument('--procedural-input-length', type=int)
    args = parser.parse_args()
    procedural = None
    if args.procedural_task or args.procedural_tokens or args.procedural_input_length is not None:
        if not args.procedural_task or not 0 < args.procedural_tokens <= (STEPS-1)*BATCH_TOKENS:
            parser.error('Select a procedural task and a positive token budget with room for FineWeb')
        length = args.procedural_input_length
        if length is None:
            length = 2048 if args.procedural_task in DYCK else 64
        try:
            ProceduralData(args.procedural_task, 42, 32768, 2048, length)
        except ValueError as error:
            parser.error(str(error))
        procedural = dict(task=args.procedural_task, tokens=args.procedural_tokens, length=length)
    steps = 0 if args.debug_only else STEPS
    suffix = f"-{procedural['task']}-len{length}-{procedural['tokens']}t" if procedural else ''
    name = args.name or args.recipe.replace('_', '-') + suffix + '-s42-1b'
    group, name, baseline = validate_identity(args.group, name, args.baseline_run)
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        parser.error('The configuration name needs an ASCII letter or digit for its job directory')
    key = f'{PROJECT}-{group}--{slug}' + ('-debug' if args.debug_only else '')
    root = HOME / 'nanochat-experiments' / key
    root.mkdir(parents=True, exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        submit(args.recipe, root, steps, group, name, baseline, procedural)
