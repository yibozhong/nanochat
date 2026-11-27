"""Validate each pinned configuration before sequential training on a single GPU."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .submit import ABLATION_ARGS, BATCH_TOKENS, CHORD_ARGS, RECIPES, STEPS, SUITE, SUITES, TOKEN_COMPONENTS, chord_args, save

ROOT = Path('..')
RESULTS = Path('results')
FA3_REVISION = 'de87b9b5af06dd9984df595bef90b2eba44b181a'


def read(path):
    return json.loads(path.read_text())


def manifest_runs(manifest, run_id, recipe):
    assert (manifest['run_id'], manifest['recipe']) == (run_id, recipe)
    runs = manifest['runs'] if recipe in SUITES else [manifest]
    if recipe in SUITES:
        assert tuple(s['recipe'] for s in runs) == SUITES[recipe]
        assert len({s['run_id'] for s in runs}) == len(runs) and runs[0]['run_id'] == run_id
        for index, s in enumerate(runs):
            if recipe == SUITE:
                assert s['baseline_run'] == ('self' if index < 2 else run_id)
            else:
                assert s['baseline_run'] == (manifest['baseline_run'] if index == 0 else run_id)
                assert s['velocity_init'] == 'zero'
            assert s['group'] == manifest['group']
    for s in runs:
        assert s['recipe'] in (*RECIPES, *ABLATION_ARGS)
        assert s['schedule_steps'] == STEPS and s['steps'] in (0, STEPS)
        assert s['seed'] == 42 and s['formal_training'] == bool(s['steps']) and s['initialization'] == 'fresh'
        assert s['wandb_project'] == 'nanochat-1b' and s['matched_backbone_init'] and s['residual_projection_init'] == 'zero'
        args = dict(option[2:].split('=', 1) for option in s['model_args'] if '=' in option)
        assert args['residual-projection-init'] == 'zero' and args['variant'] == s['connection_variant']
        assert (args['architecture'], args['attention-backend']) == (s['architecture'], s['attention_backend'])
        if s['recipe'] == 'vanilla_sequential_backward':
            assert '--sequential-backward' in s['model_args'] and s['backward_method'] == 'sequential'
        if s['recipe'] in ABLATION_ARGS or s['recipe'] in TOKEN_COMPONENTS:
            assert s['model_args'] == chord_args(s['recipe'])
        if s['recipe'] in CHORD_ARGS:
            geometry = args['chord-velocity-geometry']
            assert geometry == s.get('chord_velocity_geometry', geometry)
            assert s['model_args'] == chord_args(s['recipe'], s['velocity_init'], geometry, s.get('chord_zero_mu', False))
            assert args['chord-six-controls'] == s['chord_controls'] and '--chord-update-based' in s['model_args']
            assert args['chord-soft-force-tau'] == '0.05' and args['chord-state-dtype'] == 'bfloat16'
    return runs


def environment(s):
    env = os.environ.copy()
    if s['attention_backend'] == 'fa3':
        env.pop('NANOCHAT_DISABLE_FA3', None)
        env.update(HF_HOME=str((ROOT/'fa3-cache').resolve()), NANOCHAT_FA3_REVISION=FA3_REVISION, HF_HUB_OFFLINE='0')
    else:
        env['NANOCHAT_DISABLE_FA3'] = '1'
    return env


def train_args(s, *, smoke=False):
    warm = s.get('procedural')
    flags = [] if not warm else [
        '--procedural-task', warm['task'], '--procedural-input-length', str(warm['length']),
        '--procedural-tokens', str(2*BATCH_TOKENS if smoke else warm['tokens'])]
    if warm and warm.get('reset_vocab'):
        flags.append('--procedural-reset-vocab')
    if warm and 'reference_updates' in warm:
        flags.extend(['--procedural-reference-updates', str(8 if smoke else warm['reference_updates'])])
    return [sys.executable, '-m', 'scripts.connection_train', *flags, *s['model_args'], '--matched-backbone-init',
            '--schedule-steps', str(s['schedule_steps']), '--seed', str(s['seed']),
            '--microbatch', str(s['microbatch']), '--data-dir', '../data']


def preflight(s, directory):
    marker = directory/'preflight.json'
    if marker.exists():
        report = read(marker)
        assert report['source'] == s
        return report
    print('RESIDUAL_PHASE preflight', s['run_id'], s['recipe'], flush=True)
    save(Path('source.json'), s)
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(environment(s), WANDB_MODE='disabled')
    if s['recipe'] == 'vanilla_sequential_backward':
        from .sequential_comparison import compare
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_sequential_backward.py'],
                       env=env, check=True, timeout=1800)
        compare(s, directory, env)
    if s.get('procedural'):
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_procedural.py'], env=env, check=True, timeout=900)
    if s['recipe'] in ABLATION_ARGS:
        test = s['recipe'].removeprefix('chord_scalar_')
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_scalar_components.py::test_' + test],
                       env=env, check=True, timeout=600)
    if s['recipe'] in TOKEN_COMPONENTS:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_token_components.py',
                        '-k', s['recipe'].removeprefix('chord_token_')], env=env, check=True, timeout=600)
    if s['recipe'] == 'chord_scalar_bf16':
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-s', 'tests/test_chord_soft_scalar.py',
                        'tests/test_chord_six_controls.py'],
                       env=env, check=True, timeout=600)
    if s['recipe'] == 'chord_next_bf16':
        subprocess.run([sys.executable, '-m', 'research.connections.ngpt.next_soft_preflight'],
                       env=env, check=True, timeout=2400)
    # Diagnostic checkpoints and optimizer state live only in scratch.
    with tempfile.TemporaryDirectory(dir=ROOT, prefix='residual-smoke-') as tmp:
        out = Path(tmp)
        args = [*train_args(s, smoke=True), '--output-dir', tmp, '--steps', '4', '--run-id', 'smoke-'+s['run_id']]
        first_stop = 1 if s.get('procedural') else 2
        first = subprocess.run([*args, '--check-zero-init-gradients', '--segment-steps', str(first_stop)],
                               env=env, timeout=2400)
        if first.returncode != 85 or read(out/'progress.json')['step'] != first_stop:
            raise RuntimeError(f"{s['recipe']} failed to checkpoint at diagnostic step {first_stop} (exit {first.returncode})")
        if s.get('procedural'):
            second = subprocess.run([*args, '--resume', 'auto', '--segment-steps', '1'], env=env, timeout=2400)
            if second.returncode != 85 or read(out/'progress.json')['step'] != 2:
                raise RuntimeError(f"Procedural diagnostic failed at transfer (exit {second.returncode})")
            boundary = read(out/'transfer-check.json')
            assert boundary['identical_tensors'] > 0 and boundary['optimizer_state_entries'] == 0
            progress = read(out/'progress.json')
            assert progress['fineweb_tokens'] == 0 and progress['procedural_supervised_tokens'] > 0
        subprocess.run([*args, '--resume', 'auto'], env=env, check=True, timeout=2400)
        summary, config = read(out/'summary.json'), read(out/'config.json')
        rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows] == [1, 2, 3, 4] and all(math.isfinite(r['loss']) for r in rows)
        if s.get('procedural'):
            assert [r['phase'] for r in rows] == ['procedural', 'procedural', 'fineweb', 'fineweb']
            assert [r['phase_step'] for r in rows] == [1, 2, 1, 2]
            assert rows[2]['lr_multiplier'] == 1/40
            if 'reference_updates' in s['procedural']:
                assert rows[0]['lr_multiplier'] == rows[1]['lr_multiplier'] == 1.0
                assert [r['procedural_optimizer_updates'] for r in rows] == [4, 8, 8, 8]
            else:
                assert rows[0]['lr_multiplier'] == 1/40
            if s['procedural'].get('reset_vocab'):
                assert boundary['reinitialized_tensors'] == ['lm_head.weight', 'wte.weight']
                assert summary['procedural_val_ce'] == boundary['pre_reset_union_val_ce']
            assert math.isfinite(summary['procedural_val_ce'])
            assert summary['fineweb_tokens'] == summary['procedural_tokens'] == 2*BATCH_TOKENS
        assert summary['complete'] and summary['source'] == s and math.isfinite(summary['final_dev_ce'])
        assert config['logging_backend'] == 'local' and config['actual_attention_backend'] == s['attention_backend']
        report = dict(source=s, first_gradients=read(out/'first-gradients.json'), gpu=config['gpu'],
                      step_losses=[r['loss'] for r in rows], resumed_at=2, final_dev_ce=summary['final_dev_ce'])
        if s.get('procedural'):
            report.update(transfer=boundary, resumed_steps=[1, 2], procedural_val_ce=summary['procedural_val_ce'],
                          fineweb_tokens=summary['fineweb_tokens'],
                          procedural_supervised_tokens=summary['procedural_supervised_tokens'])
    save(marker, report)
    return report


def finished(s, out, ckpt):
    if (out/'summary.json').exists():
        summary = read(out/'summary.json')
        assert summary['complete'] and summary['source'] == s and summary['run_id'] == s['run_id']
        assert summary['steps'] == s['steps'] and summary['test_evaluated']
        for name, meta in summary['checkpoint_files'].items():
            assert (ckpt/name).stat().st_size == meta['bytes']
        return True
    if (out/'progress.json').exists():
        progress = read(out/'progress.json')
        assert progress['source'] == s and progress['run_id'] == s['run_id']
        assert all((ckpt/name).stat().st_size for name in ('latest.pt', 'best.pt'))
    return False


def main(run_id, recipe):
    runs = manifest_runs(read(ROOT/'source.json'), run_id, recipe)
    suite = recipe in SUITES
    folders = [RESULTS/s['run_id'] if suite else RESULTS for s in runs]
    reports = [preflight(s, folder) for s, folder in zip(runs, folders)]
    if recipe == SUITE:
        ref = reports[0]['first_gradients']['output_projection_norms']
        for s, report in zip(runs, reports):
            if s['recipe'] in CHORD_ARGS:
                norms = report['first_gradients']['output_projection_norms']
                ratios = [g/b for g, b in zip(norms, ref, strict=True)]
                print('CHORD_FIRST_GRAD_RATIOS', s['recipe'], ratios, flush=True)
                if max(ratios) > 10:
                    raise FloatingPointError('Chord output gradient exceeds 10x the corresponding vanilla projection')
                save(RESULTS/s['run_id']/'gradient-ratios.json', ratios)
    for s, folder in zip(runs, folders):
        if not s['formal_training']:
            continue
        ckpt = ROOT/'checkpoints'/s['run_id'] if suite else ROOT/'checkpoints'
        out = folder/'train'
        if finished(s, out, ckpt):
            continue
        save(Path('source.json'), s)
        print('RESIDUAL_PHASE formal', s['run_id'], s['recipe'], flush=True)
        print('FORMAL_TRAINING_BEGIN', s['group'], s['run_id'], flush=True)
        args = [*train_args(s), '--steps', str(s['steps']), '--evaluate-test', '--output-dir', str(out),
                '--checkpoint-dir', str(ckpt), '--formal-training', '--wandb-group', s['group'],
                '--run-id', s['run_id'], '--resume', 'auto', '--segment-seconds', str(s['checkpoint_seconds'])]
        proc = subprocess.run(args, env=environment(s))
        if proc.returncode == 85:
            return 85
        proc.check_returncode()
        assert finished(s, out, ckpt)
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
