"""Same-GPU alignment measurement and paired checks before the fresh 10B run."""
from functools import partial
import json
import os
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from xml.etree import ElementTree

import torch

from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig, build_connection_model
from nanochat.momentum import normalize_residual, tangent_project
from nanochat.packed_training import PackedData, evaluate, json_write


@torch.no_grad()
def measure_reference(source):
    configure_attention('fa2')
    data = PackedData('../data')
    with tarfile.open('../axis-reference.tar', 'r:') as archive:
        with archive.extractfile('checkpoints/best.pt') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=False)
    assert state['run_id'] == source['reference_run'] and state['step'] == source['reference_best_step']
    assert state['data_manifest'] == data.manifest and state['dev_ce'] == source['reference_best_dev_ce']
    config = ConnectionConfig.from_checkpoint(state['config'])
    assert config.ngpt_velocity_mode == 'lookahead_transport'
    model = build_connection_model(config).prepare().eval()
    model.load_state_dict(state['model'])
    del state
    sites = [s for b in model.blocks for s in (b.attn_residual,b.mlp_residual)]
    totals = torch.zeros(len(sites), 7, dtype=torch.float64, device='cuda')

    def observe(i, module, args):
        h, v, force = args
        tangent = tangent_project(h, normalize_residual(force, module.rms_floor) - h)
        dot = (tangent*v).mean(-1)
        speed2, energy = v.square().mean(-1), tangent.square().mean(-1)
        valid = (speed2 >= 1e-12) & (energy > 0)
        denom = torch.where(valid, speed2*energy, 1.)
        cosine = torch.where(valid, dot*denom.rsqrt(), 0.)
        fraction = cosine.square()
        observations = (valid.sum(), cosine.sum(), fraction.sum(), ((cosine > 0) & valid).sum(),
                        (fraction*energy).sum(), (valid*energy).sum(), speed2.new_tensor(dot.numel()))
        totals[i].add_(torch.stack(observations))

    handles = [s.register_forward_pre_hook(partial(observe,i)) for i,s in enumerate(sites)]
    try:
        dev_ce, tokens = evaluate(model,data,microbatch=source['microbatch'])
    finally:
        for hook in handles:
            hook.remove()
    assert math.isfinite(dev_ce)
    layers = []
    for i,(valid,cosine,fraction,positive,projected,energy,count) in enumerate(totals.tolist()):
        assert count == tokens and valid > 0 and energy > 0
        row = dict(block=i//2+1, sublayer='attention' if i%2==0 else 'ffn',
                   valid_tokens=int(valid), mean_signed_cosine=cosine/valid,
                   mean_squared_cosine=fraction/valid, positive_fraction=positive/valid,
                   projected_tangent_energy_fraction=projected/energy)
        assert all(math.isfinite(val) for val in row.values() if isinstance(val,float))
        assert 0 <= row['projected_tangent_energy_fraction'] <= 1.00001
        layers.append(row)
    result = dict(run_id=source['reference_run'], step=source['reference_best_step'], split='dev',
                  tokens=tokens, dev_ce=dev_ce, layers=layers)
    json_write('results/axis-reference-alignment.json',result)
    print('ARC_REFERENCE_ALIGNMENT',json.dumps(result),flush=True)
    return result


@torch.no_grad()
def endpoint_turns(checkpoint):
    """Measure actual endpoint angles on the same four dev sequences after the timed steps."""
    configure_attention('fa2')
    state = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model = build_connection_model(ConnectionConfig.from_checkpoint(state['config'])).prepare().eval()
    model.load_state_dict(state['model'])
    del state
    turns = []
    def observe(site,args,out):
        h, new_h = args[0], out[0]
        cosine = (h*new_h).mean(-1,keepdim=True)
        sine = (new_h-cosine*h).square().mean(-1,keepdim=True).sqrt()
        angles = torch.atan2(sine,cosine).rad2deg().flatten()
        turns.append(torch.stack((angles.mean(),torch.quantile(angles,.95))))
    handles = [s.register_forward_hook(observe) for b in model.blocks for s in (b.attn_residual,b.mlp_residual)]
    try:
        x,_ = PackedData('../data').batch('dev',0,4)
        model.hidden(x)
    finally:
        for h in handles:
            h.remove()
    assert len(turns) == 2*model.config.depth
    rows = torch.stack(turns).tolist()
    assert all(0<=a<=180 and 0<=b<=180 for a,b in rows)
    return dict(dev_tokens=x.numel(), step=128, sublayers=[
        dict(block=i//2+1,branch='ffn' if i%2 else 'attention',mean_degrees=a,p95_degrees=b)
        for i,(a,b) in enumerate(rows)])


def profile(root, mode, resume, measure_turns=False, signed_steps=False, token_gates=False, token_beta_gate=False,
            geometry='parallel_transport', velocity_init='token_position', state_dtype='float32',
            fused_read=False, remat=False, channel_coefficients=False):
    steps = 128
    out = root/mode
    cmd = [sys.executable, '-m', 'scripts.connection_train', '--variant', 'baseline', '--streams', '1',
           '--residual-dynamics','ngpt_lookahead','--ngpt-velocity-mode',mode,
           '--velocity-init',velocity_init,'--velocity-embedding-std','.02','--momentum-beta-init','.5',
           '--lookahead-mu-init','0','--ngpt-alpha-init','.05','--ngpt-force-rms-floor','1e-6',
           '--residual-projection-init','normal','--residual-projection-init-std','.02',
           '--no-activation-checkpoint','--no-fused-connection-grad','--attention-backend','fa2',
           '--steps',str(steps),'--schedule-steps','305176','--microbatch','16','--seed','42',
           '--data-dir','../data','--output-dir',str(out),'--run-id','axis-local-debug']
    cmd.extend(['--chord-velocity-geometry', geometry, '--chord-state-dtype', state_dtype])
    if channel_coefficients:
        cmd.append('--channel-chord-coefficients')
    if fused_read:
        cmd.append('--fuse-chord-read-norm')
    if remat:
        cmd.append('--recompute-chord-force')
    if signed_steps:
        cmd.append('--signed-chord-steps')
    if token_gates:
        cmd.append('--token-chord-gates')
    if token_beta_gate:
        cmd.append('--token-beta-gate')
    if resume:
        part = subprocess.run([*cmd,'--segment-steps','32'])
        assert part.returncode == 85, part.returncode
        assert json.loads((out/'progress.json').read_text())['step'] == 32
        cmd.extend(['--resume','auto'])
    subprocess.run(cmd,check=True)
    info = json.loads((out/'summary.json').read_text())
    cfg = json.loads((out/'config.json').read_text())
    assert info['complete'] and info['steps'] == steps and info['dev_tokens'] == 131072
    assert cfg['model']['ngpt_velocity_mode'] == mode and cfg['logging_backend'] == 'local'
    assert cfg['model']['chord_velocity_geometry'] == geometry and cfg['model']['velocity_init'] == velocity_init
    assert cfg['model']['chord_state_dtype'] == state_dtype
    assert cfg['model']['fuse_chord_read_norm'] == fused_read and cfg['model']['recompute_chord_force'] == remat
    assert cfg['model']['channel_chord_coefficients'] == channel_coefficients
    assert cfg['model']['signed_chord_steps'] == signed_steps
    assert cfg['model']['token_chord_gates'] == token_gates
    assert cfg['model']['token_beta_gate'] == token_beta_gate
    assert math.isfinite(info['final_dev_ce']) and info['final_dev_ce'] < info['initial_dev_ce']
    rows = [json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1,steps+1))
    assert all(math.isfinite(r['loss']) and 0 < r['seconds'] < math.inf for r in rows)
    events = [json.loads(s)['values'] for s in (out/'diagnostics/axis-local-debug/events.jsonl').read_text().splitlines()]
    end, = [e for e in events if e.get('step') == steps and 'dev/ce' in e]
    kappas = {k:v for k,v in end.items() if k.endswith('/kappa')}
    if mode == 'lookahead_axis_write':
        assert len(kappas) == 2*cfg['model']['depth'] and all(math.isfinite(v) for v in kappas.values())
        assert any(v != 0 for v in kappas.values())
    else:
        assert not kappas
    steps_by_component = {}
    if mode in ('lookahead_split_write','lookahead_split_learned_beta','lookahead_geodesic_split'):
        init, = [e for e in events if e.get('step') == 0 and 'dev/ce' in e]
        for suffix in ('alpha_parallel','alpha_perp'):
            rates = {k:v for k,v in end.items() if k.endswith('/'+suffix)}
            assert len(rates) == 2*cfg['model']['depth'] and all(math.isfinite(v) for v in rates.values())
            if not signed_steps:
                assert all(v>0 for v in rates.values())
            assert all(abs(init[k]-.05)<1e-6 for k in rates)
            assert any(v != init[k] for k,v in rates.items())
            steps_by_component.update(rates)
        assert any(end[f'residual/sublayer_{i}/alpha_parallel'] != end[f'residual/sublayer_{i}/alpha_perp']
                   for i in range(2*cfg['model']['depth']))
    # Identical steps and data, after compile and the target's restart at 32.
    seconds = statistics.median(r['seconds'] for r in rows[64:])
    result = dict(config=cfg['model'], parameters=cfg['parameters'], initial_dev_ce=info['initial_dev_ce'],
                  final_dev_ce=info['final_dev_ce'], median_step_seconds=seconds, tokens_per_second=32768/seconds,
                  peak_allocated_bytes=info['peak_allocated_bytes'], kappas=kappas)
    if steps_by_component:
        result['component_steps'] = steps_by_component
    if mode == 'lookahead_split_learned_beta':
        key = 'beta_at_zero_alignment' if token_beta_gate else 'beta'
        betas = {k:v for k,v in end.items() if k.endswith('/'+key)}
        assert set(betas) == {f'residual/sublayer_{i}/{key}' for i in range(2*cfg['model']['depth']-1)}
        assert all(init[k] == .5 and 0 < v < 1 for k,v in betas.items())
        assert any(v != .5 for v in betas.values())
        result['betas'] = betas
    if token_beta_gate:
        slopes = {k:v for k,v in end.items() if k.endswith('/beta_gate_slope')}
        assert len(slopes) == 2*cfg['model']['depth']-1
        assert all(init[k] == 0 and math.isfinite(v) for k,v in slopes.items())
        assert any(v != 0 for v in slopes.values())
        result['beta_gate_slopes'] = slopes
    if channel_coefficients:
        for key in ('alpha_parallel_std', 'alpha_perp_std', 'beta_std'):
            spread = [v for k,v in end.items() if k.endswith('/'+key)]
            assert spread and all(math.isfinite(v) for v in spread) and max(spread)>0
    if measure_turns:
        result['endpoint_turns'] = endpoint_turns(out/'checkpoints/latest.pt')
        torch.cuda.empty_cache()
    if token_gates:
        slopes = {k:v for k,v in end.items() if k.endswith(('/gate_perp_slope','/gate_parallel_slope'))}
        assert len(slopes) == 4*cfg['model']['depth']
        assert all(init[k] == 0 and math.isfinite(v) for k,v in slopes.items())
        for axis in ('perp','parallel'):
            assert any(v != 0 for k,v in slopes.items() if k.endswith(f'/gate_{axis}_slope'))
        result['gate_slopes'] = slopes
    print('AXIS_WRITE_CONTROL',mode,json.dumps(result),flush=True)
    shutil.rmtree(out)
    return result


def validate_channel_coefficients(source, gpu, marker, start):
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_chord_channel_coefficients.py',
                    '--junitxml=results/chord-channel-tests.xml'], check=True)
    suites = ElementTree.parse('results/chord-channel-tests.xml').getroot().findall('.//testsuite')
    assert sum(int(s.get('tests')) for s in suites) == 5
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ('errors', 'failures', 'skipped'))
    with tempfile.TemporaryDirectory(prefix='chord-channel-', dir='..') as tmp:
        arms = {name: profile(Path(tmp).resolve(), source['ngpt_velocity_mode'], resume=True,
                              geometry='project_only', velocity_init='token', state_dtype='bfloat16',
                              fused_read=True, channel_coefficients=enabled)
                for name, enabled in [('scalar', False), ('channel', True)]}
    scalar, channel = arms['scalar'], arms['channel']
    depth, width = source['model']['depth'], source['model']['width']
    assert channel['parameters'] - scalar['parameters'] == (6*depth - 1)*(width - 1)
    assert abs(channel['initial_dev_ce'] - scalar['initial_dev_ce']) < 1e-4
    assert dict(scalar['config'], channel_chord_coefficients=True) == channel['config']
    summary = {name:{k:v[k] for k in ('parameters','median_step_seconds','tokens_per_second',
                                       'peak_allocated_bytes','initial_dev_ce','final_dev_ce')}
               for name,v in arms.items()}
    json_write(marker, dict(complete=True, formal_training=False, source=source, gpu=gpu, torch=torch.__version__,
                            timed_steps=[65,128], comparison=summary, evaluation_wall_seconds=time.perf_counter()-start))
    print('CHORD_CHANNEL_BENCH_PASSED', json.dumps(summary), flush=True)


def validate_fused_branches(source, gpu, marker, start):
    assert source['chord_state_dtype'] == 'bfloat16' and source['chord_velocity_geometry'] == 'project_only'
    assert source['fuse_chord_read_norm'] and not source['recompute_chord_force']
    env = dict(os.environ, CHORD_TEST_VELOCITY_INIT=source['velocity_init'],
               CHORD_STORAGE_REPORT='results/chord-storage.json')
    subprocess.run([sys.executable,'-m','pytest','-q','tests/test_chord_fused_branch.py',
                    '--junitxml=results/fused-branch-tests.xml'],check=True,env=env)
    suites = ElementTree.parse('results/fused-branch-tests.xml').getroot().findall('.//testsuite')
    assert sum(int(s.get('tests')) for s in suites) == 9
    assert all(int(s.get(k,0)) == 0 for s in suites for k in ('errors','failures','skipped'))
    arms = {}
    with tempfile.TemporaryDirectory(prefix='chord-branches-', dir='..') as tmp:
        # Restart every arm at 32 so compilation and allocator history are matched.
        for name, fused, remat in (('separate',False,False), ('fused',True,False), ('fused_recompute',True,True)):
            result = profile(Path(tmp).resolve(), source['ngpt_velocity_mode'], resume=True,
                             geometry='project_only', velocity_init=source['velocity_init'],
                             state_dtype='bfloat16', fused_read=fused, remat=remat)
            if arms:
                control = arms['separate']
                assert {k:v for k,v in result['config'].items() if control['config'][k] != v} == {
                    k:True for k in ('fuse_chord_read_norm','recompute_chord_force') if result['config'][k]}
                assert result['parameters'] == control['parameters']
                assert abs(result['initial_dev_ce']-control['initial_dev_ce']) < 1e-4
            arms[name] = result
    summary = {name:{k:r[k] for k in ('median_step_seconds','tokens_per_second','peak_allocated_bytes','final_dev_ce')}
               for name,r in arms.items()}
    json_write(marker, dict(complete=True, formal_training=False, source=source, gpu=gpu, torch=torch.__version__,
                            tests=9, timed_steps=[65,128], results=arms, comparison=summary,
                            residual_memory=json.loads(Path('results/chord-storage.json').read_text()),
                            evaluation_wall_seconds=time.perf_counter()-start))
    print('CHORD_FUSED_BENCH_PASSED',json.dumps(summary),flush=True)


def main():
    start = time.perf_counter()
    source = json.loads(Path('source.json').read_text())
    mode = source['ngpt_velocity_mode']
    learned_beta = mode == 'lookahead_split_learned_beta'
    signed_steps = source.get('signed_chord_steps',False)
    token_gates = source.get('token_chord_gates',False)
    token_beta_gate = source.get('token_beta_gate',False)
    geometry = source.get('chord_velocity_geometry','parallel_transport')
    velocity_init = source['velocity_init']
    token_only = velocity_init == 'token'
    geometry_ablation = geometry != 'parallel_transport' and not token_only
    state_dtype = source.get('chord_state_dtype', 'float32')
    if token_only:
        assert geometry == 'project_only' and learned_beta and state_dtype == 'bfloat16'
    assert (learned_beta or mode in ('lookahead_axis_write','lookahead_split_write','lookahead_geodesic_split')) and source['hopper_only']
    geodesic = mode == 'lookahead_geodesic_split'
    split = mode == 'lookahead_split_write' or geodesic or learned_beta
    gpu = torch.cuda.get_device_name()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9,0)
    assert any(h in gpu for h in ('H100','H200')) and 'MIG' not in gpu
    marker = Path('results/ngpt-sm90.json')
    if marker.exists():
        prior = json.loads(marker.read_text())
        assert prior['complete'] and prior['source'] == source and prior['torch'] == torch.__version__
        return
    if source.get('channel_chord_coefficients'):
        validate_channel_coefficients(source, gpu, marker, start)
        return
    if source.get('fuse_chord_read_norm'):
        validate_fused_branches(source, gpu, marker, start)
        return
    suite = 'tests/test_geodesic_write.py' if geodesic else 'tests/test_split_write.py' if split else 'tests/test_axis_write.py'
    if learned_beta:
        suite = 'tests/test_signed_chord.py' if signed_steps else 'tests/test_split_learned_beta.py'
    if token_gates:
        suite = 'tests/test_token_chord.py'
    if token_beta_gate:
        suite = 'tests/test_token_beta_gate.py'
    if geometry_ablation:
        suite = 'tests/test_chord_velocity_geometry.py'
    if token_only:
        suite = 'tests/test_chord_token_velocity.py'
    env = os.environ.copy()
    if token_only:
        env['CHORD_STORAGE_REPORT'] = 'results/chord-storage.json'
    subprocess.run([sys.executable,'-m','pytest','-q',suite,
                    '--junitxml=results/axis-tests.xml'],check=True,env=env)
    suites = list(ElementTree.parse('results/axis-tests.xml').getroot().iter('testsuite'))
    tests = sum(int(s.get('tests')) for s in suites)
    assert tests == (6 if token_only else 8 if geometry_ablation else 7 if geodesic else 6 if token_gates or token_beta_gate else 4 if split else 5)
    assert all(int(s.get(k,0)) == 0 for s in suites for k in ('errors','failures','skipped'))
    alignment = None
    if not split:
        alignment = measure_reference(source)
        torch.cuda.empty_cache()
    root = Path(tempfile.mkdtemp(prefix='axis-debug-', dir='..')).resolve()
    control_mode = 'lookahead_split_write' if geodesic or learned_beta else 'lookahead_axis_write' if split else 'lookahead_transport'
    if signed_steps or token_gates or token_beta_gate or geometry_ablation or token_only:
        control_mode = mode
    control = profile(root,control_mode,resume=False,measure_turns=geodesic,
                      geometry=geometry if token_only else 'parallel_transport', state_dtype=state_dtype)
    target = profile(root,mode,resume=True,measure_turns=geodesic,signed_steps=signed_steps,
                     token_gates=token_gates,token_beta_gate=token_beta_gate,geometry=geometry,velocity_init=velocity_init,state_dtype=state_dtype)
    precision_control = None
    if token_only:
        precision_control = profile(root,mode,resume=False,geometry=geometry,velocity_init=velocity_init,state_dtype='float32')
        assert {k for k in target['config'] if target['config'][k] != precision_control['config'][k]} == {'chord_state_dtype'}
        assert target['parameters'] == precision_control['parameters']
    root.rmdir()
    changed = {k for k in target['config'] if target['config'][k] != control['config'][k]}
    expected_change = 'ngpt_velocity_mode'
    if token_only:
        expected_change = 'velocity_init'
    elif geometry_ablation:
        expected_change = 'chord_velocity_geometry'
    elif token_beta_gate:
        expected_change = 'token_beta_gate'
    elif token_gates:
        expected_change = 'token_chord_gates'
    elif signed_steps:
        expected_change = 'signed_chord_steps'
    assert changed == {expected_change}
    extra = source['learned_beta_sites'] if learned_beta and not signed_steps else 0 if split else len(alignment['layers'])
    if token_only:
        extra = -source['model']['sequence_length'] * source['model']['width']
    elif geometry_ablation:
        extra = 0
    if token_gates:
        extra = 4*source['model']['depth']
    assert target['parameters'] - control['parameters'] == extra
    if not geodesic:
        assert abs(target['initial_dev_ce'] - control['initial_dev_ce']) < 1e-4
    result = dict(complete=True, formal_training=False, source=source, gpu=gpu, torch=torch.__version__,
                  cuda=torch.version.cuda, tests=tests, resumed_at_step=32, control=control,
                  target=target, step_time_ratio=target['median_step_seconds']/control['median_step_seconds'],
                  evaluation_wall_seconds=time.perf_counter()-start)
    if precision_control is not None:
        result['fp32_token_only'] = precision_control
        result['residual_memory'] = json.loads(Path('results/chord-storage.json').read_text())
    if alignment is not None:
        result['alignment'] = alignment
    json_write(marker,result)
    print('AXIS_WRITE_PREFLIGHT_PASSED',result['step_time_ratio'],flush=True)


if __name__ == '__main__':
    main()
