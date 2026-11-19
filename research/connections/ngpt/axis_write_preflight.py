"""Same-GPU alignment measurement and paired checks before the fresh 10B run."""
from functools import partial
import json
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


def profile(root, mode, resume):
    steps = 128
    out = root/mode
    cmd = [sys.executable, '-m', 'scripts.connection_train', '--variant', 'baseline', '--streams', '1',
           '--residual-dynamics','ngpt_lookahead','--ngpt-velocity-mode',mode,
           '--velocity-init','token_position','--velocity-embedding-std','.02','--momentum-beta-init','.5',
           '--lookahead-mu-init','0','--ngpt-alpha-init','.05','--ngpt-force-rms-floor','1e-6',
           '--residual-projection-init','normal','--residual-projection-init-std','.02',
           '--no-activation-checkpoint','--no-fused-connection-grad','--attention-backend','fa2',
           '--steps',str(steps),'--schedule-steps','305176','--microbatch','16','--seed','42',
           '--data-dir','../data','--output-dir',str(out),'--run-id','axis-local-debug']
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
    if mode == 'lookahead_split_write':
        init, = [e for e in events if e.get('step') == 0 and 'dev/ce' in e]
        for suffix in ('alpha_parallel','alpha_perp'):
            rates = {k:v for k,v in end.items() if k.endswith('/'+suffix)}
            assert len(rates) == 2*cfg['model']['depth'] and all(math.isfinite(v) and v>0 for v in rates.values())
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
    print('AXIS_WRITE_CONTROL',mode,json.dumps(result),flush=True)
    shutil.rmtree(out)
    return result


def main():
    start = time.perf_counter()
    source = json.loads(Path('source.json').read_text())
    mode = source['ngpt_velocity_mode']
    assert mode in ('lookahead_axis_write','lookahead_split_write') and source['hopper_only']
    split = mode == 'lookahead_split_write'
    gpu = torch.cuda.get_device_name()
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9,0)
    assert any(h in gpu for h in ('H100','H200')) and 'MIG' not in gpu
    marker = Path('results/ngpt-sm90.json')
    if marker.exists():
        prior = json.loads(marker.read_text())
        assert prior['complete'] and prior['source'] == source and prior['torch'] == torch.__version__
        return
    suite = 'tests/test_split_write.py' if split else 'tests/test_axis_write.py'
    subprocess.run([sys.executable,'-m','pytest','-q',suite,
                    '--junitxml=results/axis-tests.xml'],check=True)
    suites = list(ElementTree.parse('results/axis-tests.xml').getroot().iter('testsuite'))
    tests = sum(int(s.get('tests')) for s in suites)
    assert tests == (4 if split else 5)
    assert all(int(s.get(k,0)) == 0 for s in suites for k in ('errors','failures','skipped'))
    alignment = None
    if not split:
        alignment = measure_reference(source)
        torch.cuda.empty_cache()
    root = Path(tempfile.mkdtemp(prefix='axis-debug-', dir='..')).resolve()
    control = profile(root,'lookahead_axis_write' if split else 'lookahead_transport',resume=False)
    target = profile(root,mode,resume=True)
    root.rmdir()
    assert {k for k in target['config'] if target['config'][k] != control['config'][k]} == {'ngpt_velocity_mode'}
    assert target['parameters'] - control['parameters'] == (0 if split else len(alignment['layers']))
    assert abs(target['initial_dev_ce'] - control['initial_dev_ce']) < 1e-4
    result = dict(complete=True, formal_training=False, source=source, gpu=gpu, torch=torch.__version__,
                  cuda=torch.version.cuda, tests=tests, resumed_at_step=32, control=control,
                  target=target, step_time_ratio=target['median_step_seconds']/control['median_step_seconds'],
                  evaluation_wall_seconds=time.perf_counter()-start)
    if alignment is not None:
        result['alignment'] = alignment
    json_write(marker,result)
    print('AXIS_WRITE_PREFLIGHT_PASSED',result['step_time_ratio'],flush=True)


if __name__ == '__main__':
    main()
