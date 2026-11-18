"""Same-allocation gate for nonzero-init residual configurations; diagnostics never create W&B."""
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import torch


def main():
    begin = time.perf_counter()
    source = json.loads(Path('source.json').read_text())
    mode = source['residual_dynamics']
    assert source['formal_training'] and mode in ('vanilla','ngpt','ngpt_heavy_ball','heavy_ball_rms','ngpt_hbformer','ngpt_lookahead')
    assert source['residual_projection_init']=='normal' and source['ngpt_force_rms_floor']==1e-6
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    cap = list(torch.cuda.get_device_capability())
    name = torch.cuda.get_device_name()
    assert (cap == [9,0] and any(n in name for n in ('H100','H200'))) or (cap == [8,0] and 'A100' in name)
    device = dict(gpu=name, capability=cap, torch=torch.__version__, cuda=torch.version.cuda)
    Path('results/device.json').write_text(json.dumps(device, indent=2)+'\n')
    marker = Path(f'results/ngpt-sm{cap[0]}{cap[1]}.json')
    if marker.exists():
        result = json.loads(marker.read_text())
        assert result['complete'] and result['source'] == source and result['capability'] == cap
        print('NGPT_PREFLIGHT_PASSED reused exact source/family validation', flush=True)
        return
    transported = mode=='ngpt_heavy_ball' and source.get('ngpt_velocity_mode')=='transport_rms'
    reused = False
    if transported:
        assert mode=='ngpt_heavy_ball'
        report=json.loads(Path('../transport-validation.json').read_text())
        assert report['complete'] and report['job']==source['validation_job']
        assert report['source']['commit']==source['validated_model_commit']
        profile,tests,resume=report['profile'],report['tests'],report['resume']
        reused=(cap==[9,0] and profile['gpu']=='NVIDIA H200' and
                profile['torch']==torch.__version__ and profile['cuda']==torch.version.cuda)
    if not reused:
        check=(['research/connections/momentum/check.sh'] if transported else
               ['research/connections/ngpt/check.sh',mode])
        if mode == 'ngpt_hbformer' and source.get('ngpt_weight_norm') is False:
            check.append('--no-ngpt-weight-norm')
        if source.get('ngpt_velocity_mode') == 'transport_layernorm':
            check.append('transport_layernorm')
        subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400','bash',*check],check=True)
        prefix='momentum' if transported else 'ngpt'
        suites=list(ET.parse(f'results/{prefix}-tests.xml').getroot().iter('testsuite'))
        tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        profile=json.loads(Path(f'results/{prefix}-profile/summary.json').read_text())
        resume=json.loads(Path(f'results/{prefix}-cli-resume.json').read_text())
    expected_tests = {'heavy_ball_rms':9, 'ngpt_hbformer':8, 'ngpt_lookahead':8}.get(mode, 57 if transported else 29)
    if source.get('ngpt_velocity_mode') == 'transport_layernorm':
        expected_tests = 9
        control = profile['models']['hbformer_rms']['config']
        target = profile['models'][mode]['config']
        assert control['ngpt_velocity_mode'] == 'transport_rms'
        assert {k for k in target if target[k] != control[k]} == {'ngpt_velocity_mode'}
    if mode == 'ngpt_lookahead':
        target = profile['models'][mode]
        for key in ('ngpt_velocity_mode', 'ngpt_weight_norm', 'lookahead_mu_init', 'velocity_init', 'velocity_embedding_std', 'momentum_beta_init'):
            assert target['config'][key] == source[key], key
        assert all(row['lookahead_displacement_rms'] == 0 for row in target['initial_residual_state'])
        assert any(row['lookahead_displacement_rms'] > 0 for row in target['final_residual_state'])
    if mode == 'ngpt_hbformer':
        assert profile['models'][mode]['config']['ngpt_weight_norm'] == source.get('ngpt_weight_norm',True)
    assert tests == dict(tests=expected_tests, failures=0, errors=0, skipped=0), tests
    assert profile['complete'] and profile['models'][mode]['steps'] == 128 and resume['complete']
    if transported or mode in ('ngpt_hbformer', 'heavy_ball_rms'):
        c=profile['models'][mode]['config']
        for key in ('ngpt_velocity_mode','momentum_velocity_rms_floor','momentum_eta_init','velocity_init',
                    'momentum_beta_init','momentum_gamma_init','velocity_embedding_std'):
            assert c[key]==source[key],key
    for name, info in profile['models'].items():
        assert info['config']['residual_projection_init']=='normal'
        assert info['config']['ngpt_force_rms_floor']==1e-6
        if name == 'heavy_ball_rms':
            assert info['config']['residual_dynamics']==name
            assert all(row['velocity_rms_max'] < 1+2e-6 for row in info['final_residual_state'])
        elif name != 'vanilla':
            assert all(row['force_floor_fraction']==0 for row in info['initial_residual_state'])
    assert not Path('../ngpt-resume-check').exists() and not Path('../ngpt-continuous-check').exists()
    result = dict(complete=True, formal_training=False, source=source, capability=cap,
                  device=device, tests=tests, profile=profile, resume=resume,
                  reused_prior_validation=reused,evaluation_wall_seconds=time.perf_counter()-begin)
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print('NGPT_PREFLIGHT_PASSED fresh formal seed42 process follows', flush=True)


if __name__ == '__main__':
    main()
