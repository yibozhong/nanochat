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
    assert source['formal_training'] and mode in ('vanilla','ngpt','ngpt_heavy_ball')
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
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400','bash',
                    'research/connections/ngpt/check.sh',mode], check=True)
    suites = list(ET.parse('results/ngpt-tests.xml').getroot().iter('testsuite'))
    tests = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert tests == dict(tests=29, failures=0, errors=0, skipped=0), tests
    profile = json.loads(Path('results/ngpt-profile/summary.json').read_text())
    resume = json.loads(Path('results/ngpt-cli-resume.json').read_text())
    assert profile['complete'] and profile['models'][mode]['steps'] == 128 and resume['complete']
    for name, info in profile['models'].items():
        assert info['config']['residual_projection_init']=='normal'
        assert info['config']['ngpt_force_rms_floor']==1e-6
        if name != 'vanilla':
            assert all(row['force_floor_fraction']==0 for row in info['initial_residual_state'])
    assert not Path('../ngpt-resume-check').exists() and not Path('../ngpt-continuous-check').exists()
    result = dict(complete=True, formal_training=False, source=source, capability=cap,
                  device=device, tests=tests, profile=profile, resume=resume,
                  evaluation_wall_seconds=time.perf_counter()-begin)
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print('NGPT_PREFLIGHT_PASSED fresh formal seed42 process follows', flush=True)


if __name__ == '__main__':
    main()
