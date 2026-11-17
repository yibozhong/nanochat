"""Same-allocation gate for pure nGPT residual; diagnostics never create W&B."""
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import torch


def main():
    begin = time.perf_counter()
    source = json.loads(Path('source.json').read_text())
    assert source['formal_training'] and source['residual_dynamics'] == 'ngpt'
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
                    'research/connections/ngpt/check.sh'], check=True)
    suites = list(ET.parse('results/ngpt-tests.xml').getroot().iter('testsuite'))
    tests = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert tests == dict(tests=66, failures=0, errors=0, skipped=0), tests
    profile = json.loads(Path('results/ngpt-profile/summary.json').read_text())
    resume = json.loads(Path('results/ngpt-cli-resume.json').read_text())
    assert profile['complete'] and profile['models']['ngpt']['steps'] == 128 and resume['complete']
    assert not Path('../ngpt-resume-check').exists() and not Path('../ngpt-continuous-check').exists()
    result = dict(complete=True, formal_training=False, source=source, capability=cap,
                  device=device, tests=tests, profile=profile, resume=resume,
                  evaluation_wall_seconds=time.perf_counter()-begin)
    marker.write_text(json.dumps(result, indent=2)+'\n')
    print('NGPT_PREFLIGHT_PASSED fresh formal seed42 process follows', flush=True)


if __name__ == '__main__':
    main()
