"""Source/GPU-family gate before the single authorized, independently reset run."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import torch


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['formal_training'] and source['model_variant']=='riemannian_momentum'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(n in name for n in ('H100','H200')) or cap==[8,0] and 'A100' in name
    device=dict(gpu=name,capability=cap,torch=torch.__version__,cuda=torch.version.cuda)
    Path('results/device.json').write_text(json.dumps(device,indent=2)+'\n')
    marker=Path('results')/('rmt-sm'+''.join(map(str,cap))+'.json')
    if marker.exists():
        result=json.loads(marker.read_text())
        assert result['complete'] and result['source']==source and result['capability']==cap
        print('Reusing passed RMT checks for this source and GPU family',flush=True);return
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','3600','bash',
        'research/connections/riemannian-momentum/check.sh'],check=True)
    suite=ET.parse('results/rmt-tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==36,suite.attrib
    assert all(int(suite.attrib.get(k,0))==0 for k in ('failures','errors','skipped'))
    profile=json.loads(Path('results/rmt-profile/summary.json').read_text())
    restore=json.loads(Path('results/rmt-cli-resume.json').read_text())
    assert profile['complete'] and restore['complete'] and profile['stability_steps']==512
    assert profile['time_ratio']<1.75
    assert not Path('../rmt-resume-check').exists() and not Path('../rmt-continuous-check').exists()
    result=dict(complete=True,formal_training=False,source=source,capability=cap,device=device,
        tests_passed=36,profile=profile,resume=restore)
    marker.write_text(json.dumps(result,indent=2)+'\n')
    print('RMT_PREFLIGHT_PASSED; formal training initializes independently from seed42',flush=True)


if __name__=='__main__':main()
