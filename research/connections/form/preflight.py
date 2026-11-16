"""Bounded verification, then the same allocation starts the authorized run."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import torch


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['formal_training'] and source['model_variant']=='form'
    assert source['form_count']==10 and source['form_minimum_width']==512
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(n in name for n in ('H100','H200')) or cap==[8,0] and 'A100' in name
    device=dict(gpu=name,capability=cap,torch=torch.__version__,cuda=torch.version.cuda)
    Path('results/device.json').write_text(json.dumps(device,indent=2)+'\n')
    marker=Path('results')/('form-sm'+''.join(map(str,cap))+'.json')
    if marker.exists():
        value=json.loads(marker.read_text())
        assert value['complete'] and value['source']==source and value['capability']==cap
        print('Reusing passed Form checks for this source and GPU family',flush=True);return
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400','bash',
        'research/connections/form/check.sh'],check=True)
    suites=ET.parse('results/form-tests.xml').getroot().findall('testsuite')
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==38, count
    assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    profile=json.loads(Path('results/form-profile/summary.json').read_text())
    resume=json.loads(Path('results/form-cli-resume.json').read_text())
    assert profile['complete'] and profile['steps_per_model']==256 and profile['step_time_ratio']<2.
    assert len(profile['events'])==2 and resume['complete'] and resume['form_completed']==1
    assert not Path('../form-resume-check').exists()
    value=dict(complete=True,formal_training=False,source=source,capability=cap,device=device,
        tests_passed=count,profile=profile,resume=resume)
    marker.write_text(json.dumps(value,indent=2)+'\n')
    print('FORM_PREFLIGHT_PASSED; formal training initializes independently from seed42',flush=True)


if __name__=='__main__':main()
