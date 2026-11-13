"""Reuse Hopper results; gate the compatible A100 backend in its allocation."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import torch


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['formal_training'] and source['pass_mix']==[.15,.85]
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    device=dict(gpu=name,capability=cap,torch=torch.__version__,cuda=torch.version.cuda)
    Path('results/device.json').write_text(json.dumps(device,indent=2)+'\n')
    if cap==[9,0] and any(n in name for n in ('H100','H200')):
        gate=json.loads(Path('research/connections/latent-feedback/validation.json').read_text())
        assert gate['complete'] and gate['gpu_tests_passed']==12 and gate['exact_checkpoint_transport_and_fixed_gradient_update']
        print('Reusing completed Hopper feedback validation',flush=True);return
    assert cap==[8,0] and 'A100' in name,device
    marker=Path('results/feedback-sm80.json')
    if marker.exists():
        v=json.loads(marker.read_text())
        assert v['complete'] and v['source']==source and v['capability']==cap
        print('Reusing successful A100 feedback validation for this run',flush=True);return
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400','bash',
        'research/connections/latent-feedback/check.sh'],check=True)
    suite=ET.parse('results/tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==12
    assert all(int(suite.attrib.get(k,0))==0 for k in ('failures','errors','skipped'))
    profile=json.loads(Path('results/profile/summary.json').read_text())
    restore=json.loads(Path('results/cli-resume.json').read_text())
    assert profile['complete'] and restore['complete'] and profile['steps_per_model']==96
    assert not Path('../resume-check').exists() and not Path('../continuous-check').exists()
    v=dict(complete=True,formal_training=False,source=source,capability=cap,device=device,
        tests_passed=12,profile=profile,resume=restore)
    marker.write_text(json.dumps(v,indent=2)+'\n')
    print('A100 feedback checks passed; formal process will initialize independently',flush=True)


if __name__=='__main__':main()
