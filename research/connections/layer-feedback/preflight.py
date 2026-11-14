"""Validate this source and GPU family before the authorized same-allocation run."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import torch


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['formal_training'] and source['model_variant']=='layer_feedback'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(n in name for n in ('H100','H200')) or cap==[8,0] and 'A100' in name
    device=dict(gpu=name,capability=cap,torch=torch.__version__,cuda=torch.version.cuda)
    Path('results/device.json').write_text(json.dumps(device,indent=2)+'\n')
    marker=Path('results')/('layer-feedback-sm'+''.join(map(str,cap))+'.json')
    if marker.exists():
        result=json.loads(marker.read_text())
        assert result['complete'] and result['source']==source and result['capability']==cap
        print('Reusing passed layer-feedback checks for this source and family',flush=True);return
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400','bash',
        'research/connections/layer-feedback/check.sh'],check=True)
    counts={}
    for name,expected in (('top-regression',12),('layer-tests',27)):
        suite=ET.parse('results/'+name+'.xml').getroot().find('testsuite')
        assert int(suite.attrib['tests'])==expected
        assert all(int(suite.attrib.get(k,0))==0 for k in ('failures','errors','skipped'))
        counts[name]=expected
    profile=json.loads(Path('results/layer-profile/summary.json').read_text())
    restore=json.loads(Path('results/layer-cli-resume.json').read_text())
    assert profile['complete'] and restore['complete'] and profile['steps_per_model']==96
    assert profile['ratio_to_top']<1.35 and profile['ratio_to_reference']<1.20
    assert not Path('../layer-resume-check').exists() and not Path('../layer-continuous-check').exists()
    result=dict(complete=True,formal_training=False,source=source,capability=cap,device=device,
        tests_passed=counts,profile=profile,resume=restore)
    marker.write_text(json.dumps(result,indent=2)+'\n')
    print('LAYER_FEEDBACK_VALIDATION_PASSED; formal process will initialize independently',flush=True)


if __name__=='__main__':main()
