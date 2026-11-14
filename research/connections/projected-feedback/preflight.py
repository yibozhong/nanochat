"""Reuse completed Hopper validation; validate the compatible A100 in-place."""
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model,optimizer_for,schedule
from nanochat.packed_training import PackedData,evaluate,json_write,seed_all,train_step


def main():
    source=json.loads(Path('source.json').read_text());mode=source['feedback_source']
    assert source['formal_training'] and source['feedback_projection_order']=='reassociated'
    assert mode in ('embedding_qkv','per_layer_qkv')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    device=dict(gpu=name,capability=cap,torch=torch.__version__,cuda=torch.version.cuda)
    json_write('results/device.json',device)
    if cap==[9,0] and any(n in name for n in ('H100','H200')):
        gate=json.loads(Path('research/connections/projected-feedback/validation.json').read_text())
        assert gate['complete'] and gate['gpu_tests_passed']==97
        assert gate['model_commit']==source['validated_model_commit']
        assert gate['variants'][mode]['resume_complete'] and gate['variants'][mode]['efficiency_review_passed']
        print('Reusing completed Hopper projected-feedback validation',flush=True);return
    assert cap==[8,0] and 'A100' in name,device
    marker=Path('results/projected-feedback-sm80.json')
    if marker.exists():
        v=json.loads(marker.read_text());assert v['complete'] and v['source']==source
        print('Reusing completed A100 projected-feedback validation',flush=True);return
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900','python','-m','pytest','-q',
        'tests/test_projected_feedback.py','--junitxml=results/projected-sm80-tests.xml'],check=True)
    suite=ET.parse('results/projected-sm80-tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==58
    assert all(int(suite.attrib.get(k,0))==0 for k in ('failures','errors','skipped'))
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    seed_all(42);data=PackedData('../data')
    c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source=mode,feedback_projection_order='reassociated')
    model=build_connection_model(c).prepare(compile_blocks=True);opt,wd=optimizer_for(model)
    start=time.monotonic();losses=[]
    for step in range(16):
        schedule(opt,step,305176,wd);losses.append(train_step(model,opt,data,step,microbatch=16,seed=42))
        assert time.monotonic()-start<600,'A100 finite-training check exceeded its bound'
    p1=evaluate(model,data,passes=1)[0];p4=evaluate(model,data,passes=4)[0]
    json_write(marker,dict(complete=True,formal_training=False,source=source,device=device,
        gpu_tests_passed=58,finite_steps=16,losses=losses,dev_ce_p1=p1,dev_ce_p4=p4))
    print('A100 validation passed; formal process initializes independently',flush=True)


if __name__=='__main__':main()
