"""Validate top-only full-gradient feedback in the formal GPU allocation."""
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from nanochat.connection_experiment import configure_attention
from nanochat.connection_model import ConnectionConfig,build_connection_model
from nanochat.packed_training import PackedData,json_write,seed_all
from scripts.connection_feedback_backprop_debug import compare_one


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['feedback_source']=='top' and source['feedback_detach'] is False
    assert source['feedback_bank']=='hidden' and source['loss']=='final-pass CE only'
    assert source['initialization']=='seed42_from_zero_no_diagnostic_weights'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(x in name for x in ('H100','H200')) or cap==[8,0] and 'A100' in name
    family=f'sm{cap[0]}{cap[1]}'
    out=Path('results/top-feedback-backprop-'+family);out.mkdir(parents=True,exist_ok=True)
    marker=out/'validation.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        assert previous['complete'] and previous['source']==source and previous['capability']==cap
        print('Reusing completed top full-gradient validation '+family,flush=True);return
    start=time.monotonic()
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    print('TOP_FEEDBACK_GATE_BEGIN '+name,flush=True)
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900',sys.executable,'-m','pytest','-q',
        'tests/test_latent_feedback.py','tests/test_layer_feedback.py','tests/test_top_feedback_backprop.py',
        '--junitxml='+str(out/'tests.xml')],check=True)
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==53 and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    models=[]
    for detach in (True,False):
        seed_all(42)
        c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source='top',feedback_detach=detach)
        models.append(build_connection_model(c).prepare())
    left,right=[m.state_dict() for m in models]
    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    del models,left,right;gc.collect();torch.cuda.empty_cache()
    data=PackedData('../data');profiles={}
    for detach in (True,False):
        profiles['detach' if detach else 'full_grad']=compare_one('top',detach,data,out,512)
        gc.collect();torch.cuda.empty_cache();json_write(out/'profiles.json',profiles)
    old,new=profiles['detach'],profiles['full_grad']
    args=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--streams','1',
        '--latent-feedback','--feedback-source','top','--no-feedback-detach',
        '--attention-backend','fa2','--steps','3','--schedule-steps','305176','--microbatch','16','--seed','42',
        '--data-dir','../data','--run-id','top-full-grad-local-debug','--segment-seconds','1800']
    resumed=Path('../top-full-grad-resume-check');continuous=Path('../top-full-grad-continuous-check')
    assert not resumed.exists() and not continuous.exists(),'Inspect unfinished diagnostics before retrying'
    first=subprocess.run(args+['--output-dir',str(resumed),'--segment-steps','2'])
    assert first.returncode==85,first.returncode
    subprocess.run(args+['--output-dir',str(resumed),'--resume','auto'],check=True)
    subprocess.run(args+['--output-dir',str(continuous)],check=True)
    subprocess.run([sys.executable,'-m','scripts.connection_feedback_resume_check','--resumed',str(resumed),
        '--continuous',str(continuous),'--data-dir','../data','--output',str(out/'resume.json')],check=True)
    assert json.loads((out/'resume.json').read_text())['complete']
    shutil.rmtree(resumed);shutil.rmtree(continuous)
    report=dict(complete=True,formal_training=False,source=source,gpu=name,capability=cap,
        torch=torch.__version__,cuda=torch.version.cuda,tests=count,finite_steps_per_model=512,
        identical_initial_parameters=True,parameters=new['parameters'],
        p2_time_ratio=new['median_seconds_by_pass']['2']/old['median_seconds_by_pass']['2'],
        detached_p2_ms=old['median_seconds_by_pass']['2']*1000,full_grad_p2_ms=new['median_seconds_by_pass']['2']*1000,
        mean_step_seconds=new['mean_step_seconds'],peak_bytes=new['peak_allocated_bytes'],
        detached_dev_p4=old['evaluations'][-1]['ce_p4'],full_grad_dev_p4=new['evaluations'][-1]['ce_p4'],
        resume_complete=True,elapsed_seconds=time.monotonic()-start)
    json_write(marker,report)
    print('TOP_FEEDBACK_GATE_PASSED '+json.dumps(report),flush=True)


if __name__=='__main__':main()
