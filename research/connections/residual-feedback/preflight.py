"""Bounded same-allocation validation, then allow a fresh formal process."""
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
    assert source['feedback_source']=='per_layer_qkv' and type(source['feedback_detach']) is bool
    detach=source['feedback_detach']
    assert source['feedback_bank']=='residual_update' and source['initialization']=='seed42_from_zero_no_diagnostic_weights'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(x in name for x in ('H100','H200')) or cap==[8,0] and 'A100' in name
    family=f'sm{cap[0]}{cap[1]}'
    variant='residual-feedback'+('' if detach else '-full-grad')
    out=Path('results/'+variant+'-'+family);out.mkdir(parents=True,exist_ok=True)
    marker=out/'validation.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        assert previous['complete'] and previous['source']==source and previous['capability']==cap
        print('Reusing completed residual-update validation '+family,flush=True);return
    start=time.monotonic()
    configure_attention('fa2');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    print('RESIDUAL_FEEDBACK_GATE_BEGIN '+name,flush=True)
    tests=[
        'tests/test_latent_feedback.py','tests/test_layer_feedback.py','tests/test_projected_feedback.py',
        'tests/test_feedback_backprop.py','tests/test_residual_feedback.py']
    if not detach:tests.append('tests/test_residual_feedback_backprop.py')
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900',sys.executable,'-m','pytest','-q',
        *tests,'--junitxml='+str(out/'tests.xml')],check=True)
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    count=sum(int(s.attrib['tests']) for s in suites)
    assert count==(170 if detach else 187) and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    # Configuration changes the bank only, not parameter initialization/count.
    models=[]
    for bank in ('hidden','residual_update'):
        seed_all(42)
        c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source='per_layer_qkv',feedback_bank=bank,feedback_detach=detach)
        models.append(build_connection_model(c).prepare())
    left,right=[m.state_dict() for m in models]
    assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
    del models,left,right;gc.collect();torch.cuda.empty_cache()
    data=PackedData('../data');profiles={}
    for bank in ('hidden','residual_update'):
        profiles[bank]=compare_one('per_layer_qkv',detach,data,out,512,bank=bank)
        gc.collect();torch.cuda.empty_cache()
        json_write(out/'profiles.json',profiles)
    old,new=profiles['hidden'],profiles['residual_update']
    ratio=new['median_seconds_by_pass']['2']/old['median_seconds_by_pass']['2']
    assert ratio<1.3,('Unexpected residual-bank overhead; inspect before training',ratio)
    if not detach:
        profiles['residual_update_detached']=compare_one('per_layer_qkv',True,data,out,512,bank='residual_update')
        gc.collect();torch.cuda.empty_cache();json_write(out/'profiles.json',profiles)
    args=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--streams','1',
        '--latent-feedback','--feedback-source','per_layer_qkv','--feedback-detach' if detach else '--no-feedback-detach',
        '--feedback-bank','residual_update',
        '--attention-backend','fa2','--steps','3','--schedule-steps','305176','--microbatch','16','--seed','42',
        '--data-dir','../data','--run-id',variant+'-local-debug','--segment-seconds','1800']
    resumed=Path('../'+variant+'-resume-check');continuous=Path('../'+variant+'-continuous-check')
    assert not resumed.exists() and not continuous.exists(),'Inspect unfinished diagnostics before retrying'
    result=subprocess.run(args+['--output-dir',str(resumed),'--segment-steps','2'])
    assert result.returncode==85,result.returncode
    subprocess.run(args+['--output-dir',str(resumed),'--resume','auto'],check=True)
    subprocess.run(args+['--output-dir',str(continuous)],check=True)
    subprocess.run([sys.executable,'-m','scripts.connection_feedback_resume_check','--resumed',str(resumed),
        '--continuous',str(continuous),'--data-dir','../data','--output',str(out/'resume.json')],check=True)
    resume=json.loads((out/'resume.json').read_text());assert resume['complete']
    shutil.rmtree(resumed);shutil.rmtree(continuous)
    report=dict(complete=True,formal_training=False,source=source,gpu=name,capability=cap,
        torch=torch.__version__,cuda=torch.version.cuda,tests=count,finite_steps_per_model=512,profile_models=len(profiles),feedback_detach=detach,
        identical_initial_parameters=True,parameters=new['parameters'],p2_time_ratio=ratio,
        hidden_p2_ms=old['median_seconds_by_pass']['2']*1000,residual_p2_ms=new['median_seconds_by_pass']['2']*1000,
        residual_mean_step_seconds=new['mean_step_seconds'],residual_peak_bytes=new['peak_allocated_bytes'],
        hidden_dev_p4=old['evaluations'][-1]['ce_p4'],residual_dev_p4=new['evaluations'][-1]['ce_p4'],
        resume_complete=True,elapsed_seconds=time.monotonic()-start)
    if not detach:
        detached=profiles['residual_update_detached']
        report.update(residual_detached_p2_ms=detached['median_seconds_by_pass']['2']*1000,
            residual_detached_dev_p4=detached['evaluations'][-1]['ce_p4'],
            residual_full_grad_to_detached_p2_ratio=new['median_seconds_by_pass']['2']/detached['median_seconds_by_pass']['2'])
    json_write(marker,report)
    print('RESIDUAL_FEEDBACK_GATE_PASSED '+json.dumps(report),flush=True)


if __name__=='__main__':main()
