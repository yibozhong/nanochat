"""Same-allocation history-feedback gate; no diagnostic W&B or reused weights."""
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


def production_profile():
    seed_all(42)
    c=ConnectionConfig(streams=1,latent_feedback=True,feedback_source='history_qkv',feedback_attention_backend='flash')
    m=build_connection_model(c).prepare(compile_blocks=True)
    e=torch.randn(16,2048,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    top=torch.randn_like(e);go=torch.randn_like(e)
    def run(fn,check=False):
        m.zero_grad(set_to_none=True);e.grad=None
        value=fn(e,top);value.backward(go)
        if check:
            assert torch.isfinite(value).all() and torch.isfinite(e.grad).all()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.feedback_parameters())
    timing={}
    for label,fn in (('eager',m.mix),('compiled',m.compiled_mix)):
        for _ in range(5):run(fn)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(30):run(fn)
        torch.cuda.synchronize()
        timing[label]=dict(mean_seconds=(time.perf_counter()-start)/30,peak_bytes=torch.cuda.max_memory_allocated())
        run(fn,check=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        run(m.compiled_mix);torch.cuda.synchronize()
    names=sorted({evt.key for evt in prof.key_averages()})
    flash=[n for n in names if 'flash' in n.lower()]
    assert any('backward' in n.lower() or 'bwd' in n.lower() for n in flash),flash
    assert not any('attention_math' in n for n in names)
    # A dense [16,4,2047,2047] FP32 score matrix alone is ~1 GiB.
    assert timing['compiled']['peak_bytes']<2*2**30,timing
    assert timing['compiled']['mean_seconds']<1.3*timing['eager']['mean_seconds'],timing
    return dict(shape=[16,2048,512],heads=4,backend='pytorch FlashAttention',
        packed_kv_gemm=True,training_mask_materialized=False,flash_ops=flash,timing=timing)


def main():
    source=json.loads(Path('source.json').read_text())
    assert source['feedback_source']=='history_qkv' and source['feedback_detach'] is True
    assert source['feedback_attention_backend']=='flash' and source['feedback_bank']=='hidden'
    assert source['initialization']=='seed42_from_zero_no_diagnostic_weights'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    cap=list(torch.cuda.get_device_capability());name=torch.cuda.get_device_name()
    assert cap==[9,0] and any(s in name for s in ('H100','H200')) or cap==[8,0] and 'A100' in name
    family=f'sm{cap[0]}{cap[1]}'
    out=Path('results/history-feedback-'+family);out.mkdir(parents=True,exist_ok=True);marker=out/'validation.json'
    if marker.exists():
        report=json.loads(marker.read_text());assert report['complete'] and report['source']==source and report['capability']==cap
        print('HISTORY_FEEDBACK_GATE_PASSED '+json.dumps(report),flush=True);return
    start=time.monotonic();configure_attention('fa2');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch._dynamo.config.recompile_limit=max(32,torch._dynamo.config.recompile_limit)
    print('HISTORY_FEEDBACK_GATE_BEGIN '+name,flush=True)
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900',sys.executable,'-m','pytest','-q',
        'tests/test_latent_feedback.py','tests/test_history_feedback.py','--junitxml='+str(out/'tests.xml')],check=True)
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'));count=sum(int(s.attrib['tests']) for s in suites)
    assert count==41 and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    operator=production_profile();json_write(out/'operator-profile.json',operator)
    gc.collect();torch.cuda.empty_cache()
    data=PackedData('../data');profiles={}
    for mode in ('top','history_qkv'):
        profiles[mode]=compare_one(mode,True,data,out,512)
        gc.collect();torch.cuda.empty_cache();json_write(out/'profiles.json',profiles)
    old,new=profiles['top'],profiles['history_qkv']
    ratio=new['median_seconds_by_pass']['2']/old['median_seconds_by_pass']['2']
    assert ratio<1.35,('Inspect unexpected history attention overhead before training',ratio)
    args=[sys.executable,'-m','scripts.connection_train','--variant','baseline','--streams','1',
        '--latent-feedback','--feedback-source','history_qkv','--feedback-detach','--feedback-attention-backend','flash',
        '--attention-backend','fa2','--steps','3','--schedule-steps','305176','--microbatch','16','--seed','42',
        '--data-dir','../data','--run-id','history-feedback-local-debug','--segment-seconds','1800']
    resumed=Path('../history-feedback-resume-check');continuous=Path('../history-feedback-continuous-check')
    assert not resumed.exists() and not continuous.exists(),'Inspect unfinished diagnostics before retrying'
    first=subprocess.run(args+['--output-dir',str(resumed),'--segment-steps','2']);assert first.returncode==85,first.returncode
    subprocess.run(args+['--output-dir',str(resumed),'--resume','auto'],check=True)
    subprocess.run(args+['--output-dir',str(continuous)],check=True)
    subprocess.run([sys.executable,'-m','scripts.connection_feedback_resume_check','--resumed',str(resumed),
        '--continuous',str(continuous),'--data-dir','../data','--output',str(out/'resume.json')],check=True)
    assert json.loads((out/'resume.json').read_text())['complete']
    shutil.rmtree(resumed);shutil.rmtree(continuous)
    report=dict(complete=True,formal_training=False,source=source,gpu=name,capability=cap,
        torch=torch.__version__,cuda=torch.version.cuda,tests=count,finite_steps_per_model=512,
        parameters=new['parameters'],p2_time_ratio=ratio,
        top_p2_ms=old['median_seconds_by_pass']['2']*1000,history_p2_ms=new['median_seconds_by_pass']['2']*1000,
        mean_step_seconds=new['mean_step_seconds'],peak_bytes=new['peak_allocated_bytes'],
        top_dev_p4=old['evaluations'][-1]['ce_p4'],history_dev_p4=new['evaluations'][-1]['ce_p4'],
        cache='fixed pass count: full sequence / chunks / single tokens agree',
        operator_profile=operator,resume_complete=True,elapsed_seconds=time.monotonic()-start)
    json_write(marker,report);print('HISTORY_FEEDBACK_GATE_PASSED '+json.dumps(report),flush=True)


if __name__=='__main__':main()
