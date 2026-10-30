"""Same-GPU audit of NVIDIA TE projection vs existing implementations."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

import torch
import wandb

from nanochat.manifold_connections import projection


GROUP='mhc-nvidia-projection-20261030'
TRIALS={'baseline':('baseline','official'),'mhc_official':('mhc','official'),
        'mhc_torch_compile':('mhc','torch_compile'),
        'mhc_nvidia':('mhc','nvidia'),'mhc_nvidia_precise':('mhc','nvidia_precise')}


def projection_benchmark(backends=('official','torch_compile','nvidia','nvidia_precise')):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    x=torch.randn(16,2048,4,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=(torch.randn(24,2048,device='cuda')/2048**.5).requires_grad_()
    probe=torch.randn(16,2048,24,device='cuda')/32768
    result={}
    for backend in backends:
        print('Projection microbenchmark '+backend,flush=True)
        for _ in range(10):
            x.grad=None;w.grad=None
            (projection(x,w,backend)*probe).sum().backward()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(30):
            x.grad=None;w.grad=None
            begin,mid,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            begin.record();y=projection(x,w,backend);loss=(y*probe).sum();mid.record()
            loss.backward();end.record();end.synchronize()
            samples.append(dict(forward_ms=begin.elapsed_time(mid),backward_ms=mid.elapsed_time(end),
                                total_ms=begin.elapsed_time(end)))
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        result[backend]=dict(samples=samples,means={k:statistics.mean(s[k] for s in samples) for k in samples[0]},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return dict(shape=list(x.shape),weight_shape=list(w.shape),backends=result,
        note='CUDA event intervals include launch gaps and scalar probe. PyTorch matmul TF32 disabled; NVIDIA_TF32_OVERRIDE=0 disables cuBLAS TF32 in all trials. nvidia uses TE explicit Triton TF32 and nvidia_precise uses TF32x3 for BF16 x FP32. Other mHC ops unchanged.')


def main():
    out=Path('results');source=json.loads(Path('source.json').read_text())
    run_id=uuid.uuid4().hex[:8]
    kwargs=dict(project='nanochat',entity='yibozhong657-none',group=GROUP,name='nvidia-projection-audit',
                job_type='correctness-and-profiling',id=run_id,config=dict(source=source,trials=TRIALS))
    try:run=wandb.init(**kwargs,settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:run=wandb.init(**kwargs,mode='offline')
    try:
        print('Checking NVIDIA, PyTorch and existing official paths',flush=True)
        with (out/'pytest.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','tests/test_mhc.py','tests/test_mhc_torch_projection.py','tests/test_mhc_nvidia_projection.py',
                '-q','--junitxml=results/tests.xml'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        suites=ET.parse(out/'tests.xml').getroot().findall('testsuite')
        checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        assert checks['failures']==checks['errors']==checks['skipped']==0,checks
        print('Checks passed '+json.dumps(checks),flush=True)
        from nanochat.nvidia_projection import nvidia_ops
        nvidia_ops()  # Verify exact upstream source hashes in the profiling process too.
        micro=projection_benchmark()
        (out/'projection-microbenchmark.json').write_text(json.dumps(micro,indent=2)+'\n')
        # Drop parent CUDA allocations/cache before child end-to-end profiles.
        torch.cuda.empty_cache()
        manifest=json.loads(Path('../data/manifest.json').read_text())
        for name,info in manifest['files'].items():
            digest=hashlib.sha256()
            with Path('../data',name+'.bin').open('rb') as f:
                while chunk:=f.read(8*1024*1024):digest.update(chunk)
            assert digest.hexdigest()==info['sha256'],name
        trials={}
        for name,(variant,backend) in TRIALS.items():
            print('Profiling '+name,flush=True)
            with (out/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,'-m','scripts.connection_profile','--variant',variant,
                    '--projection-backend',backend,'--wandb-group',GROUP,'--output-dir',str(out/name)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
            summary=json.loads((out/name/'summary.json').read_text())
            assert summary['complete'] and summary['source']==source and summary['optimizer_updates']==66
            assert summary['projection_backend']==backend
            trials[name]=summary
            run.log({name+'/normal_step_ms':summary['normal_ms']['mean']})
        result=dict(complete=True,source=source,run_id=run_id,checks=checks,microbenchmark=micro,
                    trials=trials,data_manifest=manifest,pretraining_launched=False)
        (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update(dict(complete=True,checks_passed=True,trials=len(TRIALS),total_diagnostic_tokens=len(TRIALS)*66*32768))
        run.save('results/*.json',base_path='.',policy='now');run.finish()
        print('Comparison complete',flush=True)
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':main()
