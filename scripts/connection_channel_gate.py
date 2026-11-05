"""Local tests, same-device benchmarks and fresh-process resume; no W&B."""
import json,math,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET

VARIANT='mhc_channel_no_sinkhorn'

def main():
    root=Path('results/channel-mixing');root.mkdir(parents=True,exist_ok=True)
    def run(args,name,expected=0):
        print('BEGIN '+name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        assert proc.returncode==expected,(name,proc.returncode)
        print('PASS '+name,flush=True)
    run(['-m','pytest','-v','tests/test_channel_mixing.py','tests/test_connection_training_resume.py',
         'tests/test_mhc_no_sinkhorn.py::test_no_sinkhorn_pipeline_fp64[False-512]',
         'tests/test_native_connections.py::test_full_size_backbone_initialization_and_optimizer_groups',
         'tests/test_native_connections.py::test_real_kernels_causal_swa_gqa_cache_and_feature_gradients[mhc]',
         '--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=16,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_channel_micro'],'micro')
    profiles={}
    for v in ('baseline','mhc','mhc_no_sinkhorn',VARIANT):
        run(['-m','scripts.connection_context_profile','--variant',v,'--sequence-length','2048',
             '--microbatch','16','--attention-backend','fa3','--output',str(root/(v+'.json'))],'profile-'+v)
        p=json.loads((root/(v+'.json')).read_text());profiles[v]=p
        assert p['complete'] and p['status']=='fits' and p['causality_passed'],p
        assert p['optimizer_updates']==60 and p['peak_reserved_bytes']<int(.9*22000*1024**2)
    assert len({p['gpu_uuid'] for p in profiles.values()})==1
    assert len({p['input_sha256'] for p in profiles.values()})==1
    args=['-m','scripts.connection_train','--variant',VARIANT,'--matched-backbone-init',
          '--steps','128','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
          '--attention-backend','fa3','--output-dir',str(root/'restart'),
          '--checkpoint-dir','../channel-gate-checkpoints','--run-id','channel-mixing-restart',
          '--segment-seconds','3600','--segment-steps','25']
    run(args,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text())
    config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==128
    assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
    assert summary['final_dev_ce']<summary['initial_dev_ce'] and math.isfinite(summary['final_dev_ce'])
    assert config['logging_backend']=='local' and config['applied_sinkhorn_iterations']==0
    assert config['protocol']['grad_accum']==1 and config['protocol']['schedule_steps']==305176
    assert summary['peak_reserved_bytes']<int(.9*22000*1024**2)
    # Explicit post-optimizer finite-parameter audit (not only finite gradients).
    import torch
    state=torch.load('../channel-gate-checkpoints/latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(p).all() for p in state['model'].values())
    assert not list(Path('results').rglob('*.wandb'))
    new=profiles[VARIANT];base=profiles['mhc_no_sinkhorn']
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,profiles=profiles,micro=json.loads((root/'micro.json').read_text()),restart=summary,
        overhead_percent=100*(new['mean_seconds']/base['mean_seconds']-1),
        additional_peak_allocated_bytes=new['peak_allocated_bytes']-base['peak_allocated_bytes'],
        additional_peak_reserved_bytes=new['peak_reserved_bytes']-base['peak_reserved_bytes'],
        limitations='Single GPU/seed/order; 60-step profile,128-step finite training and25->128 process restart. No long-run quality/stability claim.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart','micro')},indent=2),flush=True)
if __name__=='__main__':main()
