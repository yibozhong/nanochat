"""Validate both static mixers before publishing or starting formal training."""
import json,math,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET

VARIANTS=('mhc_static_linear','mhc_static_mlp')
COUNTS={'mhc':59515317,'mhc_channel_no_sinkhorn':59646133,'mhc_static_linear':126099621,'mhc_static_mlp':92545189}

def main():
    root=Path('results/static-mixing');root.mkdir(parents=True,exist_ok=True)
    def run(args,name,expected=0):
        print('BEGIN '+name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:log.write(line);print(line,end='',flush=True)
            proc.wait()
        assert proc.returncode==expected,(name,proc.returncode)
        print('PASS '+name,flush=True)
    run(['-m','pytest','-v','tests/test_static_mixing.py','tests/test_connection_training_resume.py',
         'tests/test_channel_mixing.py::test_fused_all_gradients_against_fp64[129-512-1.0]',
         'tests/test_channel_mixing.py::test_optimizer_rng_resume',
         'tests/test_native_connections.py::test_full_size_backbone_initialization_and_optimizer_groups',
         'tests/test_native_connections.py::test_real_kernels_causal_swa_gqa_cache_and_feature_gradients[mhc]',
         '--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=20,failures=0,errors=0,skipped=0),counts
    profiles={};restarts={};probes={}
    for v in COUNTS:
        run(['-m','scripts.connection_context_profile','--variant',v,'--sequence-length','2048',
             '--microbatch','16','--attention-backend','fa3','--output',str(root/(v+'.json'))],'profile-'+v)
        value=json.loads((root/(v+'.json')).read_text());profiles[v]=value
        assert value['complete'] and value['status']=='fits' and value['causality_passed'],value
        assert value['optimizer_updates']==60 and value['parameters']==COUNTS[v]
        assert value['peak_reserved_bytes']<int(.9*22000*1024**2)
    assert len({p['gpu_uuid'] for p in profiles.values()})==1
    assert len({p['input_sha256'] for p in profiles.values()})==1
    for v in VARIANTS:
        run(['-m','scripts.connection_static_probe','--variant',v,'--output',str(root/(v+'-initial.json'))],'probe-initial-'+v)
        args=['-m','scripts.connection_train','--variant',v,'--matched-backbone-init',
              '--steps','128','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
              '--attention-backend','fa3','--output-dir',str(root/('restart-'+v)),
              '--checkpoint-dir','../static-gate-'+v,'--run-id','restart-'+v,
              '--segment-seconds','3600','--segment-steps','25']
        run(args,'restart-first-'+v,85)
        assert json.loads((root/('restart-'+v)/'progress.json').read_text())['step']==25
        run(args+['--resume','auto','--segment-steps','0'],'restart-second-'+v)
        summary=json.loads((root/('restart-'+v)/'summary.json').read_text())
        config=json.loads((root/('restart-'+v)/'config.json').read_text())
        rows=[json.loads(s) for s in (root/('restart-'+v)/'steps.jsonl').read_text().splitlines()]
        assert summary['complete'] and summary['steps']==128
        assert [r['step'] for r in rows]==list(range(1,129)) and all(math.isfinite(r['loss']) for r in rows)
        assert summary['final_dev_ce']<summary['initial_dev_ce'] and math.isfinite(summary['final_dev_ce'])
        assert summary['peak_reserved_bytes']<int(.9*22000*1024**2)
        assert config['logging_backend']=='local' and config['applied_sinkhorn_iterations']==0
        assert config['protocol']['grad_accum']==1 and config['protocol']['schedule_steps']==305176
        run(['-m','scripts.connection_static_probe','--variant',v,'--checkpoint','../static-gate-'+v+'/latest.pt',
             '--output',str(root/(v+'-trained.json'))],'probe-trained-'+v)
        restarts[v]=summary;probes[v]={stage:json.loads((root/(v+'-'+stage+'.json')).read_text()) for stage in ('initial','trained')}
    assert not list(Path('results').rglob('*.wandb'))
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,profiles=profiles,restarts=restarts,probes=probes,
        overhead_vs_mhc_percent={v:100*(profiles[v]['mean_seconds']/profiles['mhc']['mean_seconds']-1) for v in VARIANTS},
        limitations='Single GPU/seed/order;60-step profiles,128-step finite training and25->128 process restarts. No long-run stability or quality guarantee; different parameter counts and static-mixer Muon updates.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restarts','probes')},indent=2),flush=True)
if __name__=='__main__':main()
