"""Bounded compact-routing validation and timing only; never formal training."""
import json,math,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root=Path('results/compact');root.mkdir(parents=True,exist_ok=True)
    def run(args,name,expected=0):
        print('BEGIN '+name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:log.write(line);print(line,end='',flush=True)
            proc.wait()
        assert proc.returncode==expected,(name,proc.returncode)
        print('PASS '+name,flush=True)
    run(['-m','pytest','-v','tests/test_compact_routing.py','--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=13,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_compact_micro'],'micro')
    profiles={}
    for name,flag in [('padded_nvidia','--no-compact-routing'),('compact8','--compact-routing')]:
        run(['-m','scripts.connection_context_profile','--variant','mhc_stream_mlp',flag,
            '--sequence-length','2048','--microbatch','16','--attention-backend','hardware_auto',
            '--output',str(root/(name+'.json'))],'profile-'+name)
        p=json.loads((root/(name+'.json')).read_text());profiles[name]=p
        assert p['complete'] and p['status']=='fits' and p['causality_passed'] and p['parameters']==58992805
        assert p['optimizer_updates']==60 and p['peak_reserved_bytes']<int(.9*22000*1024**2)
    assert len({p['gpu_uuid'] for p in profiles.values()})==1
    assert len({p['input_sha256'] for p in profiles.values()})==1
    args=['-m','scripts.connection_train','--variant','mhc_stream_mlp','--compact-routing','--matched-backbone-init',
        '--steps','50','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
        '--attention-backend','hardware_auto','--output-dir',str(root/'restart'),
        '--checkpoint-dir','../compact-gate-checkpoints','--run-id','compact-gate',
        '--segment-seconds','3600','--segment-steps','25']
    run(args,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text());config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==50 and config['model']['compact_routing']
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert [r['step'] for r in rows]==list(range(1,51)) and all(math.isfinite(r['loss']) for r in rows)
    assert not list(Path('results').rglob('*.wandb'))
    ratio=profiles['compact8']['mean_seconds']/profiles['padded_nvidia']['mean_seconds']
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,profiles=profiles,micro=json.loads((root/'micro.json').read_text()),restart=summary,
        step_time_reduction_percent=100*(1-ratio),default_enabled=False,
        limits='Same GPU/kernel-only comparison; no new formal run.8 useful routing coefficients,16-wide tensor-core tiles instead of32; final collapse remains NVIDIA. Floating-point reduction ordering differs.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(complete=True,step_time_reduction_percent=result['step_time_reduction_percent'])),flush=True)


if __name__=='__main__':main()
