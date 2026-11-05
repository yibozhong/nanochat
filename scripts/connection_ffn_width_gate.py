"""Bounded correctness/profile/restart check; never launches formal training."""
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    root=Path('results/ffn-width');root.mkdir(parents=True,exist_ok=True)
    def run(args,name,expected=0):
        with (root/(name+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==expected,(name,result.returncode)

    run(['-m','pytest','-q','tests/test_ffn_width_residual.py',
         'tests/test_connection_training_resume.py','tests/test_fla_attention.py',
         '--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={key:sum(int(s.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=28,failures=0,errors=0,skipped=0),counts
    profiles={}
    for enabled in (False,True):
        name='ffn_width_residual' if enabled else 'baseline'
        args=['-m','scripts.connection_context_profile','--variant','baseline','--sequence-length','2048',
              '--microbatch','16','--attention-backend','fa3','--output',str(root/(name+'.json'))]
        if enabled:args.append('--ffn-width-residual')
        run(args,'profile-'+name)
        value=json.loads((root/(name+'.json')).read_text());profiles[name]=value
        assert value['complete'] and value['status']=='fits' and value['causality_passed']
        assert value['parameters']==58720256 and value['optimizer_updates']==60
        assert value['model']['ffn_width_residual']==enabled
    baseline,new=profiles.values()
    assert baseline['input_sha256']==new['input_sha256'] and baseline['gpu_uuid']==new['gpu_uuid']
    args=['-m','scripts.connection_train','--variant','baseline','--ffn-width-residual',
          '--steps','50','--schedule-steps','305176','--microbatch','16','--data-dir','../data',
          '--attention-backend','fa3','--output-dir',str(root/'restart'),
          '--checkpoint-dir','../ffn-width-gate-checkpoints','--run-id','ffn-width-restart',
          '--segment-seconds','3600','--segment-steps','25']
    run(args,'restart-first',85)
    progress=json.loads((root/'restart/progress.json').read_text());assert progress['step']==25
    run(args+['--resume','auto'],'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text())
    config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(x) for x in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==50
    assert [r['step'] for r in rows]==list(range(1,51))
    assert all(math.isfinite(r['loss']) for r in rows)
    assert config['logging_backend']=='local' and config['model']['ffn_width_residual']
    assert config['protocol']['schedule_steps']==305176 and config['protocol']['grad_accum']==1
    assert summary['peak_reserved_bytes']<int(.9*22000*1024**2)
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
                checks=counts,profiles=profiles,restart=summary,
                overhead_percent=100*(new['mean_seconds']/baseline['mean_seconds']-1),
                additional_peak_allocated_bytes=new['peak_allocated_bytes']-baseline['peak_allocated_bytes'],
                additional_peak_reserved_bytes=new['peak_reserved_bytes']-baseline['peak_reserved_bytes'],
                limitations='60-step single-order profile and 25-to-50 restart only; no long-run quality or stability conclusion.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','restart')},indent=2),flush=True)


if __name__=='__main__':main()
