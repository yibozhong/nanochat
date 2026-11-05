"""Kernel optimization only: no W&B, no new formal run or changed live inputs."""
import json,math,statistics,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root=Path('results/stream-mlp-opt');root.mkdir(parents=True,exist_ok=True)
    def run(args,name,expected=0):
        print('BEGIN '+name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in proc.stdout:log.write(line);print(line,end='',flush=True)
            proc.wait()
        assert proc.returncode==expected,(name,proc.returncode)
    run(['-m','pytest','-v','tests/test_stream_mlp_optimized.py','--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=15,failures=0,errors=0,skipped=0),counts
    run(['-m','scripts.connection_stream_mlp_opt_micro'],'micro')
    micro=json.loads((root/'micro.json').read_text());winner=micro['winner']
    # Preserve an honest negative result if none of the new implementations wins.
    candidate=min((k for k in micro['scores_ms'] if k!='legacy'),key=micro['scores_ms'].get)
    profiles={}
    # ABBA whole-model repetition with unchanged routing isolates mixer changes.
    for i,kind in enumerate(('legacy',candidate,candidate,'legacy')):
        name=f'{i}-{kind}'
        run(['-m','scripts.connection_context_profile','--variant','mhc_stream_mlp',
            '--stream-mlp-kernel',kind,'--sequence-length','2048','--microbatch','16',
            '--attention-backend','hardware_auto','--output',str(root/(name+'.json'))],'profile-'+name)
        p=json.loads((root/(name+'.json')).read_text());profiles[name]=p
        assert p['complete'] and p['status']=='fits' and p['causality_passed'] and p['parameters']==58992805
        assert p['optimizer_updates']==60 and p['peak_reserved_bytes']<int(.9*22000*1024**2)
    run(['-m','scripts.connection_context_profile','--variant','mhc_stream_mlp','--stream-mlp-kernel',candidate,
        '--compact-routing','--sequence-length','2048','--microbatch','16',
        '--attention-backend','hardware_auto','--output',str(root/'combined.json')],'profile-combined')
    combined=json.loads((root/'combined.json').read_text())
    assert combined['complete'] and combined['status']=='fits' and combined['causality_passed']
    assert len({p['gpu_uuid'] for p in [*profiles.values(),combined]})==1
    assert len({p['input_sha256'] for p in [*profiles.values(),combined]})==1
    args=['-m','scripts.connection_train','--variant','mhc_stream_mlp','--stream-mlp-kernel',candidate,
        '--compact-routing','--matched-backbone-init','--steps','50','--schedule-steps','305176',
        '--microbatch','16','--data-dir','../data','--attention-backend','hardware_auto',
        '--output-dir',str(root/'restart'),'--checkpoint-dir','../stream-opt-checkpoints',
        '--run-id','stream-opt-gate','--segment-seconds','3600','--segment-steps','25']
    run(args,'restart-first',85)
    assert json.loads((root/'restart/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],'restart-second')
    summary=json.loads((root/'restart/summary.json').read_text());config=json.loads((root/'restart/config.json').read_text())
    rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
    assert summary['complete'] and summary['steps']==50 and config['model']['stream_mlp_kernel']==candidate
    assert config['logging_backend']=='local' and config['protocol']['schedule_steps']==305176
    assert [r['step'] for r in rows]==list(range(1,51)) and all(math.isfinite(r['loss']) for r in rows)
    assert not list(Path('results').rglob('*.wandb'))
    means={k:statistics.mean(p['mean_seconds'] for p in profiles.values() if p['model']['stream_mlp_kernel']==k)
           for k in ('legacy',candidate)}
    result=dict(complete=True,formal_training=False,source=json.loads(Path('source.json').read_text()),
        checks=counts,micro=micro,candidate=candidate,profiles=profiles,combined=combined,restart=summary,
        step_time_reduction_percent=100*(1-means[candidate]/means['legacy']),
        combined_time_reduction_percent=100*(1-combined['mean_seconds']/means['legacy']),
        recommended=(winner==candidate and means[candidate]<.98*means['legacy']),default_enabled=False,
        limits='One GPU family per report; no formal runs changed. FP32 arithmetic ordering changes; no extra BF16 hidden rounding.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','candidate','recommended','step_time_reduction_percent','combined_time_reduction_percent')}),flush=True)


if __name__=='__main__':main()
