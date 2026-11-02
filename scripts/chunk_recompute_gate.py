"""Causality, gradients, matched timing and real process restart; never W&B."""
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import uuid
import xml.etree.ElementTree as ET

from scripts.connection_native_gate import execute


def main():
    root=Path('results');source=json.loads(Path('source.json').read_text())
    assert not os.environ.get('WANDB_API_KEY') and not Path('../.netrc').exists()
    execute([sys.executable,'-m','pytest','tests/test_chunk_recompute.py',
        'tests/test_connection_training_resume.py','-v','-s',
        '--junitxml=results/recompute-tests.xml'],root/'tests.log')
    suites=ET.parse(root/'recompute-tests.xml').getroot().findall('testsuite')
    checks={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert checks==dict(tests=15,failures=0,errors=0,skipped=0),checks
    probes=[];trials=[];micro=2
    def profile(variant,micro,repetition=0,probe=False):
        name=f'{variant}-mb{micro}-r{repetition}'+('-probe' if probe else '')
        out=root/(name+'.json')
        cmd=[sys.executable,'-m','scripts.chunk_recompute_profile','--variant',variant,
            '--microbatch',str(micro),'--repetition',str(repetition),'--output',str(out)]
        if probe:cmd.append('--probe')
        execute(cmd,root/(name+'.log'))
        return json.loads(out.read_text())
    while micro:
        pair=[profile(v,micro,probe=True) for v in ('baseline','lookback_recompute')]
        probes.extend(pair)
        if all(p['status']=='fits' for p in pair):break
        micro//=2
    assert micro,'Neither common microbatch satisfies22GB budget; architecture unchanged, no automatic launch'
    for repeat,order in enumerate((('baseline','lookback_recompute'),('lookback_recompute','baseline'))):
        for variant in order:
            trial=profile(variant,micro,repeat)
            assert trial['status']=='fits' and trial['causality_passed']
            assert trial['optimizer_updates']==50 and len(trial['step_seconds'])==30
            trials.append(trial)
    assert len({x['gpu_uuid'] for x in probes+trials})==1
    summary={}
    for variant in ('baseline','lookback_recompute'):
        selected=[x for x in trials if x['variant']==variant]
        times=[v for x in selected for v in x['step_seconds']]
        summary[variant]=dict(mean_seconds=statistics.mean(times),tokens_per_second=32768/statistics.mean(times),
            estimated_10b_training_hours=305176*statistics.mean(times)/3600,
            peak_reserved_bytes=max(x['peak_reserved_bytes'] for x in selected),
            peak_allocated_bytes=max(x['peak_allocated_bytes'] for x in selected),
            parameters=selected[0]['parameters'],compute_estimate=selected[0]['compute_estimate'])
    summary['lookback_recompute']['additional_peak_reserved_bytes']=summary['lookback_recompute']['peak_reserved_bytes']-summary['baseline']['peak_reserved_bytes']
    summary['lookback_recompute']['additional_peak_allocated_bytes']=summary['lookback_recompute']['peak_allocated_bytes']-summary['baseline']['peak_allocated_bytes']
    resumes={}
    for variant in ('lookback_recompute',):
        ident=uuid.uuid4().hex[:8];out=root/('resume-'+variant);ckpt=Path('../gate-checkpoints')/variant
        cmd=[sys.executable,'-m','scripts.connection_train','--variant',variant,
            '--sequence-length','16384','--recompute-chunk-size','2048','--steps','50','--schedule-steps','305176',
            '--microbatch',str(micro),'--attention-backend','sdpa_cudnn','--matched-backbone-init',
            '--no-activation-checkpoint','--no-fused-connection-grad','--seed','42',
            '--data-dir','../data','--output-dir',str(out),'--checkpoint-dir',str(ckpt),
            '--run-id',ident,'--resume','auto','--segment-seconds','86400']
        execute(cmd+['--segment-steps','25'],root/(variant+'-segment1.log'),85)
        progress=json.loads((out/'progress.json').read_text());assert progress['step']==25
        execute(cmd,root/(variant+'-segment2.log'))
        result=json.loads((out/'summary.json').read_text());config=json.loads((out/'config.json').read_text())
        rows=[json.loads(x) for x in (out/'steps.jsonl').read_text().splitlines()]
        assert [x['step'] for x in rows]==list(range(1,51)) and all(math.isfinite(x['loss']) for x in rows)
        assert result['complete'] and result['steps']==50 and result['training_tokens']==50*32768
        assert result['run_id']==ident and result['source']==source and config['logging_backend']=='local'
        assert config['protocol']['schedule_steps']==305176 and config['model']['sequence_length']==16384
        assert result['final_dev_ce']<result['initial_dev_ce']
        assert set(p.name for p in ckpt.iterdir())=={'latest.pt','best.pt'}
        assert result['peak_reserved_bytes']<int(.9*22000*1024**2)
        resumes[variant]=dict(summary=result,config=config)
        shutil.rmtree(ckpt)
    assert not list(root.rglob('*.wandb')) and not (root/'wandb').exists() and not list(root.rglob('*.pt'))
    result=dict(complete=True,source=source,checks=checks,logging_backend='local',
        selected_microbatch=micro,grad_accum=2//micro,sequence_length=16384,chunk_size=2048,
        chunk_attention='full_causal',shared_kv=False, per_layer_kv=True, new_learned_parameters=0,probes=probes,trials=trials,summary=summary,resumes=resumes,
        formal_training=False, data_manifest=json.loads(Path('../data/manifest.json').read_text()),
        limitations='Same single GPU timing; analytical FLOPs omit norm/softmax/optimizer. Timing excludes setup/evaluation/checkpoint transfer. No quality conclusion from50-step diagnostic.')
    (root/'recompute-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
