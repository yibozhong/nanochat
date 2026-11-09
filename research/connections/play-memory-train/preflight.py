"""Reuse verified families; check a new family before any formal model/run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.connection_skip_scale_gate import run


def validate_marker(v,source,microbatch,capability):
    assert v['complete'] and v['preflight'] and not v['formal_training']
    assert v['source']==source and v['microbatch']==microbatch and v['capability']==capability
    assert v['checks']==dict(tests=23,failures=0,errors=0,skipped=0)
    r=v['restart'];assert r['complete'] and r['steps']==60 and r['variant']=='baseline' and r['source']==source
    assert math.isfinite(r['final_dev_ce']) and r['final_dev_ce']<r['initial_dev_ce']
    assert r['training_tokens']==60*32768 and r['peak_reserved_bytes']<.9*22000*1024**2
    c=v['model']
    assert c['mlp_memory']=='play' and c['streams']==1 and v['parameters']==58753024
    assert c['architecture']=='vanilla' and c['depth']==8 and c['width']==512 and c['heads']==4
    assert c['sequence_length']==2048 and c['play_radius_init']==.25
    assert len(v['weights'])==len(v['radii'])==8
    assert all(math.isfinite(x) and x>1e-7 for x in v['weights'].values())
    assert all(math.isfinite(x) and x>0 for x in v['radii'].values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--microbatch',type=int,required=True);a=parser.parse_args()
    source=json.loads(Path('source.json').read_text());device=json.loads(Path('results/device.json').read_text())
    cap=device['capability'];gpu=device['gpu']
    assert (cap==[9,0] and any(n in gpu for n in ('H100','H200'))) or (cap==[8,0] and 'A100' in gpu) or (cap==[12,0] and gpu=='NVIDIA RTX PRO 6000 Blackwell Server Edition')
    assert source['formal_training'] and source['model_variant']=='baseline' and source['mlp_memory']=='play'
    assert source['microbatch']==a.microbatch
    for name,digest in source['validated_execution_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    if cap in source['passed_families']:
        print('Reusing full play-memory GPU gate for this family; source hashes match',flush=True);return
    marker=Path('results')/('play-memory-family-'+'.'.join(map(str,cap))+'.json')
    if marker.exists():
        validate_marker(json.loads(marker.read_text()),source,a.microbatch,cap)
        print('Reusing successful family preflight for this run',flush=True);return
    os.environ['WANDB_MODE']='disabled'
    root=Path('results')/('play-memory-preflight-'+'.'.join(map(str,cap)))
    checkpoint=Path('../play-memory-family-checkpoints')
    shutil.rmtree(root,ignore_errors=True);shutil.rmtree(checkpoint,ignore_errors=True);root.mkdir(parents=True)
    run(['-m','pytest','-v','tests/test_play_memory.py','--junitxml='+str(root/'tests.xml')],root,'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=23,failures=0,errors=0,skipped=0)
    args=['-m','scripts.connection_train','--variant','baseline','--mlp-memory','play','--streams','1',
          '--architecture','vanilla','--play-radius-init','.25','--sequence-length','2048','--seed','42',
          '--steps','60','--schedule-steps','305176','--microbatch',str(a.microbatch),
          '--data-dir','../data','--attention-backend','hardware_auto','--output-dir',str(root/'train'),
          '--checkpoint-dir',str(checkpoint),'--run-id','play-family-check','--segment-seconds','3600',
          '--segment-steps','25']
    run(args,root,'restart-first',85)
    assert json.loads((root/'train/progress.json').read_text())['step']==25
    run(args+['--resume','auto','--segment-steps','0'],root,'restart-second')
    summary=json.loads((root/'train/summary.json').read_text());cfg=json.loads((root/'train/config.json').read_text())
    rows=[json.loads(s) for s in (root/'train/steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,61)) and all(math.isfinite(r['loss']) for r in rows)
    assert cfg['logging_backend']=='local' and cfg['protocol']['schedule_steps']==305176
    # Parent remains CUDA-context-free while the GPU children execute.
    import torch
    state=torch.load(checkpoint/'latest.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(p).all() for p in state['model'].values())
    weights={n:p.abs().max().item() for n,p in state['model'].items() if n.endswith('memory_weight')}
    radii={n:torch.nn.functional.softplus(p).min().item() for n,p in state['model'].items() if n.endswith('raw_radius')}
    v=dict(complete=True,formal_training=False,preflight=True,source=source,microbatch=a.microbatch,
           capability=cap,gpu=gpu,checks=counts,restart=summary,model=cfg['model'],parameters=cfg['parameters'],
           weights=weights,radii=radii)
    validate_marker(v,source,a.microbatch,cap)
    del state
    shutil.rmtree(checkpoint)
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(marker)
    print('PASS new-family play-memory preflight; diagnostic weights deleted',flush=True)


if __name__=='__main__':main()
