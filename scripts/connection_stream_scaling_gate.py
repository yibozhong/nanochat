"""One GPU, frozen-weight n=4/8/16/32 speed sweep. Never launches training."""
import json,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root=Path('results/stream-scaling');root.mkdir(parents=True,exist_ok=True)
    def run(args,name):
        print('BEGIN '+name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            p=subprocess.Popen([sys.executable,*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in p.stdout:log.write(line);print(line,end='',flush=True)
            p.wait()
        assert p.returncode==0,(name,p.returncode)
    run(['-m','pytest','-v','tests/test_stream_scaling.py','--junitxml='+str(root/'tests.xml')],'tests')
    suites=ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=30,failures=0,errors=0,skipped=0),counts
    records=[]
    def profile(kind,n,batch,scope,anchor=False):
        name=f'{scope}-{kind}-n{n}-b{batch}'+('-anchor' if anchor else '')
        args=['-m','scripts.connection_stream_scaling_profile','--kind',kind,'--streams',str(n),
            '--batch',str(batch),'--scope',scope,'--output',str(root/(name+'.json'))]
        if anchor:args.append('--anchor')
        run(args,name);record=json.loads((root/(name+'.json')).read_text())
        assert record['complete'] and record['parameter_updates']==0 and record['training_tokens']==0
        records.append(record);return record
    for n in (4,8,16,32):
        for kind in (('mhc','mlp') if n in (4,16) else ('mlp','mhc')):profile(kind,n,16,'micro')
    selected=None
    for batch in (16,8,4):
        trial=[]
        for n in (4,8,16,32):
            for kind in (('mhc','mlp') if n in (4,16) else ('mlp','mhc')):trial.append(profile(kind,n,batch,'model'))
        if all(r['status']=='measured' for r in trial):selected=batch;break
    # Existing optimized four-stream implementations are separate anchors,
    # never spliced into the generalized scaling curves.
    for kind in ('mhc','mlp'):
        profile(kind,4,16,'micro',True)
        profile(kind,4,selected or 4,'model',True)
    assert len({r['gpu_uuid'] for r in records})==1
    assert not list(Path('results').rglob('*.wandb')) and not list(Path('results').rglob('*.pt'))
    result=dict(complete=True,formal_training=False,speed_only=True,parameter_updates=0,
        source=json.loads(Path('source.json').read_text()),checks=counts,streams=[4,8,16,32],
        mlp_inner_rule='4*streams',selected_common_model_batch=selected,records=records,
        limitation='Generalized benchmark kernels, not official NVIDIA8/16/32 support. Same compiled cuBLAS projection on both generalized methods; official4 anchors separate. Repeated fixed weights/data; no optimizer time or quality evaluation. OOM never causes a per-variant batch reduction.')
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['scope,kind,streams,batch,anchor,status,forward_ms,forward_backward_ms,peak_reserved_bytes']
    for r in records:
        lines.append(','.join(map(str,[r['scope'],r['kind'],r['streams'],r['batch'],r['anchor'],r['status'],
            r.get('forward',{}).get('mean_ms',''),r.get('forward_backward',{}).get('mean_ms',''),
            r.get('forward_backward',{}).get('peak_reserved_bytes','')])))
    (root/'timing.csv').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,selected_common_model_batch=selected,parameter_updates=0)),flush=True)


if __name__=='__main__':main()
