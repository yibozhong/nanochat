"""Submit only the bounded native-integration gate, never formal training."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
KEY='mhc-native-integration-20261031'
REFERENCE='fa6fca599483bfec79f227c4152856772ec247ed'

def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()

def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not command(['git','status','--porcelain'],REPO)
        assert not (ROOT/'submission-intent.json').exists(),'Recover existing intent; do not resubmit'
        jobs=json.loads(command(['condor_q','yzhong239','-constraint',f'NativeConnectionKey == "{KEY}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not jobs
        for folder in ('bundles','logs','results'):(ROOT/folder).mkdir(exist_ok=True)
        commit=command(['git','rev-parse','HEAD'],REPO)
        (ROOT/'submission-intent.json').write_text(json.dumps(dict(commit=commit,key=KEY),indent=2)+'\n')
        prior=Path('/home/yzhong239/mhc-no-sinkhorn-20261031/bundles')
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(prior/wheel,ROOT/'bundles'/wheel)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={ROOT/"bundles/source.tar"}',commit],REPO)
        reference=subprocess.check_output(['git','show',f'{REFERENCE}:nanochat/gpt.py'],cwd=REPO)
        (ROOT/'bundles/reference-gpt.py').write_bytes(reference)
        old=json.loads((prior/'source.json').read_text())
        hashes={k:v for k,v in old['hashes'].items() if k in (wheel,'mhc-official-runtime-7546767.tar.gz','cuda-compat-13-1_590.48.01-1_amd64.deb')}
        for name in ('source.tar','reference-gpt.py'):
            hashes[name]=hashlib.sha256((ROOT/'bundles'/name).read_bytes()).hexdigest()
        source=dict(commit=commit,reference_commit=REFERENCE,hashes=hashes,
            purpose='Native nanochat plus NVIDIA mHC/no-Sinkhorn implementation gate',formal_training=False)
        (ROOT/'bundles/source.json').write_text(json.dumps(source,indent=2)+'\n')
        (ROOT/'bundles/inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/native/validate.sh',ROOT/'validate.sh')
        submit=(REPO/'research/connections/pretrain/gate.sub').read_text()
        submit='\n'.join(l for l in submit.splitlines() if not l.startswith(('environment =','encrypt_input_files =','# Run condor_submit')))+'\n'
        submit=submit.replace('/home/yzhong239/.netrc,','')
        submit=submit.replace('bundles/source.tar,','bundles/source.tar,bundles/reference-gpt.py,')
        submit=submit.replace('nanochat-mhc-pretrain-gate','nanochat-native-mhc-gate')
        submit=submit.replace('+ConnectionGateKey = "mhc-pretrain-20261030-gate"',f'+NativeConnectionKey = "{KEY}"')
        (ROOT/'gate.sub').write_text(submit)
        command(['condor_submit','-dry-run','gate-dryrun.ad','gate.sub'])
        job=command(['condor_submit','-terse','gate.sub']).split()[0]
        state=dict(status='gate_submitted',job=job,source=source,reviewed_head=commit,submitted_at=time.time(),formal_training_submitted=False)
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2))

if __name__=='__main__':main()
