"""One authorized 10B run, only after cache readiness and published GPU gate."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[4]
ROOT = Path(sys.argv[1]).resolve()
ACTIVE = Path('/home/yzhong239/mhc-stream-independent-20261103/ACTIVE.json')
KEY = 'mhc-attn-independent-10b-20261103'


def command(args, cwd=REPO):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=60).strip()


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    GATE=Path(json.loads(ACTIVE.read_text())['workflow'])
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'STOPPED.json').exists()
        if (ROOT/'state.json').exists(): print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover existing submission; do not duplicate'
        assert not json.loads(command(['condor_q','yzhong239','-constraint',f'StreamPretrainKey == "{KEY}"','-json']) or '[]')
        ready=json.loads(Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert ready['ready'] and ready['sequence_length']==2048
        assert ready['manifest']['training_tokens']==305176*32768
        gate=json.loads((GATE/'results.json').read_text())
        published=json.loads((GATE/'published.json').read_text())
        assert gate['complete'] and gate['checks']==dict(tests=36,failures=0,errors=0,skipped=0)
        assert published['gate_commit']==gate['source']['commit']
        assert command(['git','ls-remote','origin','refs/heads/main']).split()[0]==published['remote_commit']
        assert all(t['fits_22gb_budget'] and t['causal_prefix_max_abs']==0 for t in gate['trials'])
        assert not command(['git','status','--porcelain'])
        for folder in ('bundles','logs','results'): (ROOT/folder).mkdir(exist_ok=True)
        bundle=ROOT/'bundles'
        # Copy the exact GPU-verified source; orchestration changes do not alter it.
        for name in ('source.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            shutil.copy2(GATE/'bundles'/name,bundle/name)
        shutil.copy2(REPO/'research/connections/stream-independent/pretrain/train.sh',ROOT/'train.sh')
        hashes={k:v for k,v in gate['source']['hashes'].items() if k != 'gate-data.tar'}
        data_name=ready['data_uri'].rsplit('/',1)[-1];hashes[data_name]=ready['archive']['sha256']
        source=dict(commit=gate['source']['commit'],hashes=hashes,gate_job=published['job'],
            purpose=KEY,formal_training=True,steps=305176,training_tokens=305176*32768,
            architecture='vanilla',variant='mhc',attention_mode='per_stream_independent',
            attention_parameter_layout='dense_sliced',data_uri=ready['data_uri'],launcher_sha256=hashlib.sha256((ROOT/'train.sh').read_bytes()).hexdigest())
        assert gate['restart']['complete'] and gate['restart']['final_step']==50
        write(bundle/'source.json',source)
        (bundle/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items()))
        runid=uuid.uuid4().hex[:8]
        text=(GATE/'gate.sub').read_text()
        text=text.replace('executable = validate.sh',f'executable = train.sh\narguments = mhc {runid} 305176\nenvironment = "NANOCHAT_DATA_ARCHIVE={data_name} NANOCHAT_CHECKPOINT_SECONDS=1800"\nencrypt_input_files = /home/yzhong239/.netrc\ncheckpoint_exit_code = 85\ntransfer_checkpoint_files = checkpoints.tar,results.tar')
        text=text.replace('request_disk = 40GB','request_disk = 80GB')
        text=text.replace('bundles/gate-data.tar,','')
        text=text.replace('transfer_input_files = ',f'transfer_input_files = /home/yzhong239/.netrc,{ready["data_uri"]},')
        text=text.replace('transfer_output_files = results.tar','transfer_output_files = results.tar,checkpoints.tar')
        text=text.replace('"results.tar=results/stream-$(Cluster).$(Process).tar"','"results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/checkpoints-$(Cluster).$(Process).tar"')
        text=text.replace('logs/stream.','logs/train.').replace('nanochat-stream-independent-gate',KEY)
        text=text.replace('+StreamAttentionKey = "mhc-stream-independent-20261103"',f'+StreamPretrainKey = "{KEY}"\n+MHCRunID = "{runid}"')
        (ROOT/'train.sub').write_text(text)
        command(['condor_submit','-dry-run','dryrun.ad','train.sub'],ROOT)
        write(ROOT/'submission-intent.json',dict(source=source,run_id=runid,created_at=time.time()))
        job=command(['condor_submit','-terse','train.sub'],ROOT).split()[0]
        state=dict(status='submitted',job=job,run_id=runid,source=source,submitted_at=time.time(),formal_training=True)
        write(ROOT/'state.json',state);print(json.dumps(state,indent=2))


if __name__=='__main__': main()
