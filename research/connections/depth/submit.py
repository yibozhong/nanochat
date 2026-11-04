"""Submit one read-only GPU diagnostic against existing completed checkpoints."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

REPO=Path(__file__).resolve().parents[3]
ROOT=Path(sys.argv[1]).resolve()
PREVIOUS=Path('/home/yzhong239/nanochat-runtime-assets')


def command(args):return subprocess.check_output(args,cwd=REPO,text=True).strip()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'state.json').exists() and not (ROOT/'STOPPED.json').exists()
        assert not command(['git','status','--porcelain'])
        for d in ('bundles','logs','results'):(ROOT/d).mkdir(exist_ok=True)
        inputs={}
        for name,folder,stem in [('vanilla','mhc-pretrain-20261030','baseline-7562105.0'),
                                 ('mhc','mhc-pretrain-20261030','mhc-7562105.1'),
                                 ('native','mhc-native-pretrain-20261031','baseline-7638116.0')]:
            parent=Path('/home/yzhong239')/folder/'results'
            with tarfile.open(parent/(stem+'.tar')) as tf:
                summary=json.load(tf.extractfile('results/train/summary.json'))
            inputs[name]=dict(checkpoint='../checkpoints/'+name+'.pt',archive=stem+'-checkpoints.tar',
                checkpoint_sha256=summary['checkpoint_files']['latest.pt']['sha256'],
                historical_dev_ce=summary['final_dev_ce'])
            (ROOT/'bundles'/(stem+'-checkpoints.tar')).symlink_to(parent/(stem+'-checkpoints.tar'))
        (ROOT/'bundles/diagnostic-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
        source=dict(commit=command(['git','rev-parse','HEAD']),formal_training=False,
                    purpose='Read-only 2K depth utilization; vanilla/native/vanilla+mHC at matched 10B')
        (ROOT/'bundles/source.json').write_text(json.dumps(source,indent=2)+'\n')
        subprocess.run(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(ROOT/'bundles/source.tar'),'HEAD'],cwd=REPO,check=True)
        for name in ('gate-data.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (ROOT/'bundles'/name).symlink_to(PREVIOUS/'bundles'/name)
        # Reuse existing immutable runtime/data checksums; no new data hashing experiment.
        prior=json.loads((PREVIOUS/'state.json').read_text())['source']['hashes']
        hashes={k:v for k,v in prior.items() if k!='source.tar'}
        for name in ('source.tar','source.json','diagnostic-inputs.json'):
            hashes[name]=hashlib.sha256((ROOT/'bundles'/name).read_bytes()).hexdigest()
        (ROOT/'bundles/inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        shutil.copy2(REPO/'research/connections/depth/run.sh',ROOT/'run.sh')
        template=(PREVIOUS/'gate.sub').read_text()
        transfer=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'bundles').iterdir())]
        transfer.extend(['osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb'])
        lines=[]
        for line in template.splitlines():
            if line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(transfer)
            line=line.replace('executable = validate.sh','executable = run.sh').replace('stream-$(Cluster)','depth-$(Cluster)').replace('logs/stream.','logs/depth.').replace('nanochat-stream-independent-gate','nanochat-depth-diagnostic').replace('StreamAttentionKey','DepthDiagnosticKey').replace('mhc-stream-independent-20261103','depth-diagnostic-20261103')
            lines.append(line)
        (ROOT/'diagnostic.sub').write_text('\n'.join(lines)+'\n')
        subprocess.run(['condor_submit','-dry-run',str(ROOT/'dry-run.ads'),'diagnostic.sub'],cwd=ROOT,check=True,capture_output=True)
        (ROOT/'submission-intent.json').write_text(json.dumps(source,indent=2)+'\n')
        result=subprocess.check_output(['condor_submit','-terse','diagnostic.sub'],cwd=ROOT,text=True).strip()
        job=result.split()[0]
        state=dict(status='submitted',job=job,source=source,formal_training=False)
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2))


if __name__=='__main__':main()
