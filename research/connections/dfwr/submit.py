"""One DFWR GPU gate and the authorized10B training run."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
BASE=Path('/home/yzhong239/mhc-partition-mix-20261031')
KEY='dfwr-r32-10b-20261101'
DATA='osdf:///chtc/staging/y/yzhong239/nanochat-mhc-10b-data-7560536.tar'
GATE_DATA='osdf:///chtc/staging/y/yzhong239/nanochat-ebt-20261029/data-7444406.tar'
MODES=('dfwr',)


def command(args,cwd=ROOT):return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def save(s):
    p=ROOT/'state.tmp';p.write_text(json.dumps(s,indent=2)+'\n');p.replace(ROOT/'state.json')


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/'STOPPED.json').exists():return
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'state.json').exists() and not (ROOT/'submission-intent.json').exists(),'Recover existing state instead of duplicating jobs'
        assert not command(['git','status','--porcelain'],REPO)
        assert not json.loads(command(['condor_q','yzhong239','-constraint',f'DFWRKey == "{KEY}"','-json','-attributes','ClusterId,ProcId']) or '[]')
        protocol=json.loads((REPO/'research/connections/dfwr/protocol.json').read_text())
        assert protocol['steps']==305176 and protocol['tokens']==10000007168 and protocol['modes']==list(MODES)
        assert protocol['model']['fw_bottleneck']==32 and protocol['formal_logging']['group']==KEY
        assert ROOT.name==KEY,'Use the new r32 workflow; do not revive the stopped r8 workflow'
        for n in ('bundles','logs','results'):(ROOT/n).mkdir(exist_ok=True)
        commit=command(['git','rev-parse','HEAD'],REPO);run_ids={m:uuid.uuid4().hex[:8] for m in MODES}
        (ROOT/'submission-intent.json').write_text(json.dumps(dict(commit=commit,run_ids=run_ids,key=KEY),indent=2)+'\n')
        immutable={}
        for name in ('mhc-native-pretrain-20261031','mhc-vocab-partition-20261031','mhc-partition-mix-20261031','mhc-orthogonal-mix-10b-20261031','mhc-channel-partition-10b-20261101','nonlinear-lmhead-10b-20261101','dfwr-10b-20261101'):
            for p in (Path('/home/yzhong239')/name).rglob('*'):
                if p.is_file() and p.name in ('source.tar','source.json','inputs.sha256','train.sh','train.sub'):
                    immutable[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        (ROOT/'immutable-reference-inputs.json').write_text(json.dumps(immutable,indent=2)+'\n')
        bundle=ROOT/'bundles';prior=json.loads((BASE/'bundles/source.json').read_text())
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        shutil.copy2(BASE/'bundles'/wheel,bundle/wheel)
        command(['git','archive','--format=tar','--prefix=nanochat-src/',f'--output={bundle/"source.tar"}',commit],REPO)
        hashes={k:v for k,v in prior['hashes'].items() if k!='source.tar'}
        hashes['source.tar']=hashlib.sha256((bundle/'source.tar').read_bytes()).hexdigest()
        digest=protocol['data_archive_sha256'];assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
        for name in ('train.sh','validate.sh'):shutil.copy2(REPO/'research/connections/dfwr'/name,ROOT/name)
        source=dict(commit=commit,hashes=hashes,steps=305176,training_tokens=10000007168,
            fw_bottleneck=32,parameters=59031555,
            purpose='Depth-wise Fast Weight Residual r32, per-token state, differentiable Inner Muon,10B run',
            data_uri=DATA,data_split_sha256=protocol['data_split_sha256'],
            data_archive_sha256=digest,gate_data_uri=GATE_DATA,
            gate_data_archive_sha256=protocol['gate']['data_archive_sha256'],
            launcher_sha256=hashlib.sha256((ROOT/'train.sh').read_bytes()).hexdigest(),
            gate_launcher_sha256=hashlib.sha256((ROOT/'validate.sh').read_bytes()).hexdigest())
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        for role,name,data_digest in [('gate','data-7444406.tar',source['gate_data_archive_sha256']),
                                     ('train','nanochat-mhc-10b-data-7560536.tar',digest)]:
            folder=bundle/role;folder.mkdir()
            (folder/'inputs.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in hashes.items())+f'{data_digest}  {name}\n')
        common='''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 2
request_memory = 16GB
request_gpus = 1
gpus_minimum_memory = 22000M
gpus_minimum_capability = 9.0
require_gpus = (Capability == 9.0) && ((MaxSupportedVersion >= 13010) || regexp("^(570|580)[.]",NvidiaDriver)) && (GlobalMemoryMb >= 22000) && !regexp("MIG",DeviceName)
+WantGPULab = true
+GPUJobLength = "short"
+is_resumable = true
rank = ifThenElse(regexp("^backfill", TARGET.Name), 0.0, 1.0)
'''+f'+DFWRKey = "{KEY}"\n'
        inputs=','.join([f'bundles/{wheel}','bundles/source.tar','bundles/source.json',
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb'])
        gate=common+f'''executable = validate.sh
request_disk = 45GB
transfer_input_files = {inputs},bundles/gate/inputs.sha256,{GATE_DATA}
transfer_output_files = results.tar
transfer_output_remaps = "results.tar=results/gate-$(Cluster).$(Process).tar"
log = logs/gate.$(Cluster).$(Process).log
output = logs/gate.$(Cluster).$(Process).out
error = logs/gate.$(Cluster).$(Process).err
+DFWRRole = "gate"
+JobBatchName = "nanochat-dfwr-r32-gate"
queue 1
'''
        specs={'gate':gate}
        for mode in MODES:
            specs['train-'+mode]=common+f'''executable = train.sh
arguments = dfwr {run_ids[mode]} 305176 32
environment = "NANOCHAT_DATA_ARCHIVE=nanochat-mhc-10b-data-7560536.tar"
encrypt_input_files = /home/yzhong239/.netrc
hold = True
request_disk = 80GB
checkpoint_exit_code = 85
transfer_checkpoint_files = checkpoints.tar,results.tar
transfer_input_files = {inputs},bundles/train/inputs.sha256,{DATA},/home/yzhong239/.netrc
transfer_output_files = results.tar,checkpoints.tar
transfer_output_remaps = "results.tar=results/{mode}-$(Cluster).$(Process).tar;checkpoints.tar=results/{mode}-$(Cluster).$(Process)-checkpoints.tar"
log = logs/{mode}.$(Cluster).$(Process).log
output = logs/{mode}.$(Cluster).$(Process).out
error = logs/{mode}.$(Cluster).$(Process).err
+DFWRRole = "formal"
+DFWRMode = "{mode}"
+MHCRunID = "{run_ids[mode]}"
+JobBatchName = "nanochat-dfwr-r32-10b"
queue 1
'''
        for name,body in specs.items():
            (ROOT/(name+'.sub')).write_text(body)
            command(['condor_submit','-dry-run',name+'-dryrun.ad',name+'.sub'])
        state=dict(status='submitting',source=source,reviewed_head=commit,training_jobs={},
            gate_verified=False,expected_steps=305176,expected_tokens=10000007168,submitted_at=time.time())
        save(state)
        for mode in MODES:
            job=command(['condor_submit','-terse','train-'+mode+'.sub']).split()[0]
            assert all(x.isdigit() for x in job.split('.')) and len(job.split('.'))==2
            state['training_jobs'][mode]=dict(job=job,run_id=run_ids[mode],released=False,collected=False);save(state)
        state['gate_job']=command(['condor_submit','-terse','gate.sub']).split()[0]
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in immutable.items())
        state['status']='gate_pending_training_held';save(state)
        print(json.dumps(dict(gate=state['gate_job'],training=state['training_jobs']),indent=2))


if __name__=='__main__':main()
