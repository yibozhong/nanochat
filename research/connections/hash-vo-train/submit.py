"""Submit exactly one authorized fresh10B hash-VO run, with startup validation."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import uuid

REPO=Path(__file__).resolve().parents[3]
GATE=Path('/home/yzhong239/hash-vo-gate-20261107-r2')
ASSETS=Path('/home/yzhong239/nanochat-runtime-assets')
KEY='hash-vo-10b-20261107'
STEPS=305176


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def validate_gate(value,state):
    assert state['status']=='complete'  and state['exit_code']==0
    assert value['complete'] and not value['formal_training']
    assert value['source']==state['source']
    assert value['source']['model_variant']=='hash_vo'
    assert value['checks']==dict(tests=15,failures=0,errors=0,skipped=0)
    assert [p['variant'] for p in value['profiles']]==['baseline','hash','hash','baseline']
    for p in value['profiles']:
        assert p['complete'] and p['optimizer_updates']==60 and p['trained_causality_passed']
        assert p['capability']==[9,0] and p['attention']=='pinned FA3' and p['microbatch']==16
        if p['variant']=='hash':assert p['parameters']==121634816 and p['buckets']==16
    assert len({p['gpu_uuid'] for p in value['profiles']})==1
    assert len({p['input_sha256'] for p in value['profiles']})==1
    assert value['restart_checks']==value['checks']
    assert value['worker_restart']['passed'] and value['diagnostic_weights_removed']
    assert value['restart']['complete'] and value['restart']['steps']==60


def table_digest(path):
    raw=path.read_bytes();assert raw[:8]==b'\x93NUMPY\x01\x00'
    length=struct.unpack('<H',raw[8:10])[0]
    payload=raw[10+length:]
    assert len(payload)==32768*8
    return hashlib.sha256(payload).hexdigest()


def main(root):
    root=Path(root).resolve();assert root==Path('/home/yzhong239')/KEY
    root.mkdir(parents=True,exist_ok=True)
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (root/'STOPPED.json').exists()
        if (root/'state.json').exists():
            print((root/'state.json').read_text());return
        assert not (root/'submission-intent.json').exists(),'Resolve existing intent instead of duplicating'
        assert not any((root/name).exists() for name in ('checkpoints','checkpoints.tar','results.tar'))
        assert not (root/'results').exists() or not any((root/'results').iterdir())
        assert not command(['git','status','--porcelain'])
        head=command(['git','rev-parse','HEAD'])
        assert head==command(['git','ls-remote','origin','refs/heads/main']).split()[0]
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'HashVOTrainKey == "{KEY}"',
                                  '-json','-attributes','ClusterId,ProcId']) or '[]')
        assert not active,'Recover existing marked job'
        gate=json.loads((REPO/'research/connections/hash-vo/validation.json').read_text())
        state=json.loads((GATE/'state.json').read_text())
        validate_gate(gate,state)
        assert gate==json.loads((GATE/'collected/results/hash-vo/report.json').read_text())
        audit=json.loads((REPO/'research/connections/hash-vo/gpu-source-audit.json').read_text())
        for name,digest in {**audit['execution_sha256'],**audit['existing_production_files']}.items():
            assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest,name
        lookup_sha=table_digest(REPO/'research/connections/hash-qkv/lookup.npy')
        assert lookup_sha==gate['source']['lookup_sha256']
        data=json.loads(Path('/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048 and data['manifest']['training_tokens']==STEPS*32768
        safety=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert safety['complete'] and safety['actual_eviction']
        for name in ('bundles','logs','results'):(root/name).mkdir(exist_ok=True)
        bundle=root/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(bundle/'source.tar'),head])
        for name in ('gate-data.tar','transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'):
            (bundle/name).symlink_to(ASSETS/'bundles'/name)
        hashes=json.loads((ASSETS/'state.json').read_text())['source']['hashes'].copy()
        dependencies=json.loads((GATE/'bundles/dependencies.json').read_text())
        for dep in dependencies:
            name=dep.get('archive_name',dep['name']+'.tar.gz')
            origin=GATE/'bundles'/name
            assert hashlib.sha256(origin.read_bytes()).hexdigest()==dep['sha256']
            shutil.copy2(origin,bundle/name);hashes[name]=dep['sha256']
        shutil.copy2(GATE/'bundles/dependencies.json',bundle/'dependencies.json')
        shutil.copy2(REPO/'research/connections/hash-vo/worker.sh',root/'train.sh')
        data_name=data['data_uri'].rsplit('/',1)[-1];hashes[data_name]=data['archive']['sha256']
        profile=next(p for p in gate['profiles'] if p['variant']=='hash')
        source=dict(commit=head,formal_training=True,gate_job=state['job'],gpu_model_source=gate['source']['commit'],
            purpose='VO-only16-bucket balanced token hash10B, DeepGEMM, vanilla8layers D512 original ReLU-squared MLP',
            model_variant='hash_vo',buckets=16,lookup_sha256=lookup_sha,
            lookup_statistics_training_input_tokens=4194304,lookup_frozen=True,
            initialization='seed42_from_zero_no_validation_weights',steps=STEPS,training_tokens=STEPS*32768,
            data_uri=data['data_uri'],wandb_group=KEY,dependencies=dependencies,
            launcher_sha256=hashlib.sha256((root/'train.sh').read_bytes()).hexdigest(),
            profile_reference=dict(hash_vo=profile),checkpoint_upload_seconds=1800,microbatch=16,
            startup_gate='15 numerical checks plus separate25-to60 restart perH100/H200 family before formal run',
            formal_gpu_policy='fullH100/H200 only; public soft priority; backfill allowed')
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        for name in ('source.json','source.tar','dependencies.json'):
            hashes[name]=hashlib.sha256((bundle/name).read_bytes()).hexdigest()
        (bundle/'inputs.sha256').write_text(''.join(v+'  '+k+'\n' for k,v in hashes.items()))
        run_id=uuid.uuid4().hex[:8]
        template=(ASSETS/'gate.sub').read_text()
        template=template.replace('executable = validate.sh',f'executable = train.sh\narguments = train {run_id}')
        template=template.replace('request_disk = 40GB','request_disk = 80GB')
        template=template.replace('when_to_transfer_output = ON_EXIT',
            'when_to_transfer_output = ON_EXIT\ncheckpoint_exit_code = 85\ntransfer_checkpoint_files = checkpoints.tar,results.tar\nencrypt_input_files = /home/yzhong239/.netrc')
        inputs=[str(p.relative_to(root)) for p in sorted(bundle.iterdir())]
        inputs+=['/home/yzhong239/.netrc',data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        for line in template.splitlines():
            if line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            if line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            if line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            line=line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic','nanochat-hash-vo-10b')
            line=line.replace('DepthDiagnosticKey','HashVOTrainKey').replace('depth-diagnostic-20261103',KEY)
            lines.append(line)
        template='\n'.join(lines)+'\n'
        template=template.replace('(Capability == 9.0)','(Capability == 9.0) && regexp("H100|H200",DeviceName)')
        template=template.replace('queue 1','hold = true\nqueue 1')
        (root/'train.sub').write_text(template)
        command(['condor_submit','-dry-run','train-dryrun.ad','train.sub'],root)
        submission=dict(status='submitting',run_id=run_id,source=source,hashes=hashes,formal_training=True,
            expected_steps=STEPS,expected_tokens=STEPS*32768,submitted_at=time.time())
        (root/'submission-intent.json').write_text(json.dumps(submission,indent=2)+'\n')
        job=command(['condor_submit','-terse','train.sub'],root).split()[0]
        assert len(job.split('.'))==2 and all(x.isdigit() for x in job.split('.'))
        submission.update(status='training_submitted',job=job,
            wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        (root/'state.json').write_text(json.dumps(submission,indent=2)+'\n')
        rows=json.loads(command(['condor_q',job,'-json']));assert len(rows)==1;ad=rows[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==16384 and ad['RequestDisk']==80*1024**2
        assert ad['GPUsMinCapability']==9.0 and 'H100|H200' in ad['RequireGPUs']
        assert ad['SuccessCheckpointExitCode']==85
        assert set(ad['TransferCheckpoint'].split(','))=={'checkpoints.tar','results.tar'}
        assert set(ad['TransferOutput'].split(','))=={'checkpoints.tar','results.tar'}
        assert '.netrc' in ad['EncryptInputFiles'] and 'nanochat-uv-v4.sif' in ad['TransferInput']
        assert data_name in ad['TransferInput'] and 'deepgemm.tar.gz' in ad['TransferInput']
        assert ad['HashVOTrainKey']==KEY
        keys=['ClusterId','ProcId','JobStatus','NumJobStarts','RequestGPUs','RequestMemory','RequestDisk',
              'RequireGPUs','GPUsMinCapability','Rank','WantGPULab','SuccessCheckpointExitCode','TransferCheckpoint',
              'TransferOutput','TransferInput','EncryptInputFiles','ContainerImage','HashVOTrainKey']
        (root/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job])
        print(json.dumps({k:submission[k] for k in ('status','job','run_id','wandb_url','expected_steps','expected_tokens')},indent=2))


if __name__=='__main__':main(sys.argv[1])
