"""Submit fixed 10B-token preparation; CPU only, no W&B credentials or runs."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(sys.argv[1]).resolve()
REPO=Path(__file__).resolve().parents[3]
OLD=Path('/home/yzhong239/nanochat-ebt')


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    with (ROOT/'state.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((ROOT/'state.json').read_text())
        assert state['budget_confirmed_by_user'] and state['requested_tokens']==10_000_000_000
        assert state['wandb_cleanup_complete'] and not state.get('automation_paused_for_wandb_cleanup')
        if state.get('data_job'):
            print(state['data_job']);return
        assert not json.loads(command(['condor_q','yzhong239','-constraint','MHCDataKey == "mhc-10b-20261030"','-json']) or '[]')
        bundle=ROOT/'data-bundles';bundle.mkdir(exist_ok=True)
        for name in ('nanochat-145da40.tar','tokenizer.tar'):
            shutil.copy2(OLD/'bundles'/name,bundle/name)
        shutil.copy2(OLD/'dataset-manifest.json',bundle/'dataset-manifest.json')
        for name in ('prepare_data.py','prepare_data.sh'):
            shutil.copy2(REPO/'research/connections/pretrain'/name,bundle/name)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir() if p.name!='prepare-inputs.sha256'}
        (bundle/'prepare-inputs.sha256').write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()))
        manifest=json.loads((bundle/'dataset-manifest.json').read_text())
        inputs=[str(p.relative_to(ROOT)) for p in sorted(bundle.iterdir())]
        inputs += [v['osdf_uri'] for v in manifest['files']]
        submit=f'''universe = vanilla
container_image = file:///staging/y/yzhong239/gradformer/nanochat-uv-v4.sif
executable = data-bundles/prepare_data.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
stream_output = True
stream_error = True
requirements = (HasCHTCStaging == true)
request_cpus = 4
request_memory = 16GB
request_disk = 90GB
transfer_input_files = {','.join(inputs)}
transfer_output_files = results.tar,data.tar
transfer_output_remaps = "results.tar=results/data-$(Cluster).$(Process).tar;data.tar=osdf:///chtc/staging/y/yzhong239/nanochat-mhc-10b-data-$(Cluster).tar"
log = logs/data.$(Cluster).$(Process).log
output = logs/data.$(Cluster).$(Process).out
error = logs/data.$(Cluster).$(Process).err
+JobBatchName = "nanochat-fineweb-10b-data"
+MHCDataKey = "mhc-10b-20261030"
queue 1
'''
        (ROOT/'prepare-data.sub').write_text(submit)
        command(['condor_submit','-dry-run','prepare-data-dryrun.ad','prepare-data.sub'])
        output=command(['condor_submit','-terse','prepare-data.sub']);cluster=output.split('.')[0]
        assert cluster.isdigit(),output
        state.update(data_job=cluster+'.0',status='preparing_10b_data',data_preparation_hashes=hashes,
            data_source_commit=command(['git','rev-parse','HEAD'],REPO),
            data_uri=f'osdf:///chtc/staging/y/yzhong239/nanochat-mhc-10b-data-{cluster}.tar')
        tmp=ROOT/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'state.json')
        print(state['data_job'])


if __name__=='__main__':main()
