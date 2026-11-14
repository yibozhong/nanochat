"""One bounded, non-W&B GPU debug job. No formal training continuation."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

REPO=Path(__file__).resolve().parents[3]
ASSETS=Path('/home/yzhong239/nanochat-runtime-assets')
ROOT=Path('/home/yzhong239/feedback-backprop-debug-20261114')
KEY=ROOT.name


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'STOPPED').exists()
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover previous submission first'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'FeedbackBackpropDebugKey == "{KEY}"','-json']) or '[]')
        assert not active
        assert not command(['git','status','--porcelain'])
        commit=command(['git','rev-parse','HEAD'])
        for name in ('bundles','logs','results'):(ROOT/name).mkdir(exist_ok=True)
        bundle=ROOT/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(bundle/'source.tar'),commit])
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        for name in (wheel,):
            target=ASSETS/'bundles'/name
            assert target.is_file()
            (bundle/name).symlink_to(target)
        source=dict(commit=commit,formal_training=False,purpose='Paired detach versus cross-pass backprop: cost and short-training loss',
            variants=['embedding_qkv','per_layer_qkv'],
            training_steps_per_model=2048,tokens_per_model=2048*32768,cli_steps=3,
            detach_options=[True,False],projection_order='reassociated',
            efficiency_checks=['same-GPU P2 full-step comparison','training memory peak','paired dev P1/P2/P4 every256steps'],
            data='existing packed2K prefix from FineWeb-Edu sample-10BT',
            data_uri='osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-7685535.tar',
            runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock',
            gpu_backend='H100/H200 + FA2',seed=42,pass_mix=[.15,.85],eval_passes=4,
            timeout_seconds=2400,failure_debug_seconds=900)
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        shutil.copy2(REPO/'research/connections/feedback-backprop/debug.sh',ROOT/'debug.sh')
        (ROOT/'debug.sh').chmod(0o755)
        inputs=[str(p.relative_to(ROOT)) for p in bundle.iterdir()]
        inputs += [source['data_uri'],'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        template=(ASSETS/'gate.sub').read_text()
        lines=[]
        for line in template.splitlines():
            if line.startswith('executable ='):line='executable = debug.sh'
            elif line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            elif line.startswith('request_memory ='):line='request_memory = 32GB'
            elif line.startswith('request_disk ='):line='request_disk = 80GB'
            elif line.startswith('gpus_minimum_memory ='):line='gpus_minimum_memory = 78000M'
            elif line.startswith('require_gpus ='):
                line=line.replace('(Capability == 9.0)','(Capability == 9.0 && regexp("H100|H200",DeviceName))').replace('GlobalMemoryMb >= 22000','GlobalMemoryMb >= 78000')
            elif line.startswith('queue '):continue
            lines.append(line.replace('DepthDiagnosticKey','FeedbackBackpropDebugKey')
                .replace('depth-diagnostic-20261103',KEY).replace('nanochat-depth-diagnostic','nanochat-feedback-backprop-debug'))
        lines += ['+InteractiveJob = true','hold = true',
            'periodic_hold = (JobStatus == 2) && (time() - JobCurrentStartDate > 5400)',
            'on_exit_hold = (ExitBySignal == true) || (ExitCode != 0)','queue 1']
        (ROOT/'debug.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dry-run.ad','debug.sub'],ROOT)
        (ROOT/'submission-intent.json').write_text(json.dumps(source,indent=2)+'\n')
        job=command(['condor_submit','-terse','debug.sub'],ROOT).split()[0]
        state=dict(status='held_for_audit',job=job,source=source,submitted_at=time.time(),formal_training=False)
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==32768
        assert ad['RequestDisk']==80*1024**2 and 'H100|H200' in ad['RequireGPUs']
        assert 'GlobalMemoryMb >= 78000' in ad['RequireGPUs']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and '.netrc' not in ad['TransferInput']
        assert ad['TransferOutput']=='results.tar' and ad['InteractiveJob']
        keys=('ClusterId','ProcId','JobStatus','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
            'RequireGPUs','TransferInput','TransferOutput','ContainerImage','Rank','PeriodicHold','OnExitHold')
        (ROOT/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job])
        state['status']='debug_submitted'
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2))


if __name__=='__main__':main()
