"""One additional authorized 10B run: uncompressed start + loss warmup."""
import argparse
import fcntl
import json
from pathlib import Path
import stat
import subprocess
import time
import uuid

from submit import REPO, HOME_ROOT, ASSETS, WHEEL, command, write, stopped
import optimized_pair as shared
from optimized_preflight import VERIFIED
from warmup_preflight import PLAN, WARMUP, validate_source, validate_ready
from policy import STEPS, TOKENS

ROOT = HOME_ROOT/PLAN
MODE = 'dynamic_hca'


def launcher():
    text = shared.launcher()
    text = text.replace('research/connections/hca-train/optimized_preflight.py',
                        'research/connections/hca-train/warmup_preflight.py')
    text = text.replace('--hca-backend triton --hca-dynamic-gradient "$TASK_GRADIENT"',
                        '--hca-backend triton --hca-dynamic-gradient "$TASK_GRADIENT" '
                        '--hca-router-init uncompressed --hca-ratio-warmup-steps '+str(WARMUP))
    return text.replace(shared.PLAN, PLAN)


def initialize():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'plan.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not stopped(ROOT)
        if (ROOT/'plan.json').exists():
            return json.loads((ROOT/'plan.json').read_text())
        assert not command(['git','status','--porcelain']), 'Commit reviewed code before freezing source'
        commit = command(['git','rev-parse','HEAD'])
        # Reuse unchanged optimized kernels; new init/loss plumbing is gated
        # on the worker. This is a normal Git diff, not a content hash scan.
        assert not command(['git','diff','--name-only',VERIFIED,commit,'--',
            'nanochat/hca_triton.py','nanochat/hca_projection.py','nanochat/hca_routing.py',
            'nanochat/hca_rotary.py','research/connections/runtime','research/connections/toolchain'])
        assert not json.loads(command(['condor_q','yzhong239','-constraint',
            f'HCAPlanKey == "{PLAN}"','-json','-attributes','ClusterId,ProcId,JobStatus']) or '[]')
        data = json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length'] == 2048
        assert data['manifest']['training_steps'] == STEPS and data['manifest']['training_tokens'] == TOKENS
        assert data['manifest']['tokenizer']['vocab_size'] == 32768
        assert stat.S_IMODE((HOME_ROOT/'.netrc').stat().st_mode) == 0o600
        transport = json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        root = ROOT/MODE
        for name in ('bundles','logs','results'): (root/name).mkdir(parents=True,exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        source = dict(plan=PLAN,commit=commit,formal_training=True,attention_mode=MODE,run_id=run_id,
            steps=STEPS,training_tokens=TOKENS,seed=42,microbatch=16,batch_tokens=32768,
            initialization='seed42_from_zero_no_validation_weights',hca_backend='triton',
            unchanged_kernel_reference_commit=VERIFIED,
            data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            hca=dict(target_ratio=4,window=256,ratio_loss_weight=1,temperature=1,noise=False,
                     dynamic_gradient='confidence',router_init='uncompressed',ratio_warmup_steps=WARMUP),
            schedule='lambda(s)=min(s/2000,1); s=completed optimizer steps; first update weight=0',
            diagnostic_gate='3000 finite updates; checkpoint replay inside/after warmup; causal/cache; post-warmup mean rate')
        validate_source(source);write(root/'bundles/source.json',source)
        subprocess.run(['git','archive','--format=tar','--prefix=nanochat-src/','-o',
                        str(root/'bundles/source.tar'),commit],cwd=REPO,check=True)
        (root/'bundles'/WHEEL).symlink_to(ASSETS/'bundles'/WHEEL)
        (root/'train.sh').write_text(launcher());(root/'train.sh').chmod(0o755)
        subprocess.run(['bash','-n',str(root/'train.sh')],check=True)
        (root/'train.sub').write_text(shared.description(MODE,run_id,data['data_uri']).replace(shared.PLAN,PLAN))
        plan = dict(status='prepared',plan=PLAN,commit=commit,authorized_formal_runs=1,
            created_at=time.time(),steps_per_run=STEPS,tokens_per_run=TOKENS,
            initial_gpu_policy='Complete H100/H200 only; unchanged validated kernels',
            runs={MODE:dict(root=str(root),run_id=run_id,
                wandb_url='https://wandb.ai/yibozhong657-none/nanochat/runs/'+run_id)})
        write(ROOT/'plan.json',plan)
        return plan


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('initialize','submit','monitor','tick'));a=p.parse_args()
    if a.action == 'initialize': result=initialize()
    elif a.action == 'submit': result=shared.enqueue(MODE,plan_root=ROOT,source_validator=validate_source)
    elif a.action == 'tick': result=shared.tick(plan_root=ROOT,modes=(MODE,),ready_validator=validate_ready)
    else:
        shared.monitor(plan_root=ROOT,modes=(MODE,),ready_validator=validate_ready);result=None
    if result is not None: print(json.dumps(result,indent=2))
