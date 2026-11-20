"""One-run scheduler/start observer and small-result collector; never resubmits."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from policy import desired_fallback, require_gpus


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def command(args):
    return subprocess.check_output(args,text=True,timeout=45).strip()


def formal_step(lines,run_id,experiment='nonzero_residual'):
    formal=False;last=None
    marker=f'FORMAL_TRAINING_BEGIN {experiment} {run_id}'
    for line in lines:
        if line.startswith('FORMAL_TRAINING_BEGIN '):formal=line==marker
        if not formal or not line.startswith('{'):continue
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if 'step' in value and 'loss' in value and value['step']>=1 and math.isfinite(value['loss']):last=value
    return last


def collect(root,training):
    archive=root/'results'/f'train-{training["job"]}.tar'
    if not archive.exists():return False
    wanted=('summary.json','progress.json','config.json')
    with tarfile.open(archive) as tar:
        for filename in wanted:
            name='results/train/'+filename
            try:member=tar.getmember(name)
            except KeyError:continue
            assert member.isfile() and member.size<4*1024*1024
            value=json.load(tar.extractfile(member))
            if filename=='config.json':
                assert value['source']==training['source']
            else:
                assert value['run_id']==training['run_id'] and value['source']==training['source']
            save(root/filename,value)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.startswith(('results/ngpt-sm', 'results/arc-sm', 'results/connection-init-sm')) and member.name.endswith('.json'):
                assert member.isfile() and member.size<4*1024*1024
                gate=json.load(tar.extractfile(member))
                assert gate['complete'] and gate['source']==training['source']
                save(root/Path(member.name).name,gate)
    summary=root/'summary.json'
    if summary.exists():
        value=json.loads(summary.read_text())
        assert value['complete'] and value['steps']==training['source']['steps']
        assert value['training_tokens']==training['source']['training_tokens']
        ref_path = root/'comparison-reference.json'
        if ref_path.exists():
            reference = json.loads(ref_path.read_text())
            assert reference['step'] == value['steps'] and reference['dev_tokens'] == value['dev_tokens']
            comparison=dict(reference=reference, run_id=value['run_id'], step=value['steps'],
                            dev_ce=value['final_dev_ce'], delta_dev_ce=value['final_dev_ce']-reference['dev_ce'])
            if 'ngpt_lookahead' in reference:
                ngpt, combined = reference['ngpt'], reference['ngpt_lookahead']
                for control in (ngpt, combined):
                    assert control['step']==value['steps'] and control['dev_tokens']==value['dev_tokens']
                vanilla_gain = reference['dev_ce'] - value['final_dev_ce']
                ngpt_gain = ngpt['dev_ce'] - combined['dev_ce']
                comparison.update(lookahead_gain_on_vanilla=vanilla_gain, lookahead_gain_on_ngpt=ngpt_gain,
                                  additional_gain_with_ngpt=ngpt_gain-vanilla_gain)
            save(root/'comparison.json', comparison)
        return True
    return False


def main(root):
    root=Path(root).resolve()
    with (root/'scheduling.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'scheduling.json'
        state=json.loads(path.read_text()) if path.exists() else dict(transitions=[],poll_errors=0)
        while not any((root/n).exists() for n in ('STOPPED','STOPPED.json')):
            training=json.loads((root/'state.json').read_text());job=training['job']
            state.update(pid=os.getpid(),job=job,last_poll=time.time())
            try:
                attrs='ClusterId,ProcId,JobStatus,EnteredCurrentStatus,ExitCode,HoldReason,RemoteHost,NumJobStarts,RequireGPUs,NgptTrainKey'
                rows=json.loads(command(['condor_q',job,'-json','-attributes',attrs]) or '[]')
                if not rows:rows=json.loads(command(['condor_history',job,'-limit','1','-json','-attributes',attrs]) or '[]')
                if not rows:raise RuntimeError('Job absent from queue and history')
                ad=rows[0];state['last_job']=ad
                assert ad['NgptTrainKey']==root.name
                wanted=False if training['source'].get('hopper_only') else desired_fallback(ad,time.time())
                current='A100' in ad.get('RequireGPUs','')
                if wanted is not None and wanted!=current:
                    cluster,proc=map(int,job.split('.'))
                    condition=f'ClusterId=={cluster} && ProcId=={proc} && JobStatus=={ad["JobStatus"]} && EnteredCurrentStatus=={ad["EnteredCurrentStatus"]}'
                    if wanted:condition+=' && time()-EnteredCurrentStatus>=3600'
                    result=command(['condor_qedit','-constraint',condition,'RequireGPUs',require_gpus(wanted)])
                    state['transitions'].append(dict(time=time.time(),fallback=wanted,result=result))
                log=root/'logs'/f'train.{job}.out'
                experiment = 'arc_optimizer' if training['source'].get('matrix_optimizer') == 'arc' else 'nonzero_residual'
                latest=formal_step(log.read_text(errors='replace').splitlines(),training['run_id'],experiment) if log.exists() else None
                if latest:
                    state['latest_formal_step']=latest
                    if 'first_formal_step' not in state:
                        state['first_formal_step']=latest
                        save(root/'startup-observation.json',dict(job=job,run_id=training['run_id'],first_formal_step=latest,observed_at=time.time()))
                    status='resume_queued' if ad['JobStatus']==1 else 'training_started'
                    training.update(status=status,latest_observed_step=latest)
                else:training['status']='training_submitted' if ad['JobStatus']==1 else 'preparing'
                state['status']='monitoring'
                if ad['JobStatus']==1:
                    collect(root,training)
                if ad['JobStatus'] in (3,5) or ad['JobStatus']==4 and ad.get('ExitCode')!=0:
                    training.update(status='needs_attention',last_job=ad)
                    state['status']='needs_attention'
                    collect(root,training)
                elif ad['JobStatus']==4:
                    if collect(root,training):
                        state['status']='completed';training['status']='completed'
                    else:raise RuntimeError('Completed job missing final results')
                state['poll_errors']=0
                save(root/'state.json',training);save(path,state)
            except Exception as exc:
                state.update(poll_errors=state.get('poll_errors',0)+1,last_error=repr(exc));save(path,state)
                if state['poll_errors']>=3:
                    state['status']='needs_attention';save(path,state);return
            if state.get('status') in ('needs_attention','completed'):return
            time.sleep(60)


if __name__=='__main__':main(sys.argv[1])
