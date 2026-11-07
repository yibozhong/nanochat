"""One GPU acceptance job, including actual scheduler checkpoint/restart."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

from scripts.hash_qkv_bench import write
from scripts.hash_vo_preflight import run


def checks(root, name):
    path=root/(name+'.xml')
    run(['-m','pytest','-v','tests/test_hash_vo.py','--junitxml='+str(path)],root,name)
    suites=ET.parse(path).getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=15,failures=0,errors=0,skipped=0),counts
    return counts


def main():
    root=Path('results/hash-vo');root.mkdir(parents=True,exist_ok=True)
    source=json.loads(Path('source.json').read_text())
    report_path=root/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(
        complete=False,formal_training=False,source=source,profiles=[],phase='new')
    assert report['source']==source and not source['formal_training']
    if report['complete']:return 0
    ckpt=Path('../checkpoints')
    common=['-m','scripts.hash_vo_train','--steps','60','--schedule-steps','305176',
        '--microbatch','16','--seed','42','--attention-backend','hardware_auto',
        '--data-dir','../work/preflight-data/data','--output-dir',str(root/'restart'),
        '--checkpoint-dir',str(ckpt),'--run-id','hash-vo-worker-restart','--segment-seconds','3600']
    try:
        if report['phase']=='new':
            assert not (ckpt/'latest.pt').exists()
            report['checks']=checks(root,'checks')
            for i,variant in enumerate(('baseline','hash','hash','baseline')):
                name=f'profile-{i}-{variant}';out=root/(name+'.json')
                run(['-m','scripts.hash_vo_bench','--data','../work/preflight-data/data',
                     '--variant',variant,'--repetition',str(i),'--output',str(out)],root,name)
                p=json.loads(out.read_text())
                assert p['complete'] and p['optimizer_updates']==60 and p['trained_causality_passed']
                report['profiles'].append(p);write(report_path,report)
            assert len({p['gpu_uuid'] for p in report['profiles']})==1
            assert len({p['input_sha256'] for p in report['profiles']})==1
            run(common+['--segment-steps','25'],root,'first25',85)
            progress=json.loads((root/'restart/progress.json').read_text())
            assert progress['step']==25 and progress['source']==source
            report.update(phase='awaiting_worker_restart',first_invocation=os.environ['HASH_WORKER_INVOCATION'],
                          first_scratch=str(Path('..').resolve()))
            write(report_path,report)
            print('GATE_WORKER_CHECKPOINT_25 exit85',flush=True)
            return 85
        assert report['phase']=='awaiting_worker_restart'
        assert report['first_invocation']!=os.environ['HASH_WORKER_INVOCATION']
        assert json.loads((root/'restart/progress.json').read_text())['step']==25
        report['restart_checks']=checks(root,'restart-checks')
        run(common+['--resume','auto'],root,'resume60')
        summary=json.loads((root/'restart/summary.json').read_text())
        config=json.loads((root/'restart/config.json').read_text())
        rows=[json.loads(s) for s in (root/'restart/steps.jsonl').read_text().splitlines()]
        assert summary['complete'] and summary['steps']==60 and summary['variant']=='hash_vo'
        assert summary['source']==source and config['logging_backend']=='local'
        assert config['parameters']==121634816
        assert config['model']['hash_vo']['lookup_sha256']==source['lookup_sha256']
        assert [r['step'] for r in rows]==list(range(1,61))
        assert all(math.isfinite(r['loss']) for r in rows)
        assert math.isfinite(summary['final_dev_ce']) and summary['final_dev_ce']<summary['initial_dev_ce']
        b=sum(p['mean_ms'] for p in report['profiles'] if p['variant']=='baseline')/2
        h=sum(p['mean_ms'] for p in report['profiles'] if p['variant']=='hash')/2
        # Diagnostic weights are only scheduler checkpoint inputs, never final outputs.
        shutil.rmtree(ckpt);ckpt.mkdir()
        report.update(complete=True,phase='complete',restart=summary,baseline_ms=b,hash_vo_ms=h,
            overhead_percent=100*(h/b-1),diagnostic_weights_removed=True,
            worker_restart=dict(passed=True,first_invocation=report['first_invocation'],
                second_invocation=os.environ['HASH_WORKER_INVOCATION'],
                same_scratch=report['first_scratch']==str(Path('..').resolve())),
            limitation='Short training and recovery acceptance; not a long-run quality or stability claim.')
        write(report_path,report)
        print(json.dumps(dict(complete=True,baseline_ms=b,hash_vo_ms=h,worker_restart=report['worker_restart'])),flush=True)
        return 0
    except BaseException as error:
        report['error']=repr(error);write(report_path,report);raise


if __name__=='__main__':sys.exit(main())
