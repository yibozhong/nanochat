"""Offline reporting regressions. Fixtures never contact W&B or submit GPU jobs."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nanochat.experiment_logging import STEPS, BATCH_TOKENS, PROJECT
from research.connections.native.chord import submit as launcher
from scripts import research_sync as sync


def record(rid, baseline, points, *, complete=False, seed=42, wb='finished', job=None):
    target = STEPS * BATCH_TOKENS
    config = dict(source=dict(baseline_run=baseline, commit='fixture', seed=seed),
                  protocol=dict(steps=STEPS,batch_tokens=BATCH_TOKENS,seed=seed),
                  comparison_protocol=dict(seed=seed,architecture='vanilla',width=512,depth=8,context=2048))
    s = dict(complete=complete, tokens=max(points,default=0))
    if complete:
        s.update(training_tokens=target, final_dev_ce=points[target], training_seconds=1000)
    return sync.make_record(rid,rid,'velocity',wb,config,s,points,job or {})


class ReportsTest(unittest.TestCase):
    def test_matching_and_closed_group_review(self):
        end = STEPS*BATCH_TOKENS
        self.assertEqual(end,1_000_013_824)
        records = [record('base','self',{0:9,100:4,end:3.5},complete=True),
                   record('candidate','base',{100:3.9,end:3.4},complete=True),
                   record('stopped','base',{100:3.8},wb='crashed',job={'stopped':True}),
                   record('mismatch','base',{end:2},complete=True,seed=101),
                   record('unmatched','base',{110:3.8},wb='failed')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def reviewer(folder, stamp, rows):
                with (folder/'REPORT.md').open('a') as target:
                    target.write(f'\n<!-- reviewed: {stamp} -->\n\n**实测结论** 仅同协议排名。\n')
                return True
            sync.publish(records,root,reviewer)
            before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            sync.publish(records,root,lambda *a: self.fail('Duplicate review'))
            self.assertEqual(before,{p:p.read_bytes() for p in before})
            self.assertTrue(list((root/'velocity').glob('curves-*.png')))
            self.assertEqual(len(list(csv.DictReader((root/'velocity/curves.csv').read_text().splitlines()))),8)
            leaderboard = (root/'LEADERBOARD.md').read_text()
            self.assertIn('candidate',leaderboard)
            self.assertNotIn('mismatch',leaderboard)
            self.assertNotIn('stopped',leaderboard)
            self.assertIn('已复核',(root/'OVERVIEW.md').read_text())
            self.assertAlmostEqual(records[1]['delta_ce'],-.1)
            self.assertAlmostEqual(records[2]['delta_ce'],-.2)
            self.assertIsNone(records[2]['final_ce'])
            self.assertEqual(records[2]['status'],'主动终止')
            self.assertIsNone(records[4]['delta_ce'])

    def test_resumable_segment_and_absent_metadata(self):
        r=record('segment','self',{50:4},job={'ad':{'JobStatus':1}})
        self.assertEqual(r['status'],'等待恢复')
        self.assertIsNone(r['final_ce'])
        r=sync.make_record('other','other',None,'crashed',{}, {}, {}, {})
        self.assertEqual((r['group'],r['status']),('unassigned','失败'))
        self.assertFalse(r['protocol_id'])

    def test_only_queries_new_project_and_caches_complete_curves(self):
        target = STEPS*BATCH_TOKENS
        summary = dict(complete=True, training_tokens=target, final_dev_ce=3.1, _step=10)
        cfg = dict(source=dict(baseline_run='self'),protocol=dict(steps=STEPS,batch_tokens=BATCH_TOKENS))
        run = SimpleNamespace(id='newrun', name='baseline',group='velocity', state='finished',config=cfg,summary=summary)
        def scan(**kwargs):
            self.assertEqual(kwargs['keys'],['tokens','dev/ce'])
            return [{'tokens':0,'dev/ce':9.5},{'tokens':target,'dev/ce':3.1}]
        run.scan_history = scan
        def runs(path,per_page):
            self.assertEqual(path,sync.ENTITY+'/nanochat-1b')
            return [run]
        fake=SimpleNamespace(Api=lambda timeout:SimpleNamespace(runs=runs))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sync.sys.modules,wandb=fake):
            root=Path(tmp)
            legacy=root/'legacy-run'
            legacy.mkdir()
            (legacy/'state.json').write_text('this directory must never be opened')
            self.assertEqual(sync.fetch(root,root/'cache')[0]['final_ce'],3.1)
            run.scan_history=lambda **kwargs:self.fail('Finished history should be cached')
            self.assertEqual(sync.fetch(root,root/'cache')[0]['validation_tokens'],target)

    def test_manifest_and_worker_agree_without_submission(self):
        import subprocess
        class StopBeforeSubmit(Exception): pass
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            assets=d/'assets'
            (assets/'bundles').mkdir(parents=True)
            (assets/'bundles/transformer_engine_torch-test.whl').touch()
            (assets/'gate.sub').write_text('queue 1\n')
            (d/'datasets/fineweb-edu-10bt').mkdir(parents=True)
            (d/'datasets/fineweb-edu-10bt/current-packed-cache.json').write_text(json.dumps(
                dict(ready=True,manifest=dict(training_steps=305176,dataset_revision='v',tokenizer={}),data_uri='test')))
            def fake(*args,**kwargs):
                if args[:2]==('condor_submit','-dry-run'): raise StopBeforeSubmit
                if args[0]=='condor_submit': self.fail('Real submission attempted')
                return 'commit' if args[:2]==('git','rev-parse') else ''
            with patch.object(launcher,'run',fake), patch.object(launcher,'HOME',d), patch.object(launcher,'ASSETS',assets):
                for recipe in launcher.RECIPES:
                    root=d/recipe
                    root.mkdir()
                    with self.assertRaises(StopBeforeSubmit):
                        launcher.submit(recipe,root,STEPS,'velocity',recipe,'self')
                    manifest=(root/'bundles/source.json').read_text()
                    s=json.loads(manifest)
                    self.assertEqual((s['schedule_steps'],s['steps'],s['wandb_project']),(STEPS,STEPS,PROJECT))
                    (root/'source.json').write_text(manifest)
                    worker=(root/'train.sh').read_text().split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
                    subprocess.run([sync.sys.executable,'-c',worker,s['run_id'],recipe],cwd=root,check=True)


if __name__ == '__main__':
    unittest.main()
