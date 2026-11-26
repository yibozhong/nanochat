"""Offline reporting regressions. Fixtures never contact W&B or submit GPU jobs."""
import base64
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nanochat.experiment_logging import STEPS, BATCH_TOKENS, PROJECT
from research.connections.native.chord import submit as launcher
from research.connections.native.chord import run_queue
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
    def test_frozen_chord_records_take_precedence_over_missing_wandb_data(self):
        root = launcher.REPO/'research'
        rows = sync.include_archived_runs([{'run_id': '9c3d50ec'}], root)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['group'] == 'legacy-chord-1b' for r in rows))
        archived = {r['run_id']: r for r in rows}
        self.assertEqual(archived['9c3d50ec']['status'], '已完成')
        self.assertAlmostEqual(archived['9c3d50ec']['final_ce'], 3.2080763)
        self.assertEqual(archived['62357991']['trained_tokens'], 0)

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

    def test_portable_dashboard_progress_archive_and_rankings(self):
        target = STEPS*BATCH_TOKENS
        baseline = record('baseline','self',{100:4,target:3.1},complete=True)
        worse = record('worse','baseline',{100:4.1,target:3.2},complete=True)
        stopped = record('stopped','baseline',{100:4.5},job={'stopped':True,'stop_reason':'复核后停止'})
        failed = record('failed','baseline',{100:4.8},wb='failed')
        unmatched = record('different','baseline',{target:2.9},complete=True,seed=19)
        running = record('running','baseline',{100:3.95},wb='running')
        running.update(status='运行中',trained_tokens=12345,name='preview <script>unsafe()</script>')
        queued = record('queued','baseline',{},wb='',job={'ad':{'JobStatus':1}})
        queued['trained_tokens']=None
        records = [baseline,worse,stopped,failed,unmatched,running,queued]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'velocity'
            folder.mkdir()
            (folder/'REPORT.md').write_text(sync.REPORT.format(group='velocity')+
                '\n## 结论\n\n手工复核的结果。\n\n<script>alert(1)</script>\n')
            legacy = root/'legacy'
            legacy.mkdir()
            (legacy/'REPORT.md').write_text('# 历史项目不应出现在看板')
            sync.publish(records,root,synced_at='2026-11-25 12:00 CST')
            html=(root/'index.html').read_text()
            class Page(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.sections={}
                    self.current=None
                    self.images=[]
                    self.downloads={}
                def handle_starttag(self,tag,attrs):
                    a=dict(attrs)
                    if tag=='section':
                        self.current=a['id']
                        self.sections[self.current]=[]
                    if tag=='img':
                        self.images.append(a['src'])
                    if tag=='a' and 'download' in a:
                        self.downloads[a['download']]=a['href']
                    if tag=='script':
                        raise AssertionError('Unexpected executable content in static page')
                def handle_endtag(self,tag):
                    if tag=='section':
                        self.current=None
                def handle_data(self,data):
                    if self.current:
                        self.sections[self.current].append(data)
            page=Page()
            page.feed(html)
            regions={k:' '.join(v) for k,v in page.sections.items()}
            self.assertEqual(set(regions),{'overview','leaderboard','experiments','running','archive'})
            self.assertIn('手工复核的结果',regions['overview'])
            self.assertLess(regions['leaderboard'].index('baseline'),regions['leaderboard'].index('worse'))
            self.assertNotIn('different',regions['leaderboard'])
            self.assertIn('12,345 / 1,000,013,824',regions['running'])
            self.assertIn('排队',regions['running'])
            self.assertNotIn('stopped',regions['running'])
            for expected in ('stopped','复核后停止','失败','最终 CE 高于基线','手工复核的结果'):
                self.assertIn(expected,regions['archive'])
            self.assertNotIn('different',regions['archive'])
            self.assertNotIn('历史项目不应出现在看板',html)
            self.assertIn('2026-11-25 12:00 CST',html)
            self.assertTrue(page.images)
            for url in page.images:
                self.assertTrue(url.startswith('data:image/png;base64,'))
                self.assertTrue(base64.b64decode(url.split(',',1)[1]).startswith(b'\x89PNG'))
            for name in ('curves.csv','baseline.json'):
                contents=base64.b64decode(page.downloads[name].split(',',1)[1])
                source=folder/name if name.endswith('.csv') else folder/'configs'/name
                self.assertEqual(contents,source.read_bytes())

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

    def test_chord_submission_default_and_static_ablation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'HOME', Path(tmp)), \
                patch.object(launcher, 'submit') as submit:
            args = ['--group', 'residual-mechanisms', '--baseline-run', 'a2793a7e']
            launcher.main(args)
            default = submit.call_args.args[0]
            self.assertEqual(default, 'chord_next_bf16')
            flags = launcher.chord_args(default)
            self.assertEqual((launcher.CHORD_ARGS[default].parent/'chord.args').read_text().splitlines(), flags)
            self.assertIn('--chord-six-controls=learned_dynamic', flags)
            self.assertIn('--velocity-init=token', flags)
            launcher.main([*args, '--recipe', 'chord_scalar_bf16'])
            call = submit.call_args.args
            self.assertEqual(call[0], 'chord_scalar_bf16')
            static = launcher.chord_args(call[0], call[-1])
            self.assertEqual(set(flags) ^ set(static),
                             {'--chord-six-controls=learned_dynamic', '--chord-six-controls=static'})

    def test_active_chord_recipes_and_legacy_next(self):
        self.assertEqual(set(launcher.RECIPES), {
            'chord_scalar_bf16', 'chord_next_bf16', 'chord_token_sphere_only', 'chord_token_lookahead_only',
            'vanilla_baseline', 'vanilla_baseline_zero', 'native_baseline', 'vanilla_mhc', 'vanilla_attnres', 'mhc'})
        scalar = launcher.CHORD_ARGS['chord_scalar_bf16'].read_text().splitlines()
        next_args = launcher.CHORD_ARGS['chord_next_bf16'].read_text().splitlines()
        self.assertIn('--velocity-init=token', scalar)
        self.assertIn('--residual-projection-init=zero', scalar)
        self.assertIn('--chord-state-dtype=bfloat16', scalar)
        self.assertIn('--chord-soft-force-tau=0.05', scalar)
        self.assertIn('--chord-update-based', scalar)
        self.assertEqual(set(scalar) - set(next_args), {'--chord-six-controls=static'})
        self.assertEqual(set(next_args) - set(scalar), {'--chord-six-controls=learned_dynamic'})
        self.assertEqual(len(next_args), len(set(next_args)))
        legacy = launcher.CHORD_ARGS['chord_next_bf16'].parent / 'legacy/chord_next_token.args'
        self.assertEqual(legacy.read_text().splitlines(), next_args)
        data = dict(data_uri='test', manifest=dict(dataset_revision='v', tokenizer={}))
        for recipe in launcher.CHORD_ARGS:
            for init in launcher.VELOCITY_INITS:
                s = launcher.make_source(recipe, 'key', 0, 'test', 'test', 'self', 'commit', data, init)
                self.assertEqual(run_queue.manifest_runs(s, s['run_id'], recipe), [s])
                self.assertEqual([f for f in s['model_args'] if f.startswith('--velocity-init=')], ['--velocity-init='+init])
                self.assertEqual('velocity_projection_peak_lr' in s, init == 'projected_h0')
                self.assertEqual('velocity_embedding_peak_lr' in s, init == 'token')
        with tempfile.TemporaryDirectory() as tmp:
            for retired in ('chord', 'vanilla_next_soft', 'vanilla_next_soft_h0'):
                with self.assertRaisesRegex(ValueError, 'Unknown active recipe'):
                    launcher.submit(retired, Path(tmp), STEPS, 'velocity', retired, 'self')
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_suite_scheduler_uses_current_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / (PROJECT + '-residual-suite')
            (folder/'logs').mkdir(parents=True)
            runs = [dict(run_id=rid, formal_training=True, wandb_project=PROJECT) for rid in ('a','b','c')]
            (folder/'state.json').write_text(json.dumps(dict(source=dict(runs=runs), job='1.0')))
            (folder/'logs/train.1.0.out').write_text('RESIDUAL_PHASE preflight b native_baseline\n')
            ad = json.dumps([dict(ClusterId=1, ProcId=0, JobStatus=2)])
            with patch.object(sync.shutil, 'which', return_value='condor_q'), patch.object(sync.subprocess, 'check_output', return_value=ad):
                states = sync.scheduler_states(root)
            statuses = [record(rid,'self',{},wb='',job=states[rid])['status'] for rid in ('a','b','c')]
            self.assertEqual(statuses, ['排队','准备中','排队'])

    def test_suite_validates_all_before_training_and_resumes_in_order(self):
        import subprocess
        runs = [dict(run_id=key, recipe=key, group='residual-mechanisms', formal_training=True,
                     steps=STEPS, checkpoint_seconds=1800) for key in launcher.RESIDUAL_STUDY]
        report = dict(first_gradients=dict(output_projection_norms=[1.]))
        with patch.object(run_queue, 'read'), patch.object(run_queue, 'manifest_runs', return_value=runs), \
                patch.object(run_queue, 'preflight', return_value=report) as gate, patch.object(run_queue,'save'), \
                patch.object(run_queue, 'train_args', return_value=['train']), patch.object(run_queue, 'environment'), \
                patch.object(run_queue, 'finished') as finished, patch.object(run_queue.subprocess, 'run') as train:
            gate.side_effect = [report]*5 + [ValueError('last config failed')]
            with self.assertRaisesRegex(ValueError, 'last config failed'):
                run_queue.main('id', launcher.SUITE)
            train.assert_not_called()
            gate.side_effect = None
            # A completed baseline is skipped. The next entry returns a durable
            # checkpoint, so the other four may only start on a later invocation.
            finished.side_effect = [True, False]
            train.return_value = subprocess.CompletedProcess([], 85)
            self.assertEqual(run_queue.main('id', launcher.SUITE), 85)
            args = train.call_args.args[0]
            self.assertEqual(args[args.index('--run-id')+1], 'native_baseline')
            self.assertEqual(args[args.index('--resume')+1], 'auto')
            train.assert_called_once()

    def test_manifest_and_worker_agree_without_submission(self):
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
                for recipe in (*launcher.RECIPES, *launcher.SUITES):
                    root=d/recipe
                    root.mkdir()
                    with self.assertRaises(StopBeforeSubmit):
                        launcher.submit(recipe,root,STEPS,'velocity',recipe,
                                            'a2793a7e' if recipe == launcher.ABLATION_SUITE else 'self')
                    manifest=(root/'bundles/source.json').read_text()
                    s=json.loads(manifest)
                    runs = run_queue.manifest_runs(s, s['run_id'], recipe)
                    for member in runs:
                        self.assertEqual((member['schedule_steps'], member['steps'], member['wandb_project']),
                                         (STEPS, STEPS, PROJECT))
                    if recipe == launcher.SUITE:
                        self.assertEqual([m['architecture'] for m in runs],
                                         ['vanilla', 'native', 'vanilla', 'vanilla', 'vanilla', 'vanilla'])
                        self.assertEqual([m['baseline_run'] for m in runs], ['self', 'self'] + [s['run_id']]*4)
                        self.assertEqual([m['connection_variant'] for m in runs],
                                         ['baseline', 'baseline', 'attnres', 'mhc', 'baseline', 'baseline'])
                    if recipe == launcher.ABLATION_SUITE:
                        self.assertEqual([m['recipe'] for m in runs], list(launcher.ABLATION_ARGS))
                        self.assertEqual([m['baseline_run'] for m in runs], ['a2793a7e', s['run_id'], s['run_id']])
                        self.assertTrue(all(m['velocity_init'] == 'zero' for m in runs))
                        scalar = launcher.ABLATION_ARGS['chord_scalar_from_rest'].read_text().splitlines()
                        self.assertFalse(any(f.startswith('--chord-six-controls') or f=='--chord-update-based' for f in scalar))
                        self.assertEqual(runs[0]['model_args'], [f.replace('--velocity-init=token', '--velocity-init=zero') for f in scalar])
                    if recipe in launcher.CHORD_ARGS:
                        self.assertEqual(s['model_args'], launcher.CHORD_ARGS[recipe].read_text().splitlines())
                        self.assertEqual(s['velocity_init'], 'token')


if __name__ == '__main__':
    unittest.main()
