"""Submit-node regressions for the historical experiment monitors."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from research.connections import gpu_policy, train_monitor


CONNECTIONS = Path(__file__).resolve().parents[1] / 'research/connections'


class TrainMonitorTest(unittest.TestCase):
    def test_marker_filters_and_first_or_last_step(self):
        for folder in ('latent-feedback', 'layer-feedback', 'skip-scale', 'attnres-scale',
                       'scaled-hc-train', 'scaled-hc-all-train'):
            with self.subTest(folder=folder):
                variant = folder.removesuffix('-train').replace('-', '_')
                startup_only = folder.endswith('-train')
                feedback = folder.endswith('-feedback')
                entry = runpy.run_path(str(CONNECTIONS/folder/('watch_start.py' if startup_only else 'watch_train.py')))
                read = entry['formal_step' if feedback else 'first_formal_step']
                lines = ['{"step":1,"loss":7}', f'FORMAL_TRAINING_BEGIN {variant} other',
                         '{"step":2,"loss":5}', 'FORMAL_TRAINING_BEGIN unrelated run', '{"step":3,"loss":4}']
                self.assertIsNone(read(lines, 'run'))
                lines += [f'FORMAL_TRAINING_BEGIN {variant} run', '{"step":1,"loss":NaN}', '{partial',
                          '{"step":4,"loss":3}', '{"step":5,"loss":2}', f'FORMAL_TRAINING_BEGIN {variant} other',
                          '{"step":6,"loss":1}']
                self.assertEqual(read(lines, 'run')['step'], 5 if feedback else 4)

    def test_old_logs_without_a_start_marker(self):
        lines = ['BEGIN preflight', '{"step":1,"loss":4}', '{"preflight":true,"complete":true}',
                 '{"step":2,"loss":3}']
        self.assertEqual(train_monitor.first_step_after_preflight(lines)['step'], 2)
        self.assertEqual(train_monitor.first_unmarked_step(lines)['step'], 1)

    def test_gpu_fallback_only_during_the_same_idle_period(self):
        for policy, blackwell in [(gpu_policy.scale_gpus, True), (gpu_policy.feedback_gpus, False)]:
            ad = dict(JobStatus=1, EnteredCurrentStatus=100, RequireGPUs=policy(False))
            with patch.object(train_monitor.subprocess, 'check_output', return_value='ok') as command:
                with patch('time.time', return_value=3699):
                    self.assertIsNone(train_monitor._change_fallback('3.0', ad, policy))
                    command.assert_not_called()
                with patch('time.time', return_value=3700):
                    self.assertTrue(train_monitor._change_fallback('3.0', ad, policy)['fallback'])
                args = command.call_args.args[0]
                self.assertIn('EnteredCurrentStatus==100 && time()-EnteredCurrentStatus>=3600', args[2])
                self.assertEqual(args[-1], policy(True))
                self.assertEqual('Blackwell' in args[-1], blackwell)

    def test_wrong_job_identity_never_edits_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = dict(job='3.0', run_id='run', source={'model_variant': 'latent_feedback'})
            (root/'state.json').write_text(json.dumps(initial))
            other = dict(JobStatus=1, EnteredCurrentStatus=0, LatentFeedbackTrainKey='another-job')
            with patch.object(train_monitor, '_job_ad', return_value=other), patch('time.sleep'), \
                    patch.object(train_monitor, '_change_fallback') as edit:
                entry = runpy.run_path(str(CONNECTIONS/'latent-feedback/watch_train.py'))
                entry['main'](root)
                edit.assert_not_called()
            self.assertEqual(json.loads((root/'scheduling.json').read_text())['status'], 'needs_attention')
            self.assertEqual(json.loads((root/'state.json').read_text()), initial)


if __name__ == '__main__':
    unittest.main()
