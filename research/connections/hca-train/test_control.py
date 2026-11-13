"""AP-safe stdlib tests for the conditional submission boundary (no torch)."""
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

import controller
import bridge
import b4
from policy import MODES, STEPS, TOKENS, CORE_SOURCE, TEST_COUNTS, validate_shared, first_formal_step
from submit import submission_text


def report():
    modes = {}
    for mode in MODES:
        modes[mode] = dict(complete=True, formal_training=False, status='passed', mode=mode,
            microbatch=4, commit='frozen', steps=3, replayed_step=2, replay_model_equal=True,
            replay_optimizer_equal=True, all_gradients_finite=True, peak_reserved_bytes=1000000,
            parameters=60817408 if mode == 'fixed_hca' else 60821512,
            losses=[10., 9., 8.], steady_step_seconds=2., pool_gradient_nonzero=True,
            router_gradient_nonzero=True, gpu_uuid='same-device', estimated_train_only_hours=10.,
            config=dict(attention_mode=mode, variant='baseline', architecture='vanilla',
                depth=8, width=512, heads=4, sequence_length=2048, vocab_size=32768, streams=1,
                hca_block_size=16, hca_window=256, hca_ratio_loss_weight=1., hca_temperature=1.,
                hca_noise=True, activation_checkpoint=False, hca_backend='triton'),
            restore_contract='exact_state_rng_data_and_fixed_cce_gradient_replay',
            loaded_model_equal=True, loaded_optimizer_equal=True, loaded_rng_equal=True,
            loaded_data_row_equal=True, actual_cce_replay_loss_equal=True,
            controlled_backward_loss_equal=True, controlled_backward_model_equal=True,
            controlled_backward_optimizer_equal=True, controlled_backward_rng_equal=True,
            cce_replay_microbatches=4)
    return dict(complete=True, formal_training=False, commit='frozen', capability=[9, 0],
        core_tests=TEST_COUNTS.copy(), core_source='frozen', comparison_complete=True,
        hca_backend='triton', kernel_tests=dict(tests=24, failures=0, errors=0, skipped=0),
        speed=dict(complete=True,formal_training=False,source=dict(commit='frozen'),gpu_uuid='same-device',microbatch=4,
                   modes={m:dict(relative_to_baseline=2.) for m in MODES}),
        microbatch=4, modes=modes, attention_mode='fixed_hca', run_id='fixed-id')


class ControlTests(unittest.TestCase):
    def test_b4_requires_its_own_ratio_recipe_and_evidence(self):
        original=report();probe=original['modes']['fixed_hca']
        probe['microbatch']=16;probe['cce_replay_microbatches']=1
        probe['config']['hca_block_size']=4
        ready=dict(complete=True,formal_training=False,commit='frozen',attention_mode='fixed_hca',
            block_size=4,microbatch=16,hca_backend='triton',tests=dict(tests=3,failures=0,errors=0,skipped=0),
            probe=probe,speed=dict(complete=True,formal_training=False,source=dict(commit='frozen'),microbatch=16,
                modes=dict(full={},fixed_hca=dict(config=dict(hca_block_size=4),relative_to_baseline=2.))))
        self.assertEqual(b4.validate(ready,'frozen'),16)
        probe['config']['hca_block_size']=16
        with self.assertRaises(AssertionError):b4.validate(ready,'frozen')

    def bridge_fixture(self, root, *, exit_code=85, best=True):
        results = root/'results/debug-123.0.tar'; results.parent.mkdir(exist_ok=True)
        checkpoints = root/'results/debug-123.0-checkpoints.tar'
        source = dict(commit='frozen', attention_mode='fixed_hca', run_id='fixed-id')
        state = dict(job='123.0', run_id='fixed-id', source=source, microbatch=4,
                     bridge=dict(debug_root=str(root), resume_job='456.0', request='request-formal.sh'))
        progress = dict(run_id='fixed-id', source=source, step=100, tokens=100*32768,
                        best_dev_ce=3., training_seconds=10.)
        config = dict(source=source, logging_backend='wandb',
                      model=dict(attention_mode='fixed_hca', hca_backend='triton'),
                      protocol=dict(microbatch=4, seed=42, steps=STEPS, schedule_steps=STEPS))
        entries = {'results/debug-requests/request-formal.sh.exit': str(exit_code).encode(),
                   'results/train/progress.json': json.dumps(progress).encode(),
                   'results/train/config.json': json.dumps(config).encode(),
                   'results/source.json': json.dumps(source).encode(),
                   'results/hca-startup/passed.json': json.dumps(report()).encode()}
        with tarfile.open(results, 'w') as tf:
            for name, raw in entries.items():
                member = tarfile.TarInfo(name); member.size = len(raw)
                tf.addfile(member, io.BytesIO(raw))
        with tarfile.open(checkpoints, 'w') as tf:
            for name in (('latest.pt', 'best.pt') if best else ('latest.pt',)):
                member = tarfile.TarInfo('checkpoints/'+name); member.size = 4
                tf.addfile(member, io.BytesIO(b'test'))
        return results, checkpoints, state

    def test_bridge_rejects_failure_or_missing_checkpoint_before_release(self):
        for kwargs in (dict(exit_code=1), dict(best=False)):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, _, state = self.bridge_fixture(root, **kwargs)
                with patch.object(bridge.subprocess, 'run') as command:
                    with self.assertRaises((AssertionError, KeyError)):
                        bridge.handoff(root, state, dict(JobStatus=4, ExitCode=0))
                    command.assert_not_called()

    def test_bridge_verifies_archives_and_preserves_container_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); results, checkpoints, state = self.bridge_fixture(root)
            evidence = bridge.validate_archives(results, checkpoints, state)
            self.assertEqual(evidence['progress']['step'], 100)
            (root/'bundles').mkdir(); (root/'logs').mkdir()
            (root/'bundles/source.json').write_text(json.dumps(state['source']))
            held = dict(JobStatus=5, HoldReason='User requested interactive debug; repair',
                        HCARunId='fixed-id', HCATrainMode='fixed_hca', RequestGPUs=1, RequestCpus=2,
                        RequestMemory=32768, RequestDisk=80*1024**2, SuccessCheckpointExitCode=85,
                        TransferCheckpoint='results.tar,checkpoints.tar', RequireGPUs='H100|H200 78000',
                        EncryptInputFiles='/home/user/.netrc',
                        TransferInput='bundles/source.tar,file:///staging/nanochat-uv-v4.sif')
            updated = dict(held, Cmd=str(root/'bridge-resume.sh'),
                           TransferInput=held['TransferInput']+',bundles/bridge-results.tar,bundles/bridge-checkpoints.tar,train.sh')
            with patch.object(bridge.subprocess, 'check_output', side_effect=[json.dumps([held]), json.dumps([updated])]), \
                 patch.object(bridge.subprocess, 'run') as command:
                bridge.handoff(root, state, dict(JobStatus=4, ExitCode=0))
                self.assertEqual(command.call_args.args[0], ['condor_release', '456.0'])
            self.assertEqual(state['job'], '456.0'); self.assertNotIn('bridge', state)
            self.assertEqual(state['durable_step'], 100)
            self.assertTrue((root/'bundles/bridge-checkpoints.tar').is_symlink())
            self.assertIn('[ ! -f checkpoints.tar ] && [ ! -f results.tar ]', (root/'bridge-resume.sh').read_text())

    def test_both_full_shape_reports_required(self):
        self.assertEqual(validate_shared(report(), 'frozen'), 4)
        for key, value in [('complete', False), ('formal_training', True), ('commit', 'other'),
                           ('capability', [8, 0]), ('comparison_complete', False), ('microbatch', 3)]:
            with self.subTest(key=key):
                r = report(); r[key] = value
                with self.assertRaises(AssertionError): validate_shared(r, 'frozen')
        r = report(); del r['modes']['dynamic_hca']
        with self.assertRaises(AssertionError): validate_shared(r, 'frozen')

    def test_gradients_restore_capacity_and_equal_recipe_required(self):
        for key, value in [('controlled_backward_model_equal', False), ('controlled_backward_optimizer_equal', False),
                           ('loaded_model_equal', False), ('loaded_rng_equal', False),
                           ('actual_cce_replay_loss_equal', False), ('controlled_backward_rng_equal', False),
                           ('cce_replay_microbatches', 1),
                           ('all_gradients_finite', False), ('pool_gradient_nonzero', False),
                           ('router_gradient_nonzero', False), ('losses', [10., float('nan'), 9.]),
                           ('peak_reserved_bytes', 1000*1024**3), ('gpu_uuid', 'different-device'),
                           ('microbatch', 8), ('parameters', 1), ('steps', 2)]:
            with self.subTest(key=key):
                r = report(); r['modes']['dynamic_hca'][key] = value
                with self.assertRaises(AssertionError): validate_shared(r, 'frozen')

    def test_diagnostics_not_formal(self):
        lines = ['{"step": 1, "loss": 9}', 'FORMAL_TRAINING_BEGIN dynamic_hca other', '{"step": 2, "loss": 8}']
        self.assertIsNone(first_formal_step(lines, 'fixed_hca', 'fixed-id'))
        lines += ['FORMAL_TRAINING_BEGIN fixed_hca fixed-id', '{"step": 0, "loss": 8}', '{"step": 1, "loss": 7}']
        self.assertEqual(first_formal_step(lines, 'fixed_hca', 'fixed-id')['step'], 1)

    def test_incremental_log_success_and_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'logs').mkdir()
            state = dict(job='123.0', run_id='fixed-id', source=dict(attention_mode='fixed_hca'), formal_started=False)
            event = json.dumps(dict(event='HCA_STARTUP_PASSED', run_id='fixed-id', attention_mode='fixed_hca', report=report()))
            path = root/'logs/train.123.0.out'
            path.write_text('{"step": 1, "loss": 10}\n'+event+'\nFORMAL_TRAINING_BEGIN fixed_hca fixed-id\n{"step": 1')
            with patch.object(controller, 'ROOT', root):
                controller.consume(root, state, dict(commit='frozen'))
                self.assertFalse(state['formal_started']); self.assertTrue(state['startup_passed'])
                self.assertTrue((root/'validated-shared.json').exists())
                with path.open('a') as f: f.write(', "loss": 8}\n')
                controller.consume(root, state, dict(commit='frozen'))
                self.assertTrue(state['formal_started'])
                self.assertEqual(state['first_formal_step']['loss'], 8)

    def test_failed_or_wrong_run_cannot_unlock_dynamic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'logs').mkdir(); path = root/'logs/train.123.0.out'
            state = dict(job='123.0', run_id='fixed-id', source=dict(attention_mode='fixed_hca'), formal_started=False)
            path.write_text('HCA_PREFLIGHT_FAILED fixed_hca fixed-id exit=1\n')
            with patch.object(controller, 'ROOT', root):
                controller.consume(root, state, dict(commit='frozen'))
                self.assertEqual(state['status'], 'needs_attention')
                self.assertFalse((root/'validated-shared.json').exists())
                state.pop('stdout_offset')
                path.write_text(json.dumps(dict(event='HCA_STARTUP_PASSED', run_id='wrong', attention_mode='fixed_hca', report=report()))+'\n')
                with self.assertRaises(AssertionError): controller.consume(root, state, dict(commit='frozen'))
                self.assertFalse((root/'validated-shared.json').exists())

    def test_failed_fixed_never_submits_dynamic_even_with_saved_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixed = root/'fixed_hca'; fixed.mkdir()
            state = dict(status='needs_attention', job='123.0', formal_started=False)
            (fixed/'state.json').write_text(json.dumps(state))
            (root/'validated-shared.json').write_text(json.dumps(report()))
            plan = dict(commit='frozen', runs={m: dict(root=str(root/m)) for m in MODES})
            (root/'plan.json').write_text(json.dumps(plan))
            with patch.object(controller, 'ROOT', root), patch.object(controller, 'enqueue') as enqueue:
                self.assertFalse(controller.tick()); enqueue.assert_not_called()

    def test_submit_transport_and_conditional_input(self):
        for mode in MODES:
            text = submission_text(mode, 'testid', 'osdf:///data.tar')
            self.assertIn('hold = true', text)
            self.assertIn('checkpoint_exit_code = 85', text)
            self.assertIn('transfer_checkpoint_files = checkpoints.tar,results.tar', text)
            self.assertIn('encrypt_input_files = /home/yzhong239/.netrc', text)
            self.assertIn('request_disk = 80GB', text)
            self.assertIn('request_memory = 32GB', text)
            self.assertEqual('bundles/validated-shared.json' in text, mode == 'dynamic_hca')
            self.assertEqual(text.count('queue 1'), 1)
            self.assertNotIn('periodic_release', text)

    def test_collect_transferred_final_results_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'results').mkdir()
            state = dict(job='123.0', run_id='fixed-id', source=dict(attention_mode='fixed_hca'), microbatch=4)
            summary = dict(complete=True, steps=STEPS, training_tokens=TOKENS, run_id=state['run_id'],
                source=state['source'], test_evaluated=True, final_dev_ce=2., best_dev_ce=2.,
                final_test_ce=2., best_selected_test_ce=2., checkpoint_files={n: dict(bytes=4) for n in ('latest.pt', 'best.pt')})
            config = dict(model=dict(attention_mode='fixed_hca'), protocol=dict(microbatch=4, seed=42))
            with tarfile.open(root/'results/train-123.0.tar', 'w') as tf:
                for name, value in [('summary', summary), ('config', config)]:
                    raw = json.dumps(value).encode(); member = tarfile.TarInfo('results/train/'+name+'.json'); member.size = len(raw)
                    tf.addfile(member, io.BytesIO(raw))
            with tarfile.open(root/'results/train-123.0-checkpoints.tar', 'w') as tf:
                for name in ('latest.pt', 'best.pt'):
                    member = tarfile.TarInfo('checkpoints/'+name); member.size = 4
                    tf.addfile(member, io.BytesIO(b'test'))
            controller.collect(root, state)
            self.assertEqual(state['status'], 'complete')
            self.assertTrue((root/'reports/train/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
