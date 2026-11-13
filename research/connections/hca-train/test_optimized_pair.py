"""AP-safe checks of the new authorization, gate and launch contracts."""
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from optimized_preflight import (PLAN, VERIFIED, STEPS, TOKENS, validate_source,
                                 validate_stability, validate_ready, validate_cache)
from optimized_pair import launcher, description


def source(mode='dynamic_hca'):
    return dict(plan=PLAN, commit='test-commit', run_id='test-run', formal_training=True,
                attention_mode=mode, steps=STEPS, training_tokens=TOKENS, seed=42,
                microbatch=16, batch_tokens=32768, initialization='seed42_from_zero_no_validation_weights',
                hca_backend='triton', verified_runtime_commit=VERIFIED,
                hca=dict(target_ratio=4, window=256, ratio_loss_weight=1, temperature=1,
                         noise=False, dynamic_gradient='confidence' if mode == 'dynamic_hca' else 'dmc'))


class Contracts(unittest.TestCase):
    def test_source_rejects_legacy_or_budget_drift(self):
        for mode in ('fixed_hca', 'dynamic_hca'):
            validate_source(source(mode))
        for key, val in (('steps', STEPS+1), ('training_tokens', TOKENS+32768),
                         ('seed', 43), ('microbatch', 8), ('plan', 'hca-10b-20261112')):
            s = source(); s[key] = val
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_source(s)
        for key, val in (('target_ratio', 16), ('noise', True), ('dynamic_gradient', 'dmc')):
            s = source(); s['hca'][key] = val
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_source(s)

    def test_rate_is_mean_constraint_not_per_chunk_quota(self):
        s = source()
        report = dict(complete=True, formal_training=False, status='finite_limit', source=s, steps=2000,
            args=dict(mode='dynamic_hca', gradient='confidence', block_size=4, ratio_weight=1,
                      seed=42, resume=None, save_state=True),
            validation=[dict(step=2000, tokens=131072, ce=4., hard_keep_rate=.25)])
        rows = [dict(step=i+1, loss=5., ce=4., objective=5., hard_keep_rate=.1 if i%2 else .4,
                     soft_keep_rate=.3, seconds=.1) for i in range(2000)]
        result = validate_stability(report, rows, s)
        self.assertAlmostEqual(result['train_last500_summary_compression'], 4.)
        bad = copy.deepcopy(rows); bad[500]['loss'] = float('nan')
        with self.assertRaises(AssertionError): validate_stability(report, bad, s)
        bad = copy.deepcopy(rows)
        for row in bad[-500:]: row['hard_keep_rate'] = .0625
        with self.assertRaises(AssertionError): validate_stability(report, bad, s)
        report['validation'][-1]['hard_keep_rate'] = .0625
        with self.assertRaises(AssertionError): validate_stability(report, rows, s)

    def test_cache_failure_cannot_authorize_training(self):
        s = source()
        r = dict(complete=True, status='passed', formal_training=False, source=s, checkpoint_source=s,
            step=2000, transplanted_weights=False, causal_prefix_exact=True, cache_replay_exact=True,
            cache_replay_stats_equal=True, cache_replay_tensors_exact=True, hidden_finite=True,
            hidden_relative_l2=0., hidden_normalized_peak=0.,
            layers=[dict(boundary_mismatches=0, open_chunk_tokens=7, full_open_chunk_tokens=7) for _ in range(8)])
        validate_cache(r, s)
        r['layers'][2]['open_chunk_tokens'] = 0
        with self.assertRaises(AssertionError): validate_cache(r, s)

    def test_startup_marker_is_bound_to_run_and_source(self):
        s = source()
        r = dict(complete=True, formal_training=False, source=copy.deepcopy(s), microbatch=16, capability=[9, 0],
            verified_runtime_commit=VERIFIED, steady_step_seconds=.1, diagnostic_steps=2000,
            replay_passed=True, cache_checks_passed=2,
            train_last500_summary_compression=4., dev_summary_compression=4.)
        validate_ready(r, s)
        for key, val in (('run_id', 'another-run'), ('commit', 'another-commit')):
            changed = copy.deepcopy(s); changed[key] = val
            with self.subTest(key=key), self.assertRaises(AssertionError): validate_ready(r, changed)
        r['cache_checks_passed'] = 1
        with self.assertRaises(AssertionError): validate_ready(r, s)

    def test_wrapper_reuses_checkpoint_transport_and_fresh_formal_process(self):
        script = launcher()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'train.sh'; p.write_text(script)
            subprocess.run(['bash', '-n', str(p)], check=True)
        self.assertIn('exit 74', script)
        self.assertIn('trap checkpoint_archives_on_exit EXIT', script)
        self.assertIn('--resume auto --segment-seconds 1800', script)
        self.assertIn('--hca-block-size 4', script)
        self.assertIn('--no-hca-noise', script)
        self.assertLess(script.index('optimized_preflight.py'), script.index('export WANDB_MODE=online'))
        self.assertLess(script.index('export WANDB_MODE=online'), script.index('FORMAL_TRAINING_BEGIN'))
        self.assertNotIn('--resume ../diagnostic', script)
        self.assertNotIn('hca_root_cause_v1', script)

    def test_submit_description_is_independent_and_held(self):
        for mode in ('fixed_hca', 'dynamic_hca'):
            sub = description(mode, 'test-run', 'osdf:///existing-data.tar')
            self.assertIn('arguments = '+mode+' test-run', sub)
            self.assertIn('hold = true', sub)
            self.assertIn('checkpoint_exit_code = 85', sub)
            self.assertIn('transfer_checkpoint_files = checkpoints.tar,results.tar', sub)
            self.assertIn('container_image = file:///staging/', sub)
            self.assertIn('encrypt_input_files = /home/yzhong239/.netrc', sub)
            self.assertIn('+HCAPlanKey = "'+PLAN+'"', sub)
            self.assertNotIn('validated-shared.json', sub)
            self.assertNotIn('core-evidence.json', sub)


if __name__ == '__main__':
    unittest.main()
