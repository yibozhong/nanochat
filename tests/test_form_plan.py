"""Pure-stdlib checks: safe to run on a submission node."""
import random
import unittest

from nanochat.form_plan import form_schedule, form_budget, plan_reallocation, round_growth


class FormPlanTests(unittest.TestCase):
    def test_production_schedule_and_budget(self):
        steps = form_schedule(305176)
        self.assertEqual(len(steps), 10)
        self.assertEqual(steps, [k * 305176 // 11 for k in range(1, 11)])
        self.assertEqual(form_budget(8 * 2048, 8, 512), 1216)

    def test_rounding_largest_remainder_and_ties(self):
        self.assertEqual(round_growth([33, 33, 62], 128), [64, 0, 64])
        self.assertEqual(round_growth([0, 128, 0], 128), [0, 128, 0])

    def test_global_pruning_minimum_and_retained_donors(self):
        scores = [list(range(80)), list(range(100, 180))]
        p = plan_reallocation(scores, 32, 64)
        self.assertEqual(p['prune'], [list(range(48)), list(range(16))])
        self.assertEqual(p['donor_counts'], [0, 64])
        self.assertEqual(p['growth'], [0, 64])
        self.assertEqual(p['widths_after'], [32, 128])
        for l, j in p['donors']:
            self.assertIn(j, p['keep'][l])

    def test_ten_forms_conserve_exact_budget(self):
        for randomized in (False, True):
            rng = random.Random(1337)
            widths = [2048] * 8
            for _ in range(10):
                p = plan_reallocation([[rng.gauss(0, 1) for _ in range(m)] for m in widths],
                                      512, 1216, rng if randomized else None)
                self.assertEqual(sum(map(len, p['prune'])), 1216)
                self.assertEqual(sum(p['growth']), 1216)
                self.assertTrue(all(g % 64 == 0 for g in p['growth']))
                widths = p['widths_after']
                self.assertEqual(sum(widths), 16384)
                self.assertGreaterEqual(min(widths), 512)

    def test_no_layer_upper_bound(self):
        scores = [[0.] * 256, [1.] * 256, [2.] * 256]
        p = plan_reallocation(scores, 64, 64)
        self.assertEqual(p['widths_after'], [192, 256, 320])

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):form_schedule(10)
        with self.assertRaises(ValueError):form_budget(32, 2, 16)
        with self.assertRaises(ValueError):plan_reallocation([[0.] * 64] * 2, 64, 64)
        with self.assertRaises(FloatingPointError):plan_reallocation([[float('nan')] * 128], 32, 64)


if __name__ == '__main__':
    unittest.main()
