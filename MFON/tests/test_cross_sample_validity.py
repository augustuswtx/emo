import unittest

import numpy as np

from audit_cross_sample_validity import (
    pairwise_concordance,
    residualize,
    summarize,
    within_batch_summary,
)


class CrossSampleValidityTest(unittest.TestCase):
    def test_concordance_recognizes_order_and_reverse(self):
        score = np.arange(20.0)
        self.assertEqual(pairwise_concordance(score, score, max_pairs=5000), 1.0)
        self.assertEqual(pairwise_concordance(score, -score, max_pairs=5000), 0.0)

    def test_residualize_removes_linear_control(self):
        control = np.arange(10.0)
        residual = residualize(3.0 * control + 2.0, control)
        self.assertLess(np.nanmax(np.abs(residual)), 1e-10)

    def test_summary_uses_negative_loss_as_fidelity(self):
        score = np.arange(1.0, 11.0)
        loss = score[::-1]
        result = summarize('vision', score, loss, np.ones(10), np.ones(10))
        self.assertAlmostEqual(result['spearman_score_fidelity'], 1.0)
        self.assertGreater(result['pairwise_concordance'], 0.99)

    def test_within_batch_summary_never_compares_different_batches(self):
        scores = [np.array([1.0, 2.0, 3.0]), np.array([101.0, 102.0, 103.0])]
        losses = [np.array([3.0, 2.0, 1.0]), np.array([30.0, 20.0, 10.0])]
        result = within_batch_summary('infonce', scores, losses)
        self.assertEqual(result['n_batches'], 2)
        self.assertEqual(result['comparable_pairs'], 6)
        self.assertAlmostEqual(result['within_batch_spearman_mean'], 1.0)
        self.assertAlmostEqual(result['within_batch_pairwise_concordance'], 1.0)

    def test_within_batch_summary_aggregates_pair_counts(self):
        scores = [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])]
        losses = [np.array([2.0, 1.0]), np.array([1.0, 2.0, 3.0])]
        result = within_batch_summary('mixed', scores, losses)
        self.assertEqual(result['comparable_pairs'], 4)
        self.assertAlmostEqual(result['within_batch_pairwise_concordance'], 0.25)


if __name__ == '__main__':
    unittest.main()
