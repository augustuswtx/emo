import unittest

import numpy as np

from modality_utility_stats import regression_metrics, thirds_by_baseline_utility


class ModalityUtilityTest(unittest.TestCase):
    def test_strata_are_baseline_ranked_and_equal_count(self):
        utility = np.array([2.0, -1.0, 0.0, 3.0, 1.0, 4.0])
        indices = np.arange(utility.size)
        groups, ties = thirds_by_baseline_utility(utility, indices)
        self.assertEqual(groups.tolist(), [1, 0, 0, 2, 1, 2])
        self.assertEqual(np.bincount(groups).tolist(), [2, 2, 2])
        self.assertEqual(ties, [False, False])

    def test_boundary_ties_are_reported_and_resolved_by_index(self):
        utility = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        indices = np.arange(utility.size)
        groups, ties = thirds_by_baseline_utility(utility, indices)
        self.assertEqual(groups.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(ties, [True, False])

    def test_metrics_use_per_sample_absolute_and_squared_error(self):
        metrics = regression_metrics([0, 1, 2], [0, 2, 1])
        self.assertAlmostEqual(metrics['MAE'], 2 / 3)
        self.assertAlmostEqual(metrics['Loss'], 2 / 3)
        self.assertAlmostEqual(metrics['Corr'], 0.5)


if __name__ == '__main__':
    unittest.main()
