import unittest

import numpy as np

from analyze_visual_utility_predictor import (
    average_precision,
    evaluate,
    fit_logistic,
    predict,
    roc_auc,
    spearman,
    split_groups,
)


class VisualUtilityPredictorTest(unittest.TestCase):
    def test_metrics_are_exact_for_perfect_ordering(self):
        y = np.asarray([0.0, 1.0, 0.0, 1.0])
        utility = np.asarray([-2.0, 1.0, -1.0, 2.0])
        score = np.asarray([0.1, 0.8, 0.2, 0.9])
        self.assertEqual(roc_auc(y, score), 1.0)
        self.assertEqual(average_precision(y, score), 1.0)
        self.assertAlmostEqual(spearman(score, utility), 1.0)
        result = evaluate(y, utility, score)
        self.assertEqual(result['top_quarter_harmful_rate'], 0.0)

    def test_logistic_baseline_learns_separable_signal(self):
        x = np.asarray([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = fit_logistic(x, y, 1000, 0.1, 1e-3)
        score = predict(model, x)
        self.assertGreater(roc_auc(y, score), 0.99)

    def test_split_keeps_complete_groups(self):
        groups = [{'pair_number': i} for i in range(10)]
        train, test = split_groups(groups, 0.7, 7)
        train_ids = {group['pair_number'] for group in train}
        test_ids = {group['pair_number'] for group in test}
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(len(train) + len(test), 10)


if __name__ == '__main__':
    unittest.main()

