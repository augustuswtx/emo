import unittest

import numpy as np

from analyze_visual_utility_stability import fit_predict_mlp


class VisualUtilityStabilityAnalysisTest(unittest.TestCase):
    def test_fixed_mlp_learns_simple_nonlinear_signal(self):
        train_x = np.asarray([
            [-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0],
        ] * 16)
        train_y = np.asarray([1.0, 0.0, 0.0, 1.0] * 16)
        score = fit_predict_mlp(train_x, train_y, train_x, 7, steps=500)
        prediction = (score >= 0.5).astype(np.float64)
        self.assertGreaterEqual(float((prediction == train_y).mean()), 0.99)


if __name__ == '__main__':
    unittest.main()
