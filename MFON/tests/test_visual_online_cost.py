import unittest

import torch

from audit_visual_online_cost import positive_visual_weights


class PositiveVisualWeightsTest(unittest.TestCase):
    def test_positive_scores_get_exact_batch_budget(self):
        utility = torch.tensor([-1.0, 0.2, 0.0, 3.0])
        weights, active = positive_visual_weights(utility, base_weight=0.3)
        self.assertTrue(active)
        self.assertTrue(torch.allclose(
            weights, torch.tensor([0.0, 0.6, 0.0, 0.6])
        ))
        self.assertAlmostEqual(weights.mean().item(), 0.3)

    def test_all_nonpositive_batch_abstains(self):
        weights, active = positive_visual_weights(
            torch.tensor([-2.0, -0.1, 0.0])
        )
        self.assertFalse(active)
        self.assertTrue(torch.equal(weights, torch.zeros(3)))

    def test_nonfinite_utility_rejected(self):
        with self.assertRaises(ValueError):
            positive_visual_weights(torch.tensor([float('nan')]))


if __name__ == '__main__':
    unittest.main()
