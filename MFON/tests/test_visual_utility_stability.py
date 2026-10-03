import unittest

import torch

from audit_visual_utility_stability import stability_columns


class VisualUtilityStabilityTest(unittest.TestCase):
    def test_stability_columns_distinguish_stable_and_unstable_samples(self):
        utility = torch.tensor([
            [1.0, 1.0, -1.0],
            [2.0, -1.0, -2.0],
            [3.0, 1.0, -3.0],
        ])
        result = stability_columns(utility)
        self.assertEqual(result['positive_fraction'][0].item(), 1.0)
        self.assertAlmostEqual(result['positive_fraction'][1].item(), 2.0 / 3.0)
        self.assertEqual(result['positive_fraction'][2].item(), 0.0)
        self.assertEqual(result['majority_fraction'][0].item(), 1.0)
        self.assertAlmostEqual(result['majority_fraction'][1].item(), 2.0 / 3.0)
        self.assertEqual(result['majority_fraction'][2].item(), 1.0)
        self.assertEqual(result['first_matches_mean'].tolist(), [True, True, True])
        self.assertAlmostEqual(result['context_sign_agreement'][0].item(), 1.0)
        self.assertAlmostEqual(result['context_sign_agreement'][1].item(), 1.0 / 3.0)

    def test_requires_multiple_contexts(self):
        with self.assertRaises(ValueError):
            stability_columns(torch.ones(1, 3))


if __name__ == '__main__':
    unittest.main()
