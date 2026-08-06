import unittest

import torch

from interventional_reliability import (
    ReliabilityHead,
    TemporalReliabilityHead,
    blend_corruption,
    gaussian_corruption_pair,
    ordinal_reliability_pair,
    sample_ordered_severities,
    scale_active_content,
)


class InterventionalReliabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_ordered_severity_sampling(self):
        features = torch.randn(64, 5, 3)
        low, high = sample_ordered_severities(features, 1.5, 1.0)
        self.assertTrue(torch.all(low >= 0))
        self.assertTrue(torch.all(high >= low))
        self.assertTrue(torch.all(high <= 1.5))

    def test_disabled_corruption_has_zero_severity(self):
        features = torch.randn(8, 5, 3)
        low, high = sample_ordered_severities(features, 1.0, 0.0)
        self.assertTrue(torch.equal(low, torch.zeros_like(low)))
        self.assertTrue(torch.equal(high, torch.zeros_like(high)))

    def test_corruption_preserves_padding_and_order(self):
        features = torch.tensor([[[1.0, -1.0], [2.0, 1.0], [0.0, 0.0]]])
        low = torch.tensor([0.25])
        high = torch.tensor([0.75])
        mild, severe = gaussian_corruption_pair(features, low, high)
        self.assertTrue(torch.equal(mild[:, 2], features[:, 2]))
        self.assertTrue(torch.equal(severe[:, 2], features[:, 2]))
        mild_distance = (mild - features).norm()
        severe_distance = (severe - features).norm()
        self.assertGreater(severe_distance.item(), mild_distance.item())

    def test_head_is_invariant_to_positive_energy_scaling(self):
        head = ReliabilityHead(feature_dim=4, hidden_dim=8).eval()
        features = torch.randn(3, 6, 4)
        features[:, 4:] = 0
        factors = torch.tensor([0.5, 1.7, 3.0])
        scaled = scale_active_content(features, factors)
        self.assertTrue(torch.allclose(head(features), head(scaled), atol=2e-5))

    def test_head_is_invariant_to_padding_location(self):
        head = ReliabilityHead(feature_dim=3, hidden_dim=8).eval()
        content = torch.randn(2, 3)
        left = torch.zeros(1, 5, 3)
        right = torch.zeros(1, 5, 3)
        left[0, :2] = content
        right[0, 2:4] = content
        self.assertTrue(torch.allclose(head(left), head(right), atol=1e-7))

    def test_temporal_head_is_scale_and_padding_invariant(self):
        head = TemporalReliabilityHead(feature_dim=3, hidden_dim=8).eval()
        content = torch.randn(1, 4, 3)
        left = torch.zeros(1, 7, 3)
        right = torch.zeros(1, 7, 3)
        left[:, :4] = content
        right[:, 2:6] = content * 2.5
        self.assertTrue(torch.allclose(head(left), head(right), atol=2e-5))

    def test_corruption_blend_respects_progress(self):
        clean = torch.zeros(2, 3, 4)
        corrupted = torch.ones(2, 3, 4)
        self.assertTrue(torch.equal(blend_corruption(clean, corrupted, 0.0), clean))
        self.assertTrue(torch.equal(blend_corruption(clean, corrupted, 1.0), corrupted))
        self.assertTrue(
            torch.allclose(
                blend_corruption(clean, corrupted, 0.25),
                torch.full_like(clean, 0.25),
            )
        )

    def test_ordinal_objective_reaches_head_parameters(self):
        head = ReliabilityHead(feature_dim=3, hidden_dim=8)
        features = torch.randn(6, 7, 3)
        result = ordinal_reliability_pair(
            head,
            features,
            max_severity=1.0,
            corrupt_prob=1.0,
            margin=0.2,
            invariance_weight=0.1,
        )
        result['loss'].backward()
        gradients = [parameter.grad for parameter in head.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertEqual(result['q_high'].shape, (6,))

    def test_zero_corruption_has_no_rank_penalty(self):
        head = ReliabilityHead(feature_dim=3, hidden_dim=8)
        features = torch.randn(4, 5, 3)
        result = ordinal_reliability_pair(
            head,
            features,
            max_severity=1.0,
            corrupt_prob=0.0,
            margin=0.2,
            invariance_weight=0.0,
        )
        self.assertTrue(torch.allclose(result['corrupted'], features))
        self.assertAlmostEqual(result['rank_loss'].item(), 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
