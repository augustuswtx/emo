import unittest

import torch

from audit_model_quality import (
    active_rms,
    active_steps,
    add_active_gaussian_noise,
    drop_active_timesteps,
    mask_active_contiguous_block,
    parse_severities,
    remove_active_modality,
    shift_active_timesteps,
    spearman,
)


class QualityAuditTest(unittest.TestCase):
    def test_padding_is_not_corrupted(self):
        features = torch.tensor([[[1.0, -1.0], [0.0, 0.0]]])
        generator = torch.Generator().manual_seed(7)
        noisy = add_active_gaussian_noise(features, 1.0, generator)
        self.assertTrue(torch.equal(noisy[:, 1], features[:, 1]))
        self.assertFalse(torch.equal(noisy[:, 0], features[:, 0]))

    def test_zero_severity_is_identity_copy(self):
        features = torch.randn(2, 3, 4)
        generator = torch.Generator().manual_seed(7)
        output = add_active_gaussian_noise(features, 0.0, generator)
        self.assertTrue(torch.equal(output, features))
        self.assertNotEqual(output.data_ptr(), features.data_ptr())

    def test_timestep_dropout_is_deterministic_and_preserves_padding(self):
        features = torch.tensor([[[1.0], [2.0], [3.0], [0.0]]])
        first = drop_active_timesteps(
            features, 0.5, torch.Generator().manual_seed(11)
        )
        second = drop_active_timesteps(
            features, 0.5, torch.Generator().manual_seed(11)
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first[0, 3, 0].item(), 0.0)

    def test_contiguous_mask_removes_requested_active_fraction(self):
        features = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [0.0]]])
        output = mask_active_contiguous_block(
            features, 0.5, torch.Generator().manual_seed(13)
        )
        self.assertEqual(torch.count_nonzero(output).item(), 2)
        self.assertEqual(output[0, 4, 0].item(), 0.0)

    def test_temporal_shift_preserves_active_values_and_padding(self):
        features = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [0.0]]])
        output = shift_active_timesteps(features, 1.0)
        self.assertFalse(torch.equal(output[:, :4], features[:, :4]))
        self.assertEqual(
            sorted(output[0, :4, 0].tolist()), sorted(features[0, :4, 0].tolist())
        )
        self.assertEqual(output[0, 4, 0].item(), 0.0)

    def test_modality_missing_zeros_only_active_values(self):
        features = torch.tensor([[[1.0, -1.0], [0.0, 0.0]]])
        output = remove_active_modality(features, 1.0)
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        with self.assertRaises(ValueError):
            remove_active_modality(features, 0.5)

    def test_active_statistics_ignore_padding(self):
        features = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        self.assertEqual(active_steps(features).item(), 1.0)
        self.assertAlmostEqual(active_rms(features).item(), (25.0 / 2.0) ** 0.5, places=6)

    def test_spearman_direction(self):
        self.assertAlmostEqual(spearman([0, 1, 2], [3, 2, 1]), -1.0)

    def test_severity_validation(self):
        self.assertEqual(parse_severities('0,0.5,1'), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            parse_severities('0.5,1')
        with self.assertRaises(ValueError):
            parse_severities('0,1,0.5')


if __name__ == '__main__':
    unittest.main()
