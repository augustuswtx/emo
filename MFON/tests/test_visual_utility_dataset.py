import unittest

import torch

from audit_visual_utility_dataset import FEATURE_NAMES, cheap_visual_features


class ToyReliability(torch.nn.Module):
    def forward(self, vision):
        return torch.sigmoid(vision.mean(dim=(1, 2)))


class ToyTeacher(torch.nn.Module):
    def forward(self, vision):
        return vision.mean(dim=1)


class ToyUtilityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_reliability = ToyReliability()
        self.vision_encoder_froze = ToyTeacher()

    def forward(self, text, vision, audio, mode='test'):
        visual = vision.mean(dim=1) * 0.5
        prediction = visual.sum(dim=1)
        return prediction, (None, visual, None)


class VisualUtilityDatasetTest(unittest.TestCase):
    def test_feature_columns_are_finite_per_sample(self):
        model = ToyUtilityModel().eval()
        vision = torch.tensor([
            [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]],
            [[-0.5, 0.5], [0.2, -0.1], [0.3, 0.4]],
        ])
        inputs = (torch.zeros(2, 1), vision, torch.zeros(2, 1),
                  torch.tensor([0.5, -0.2]))
        columns, teacher = cheap_visual_features(model, inputs)
        self.assertEqual(list(columns), FEATURE_NAMES)
        self.assertEqual(tuple(teacher.shape), (2, 2))
        for value in columns.values():
            self.assertEqual(tuple(value.shape), (2,))
            self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertEqual(columns['vision_active_timesteps'].tolist(), [2.0, 3.0])

    def test_single_timestep_has_zero_temporal_activity(self):
        model = ToyUtilityModel().eval()
        vision = torch.tensor([[[1.0, 2.0]], [[0.0, 0.0]]])
        inputs = (torch.zeros(2, 1), vision, torch.zeros(2, 1),
                  torch.zeros(2))
        columns, _ = cheap_visual_features(model, inputs)
        self.assertEqual(columns['vision_temporal_activity'].tolist(), [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()

