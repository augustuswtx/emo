import unittest

import torch

from audit_visual_directional_difference import central_visual_kl_derivative
from budgeted_auxiliary import per_sample_kl


class ToyVisualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(
            [[0.8, -0.3], [0.2, 0.5]], dtype=torch.float64
        ))
        self.fail_on_shift = False

    def forward(self, text, vision, audio, mode='test'):
        if self.fail_on_shift and self.weight[0, 0].item() != 0.8:
            raise RuntimeError('injected forward failure')
        embedding = vision @ self.weight.T
        return None, (None, embedding, None)


class VisualDirectionalDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = ToyVisualModel()
        self.inputs = (
            None,
            torch.tensor([[1.0, 0.5], [-0.25, 1.5]], dtype=torch.float64),
            None,
            None,
        )
        self.target = torch.tensor(
            [[0.1, 0.4], [0.3, -0.2]], dtype=torch.float64
        )

    def test_central_difference_matches_each_parameter_dot(self):
        direction = torch.tensor(
            [[0.3, -0.1], [0.2, 0.4]], dtype=torch.float64
        )
        direction = direction / direction.norm()
        original = self.model.weight.detach().clone()
        _, embeddings = self.model(*self.inputs[:3], mode='test')
        losses = per_sample_kl(embeddings[1], self.target)
        exact = torch.stack([
            (torch.autograd.grad(losses[i], self.model.weight,
                                 retain_graph=True)[0] * direction).sum()
            for i in range(2)
        ])
        estimated, _, _ = central_visual_kl_derivative(
            self.model, self.inputs, self.target,
            self.model.weight, direction, 1e-4
        )
        self.assertTrue(torch.allclose(estimated, exact, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.equal(self.model.weight.detach(), original))

    def test_parameter_restored_on_forward_failure(self):
        original = self.model.weight.detach().clone()
        self.model.fail_on_shift = True
        with self.assertRaisesRegex(RuntimeError, 'injected forward failure'):
            central_visual_kl_derivative(
                self.model, self.inputs, self.target, self.model.weight,
                torch.ones_like(self.model.weight) / 2.0, 1e-4
            )
        self.assertTrue(torch.equal(self.model.weight.detach(), original))


if __name__ == '__main__':
    unittest.main()
