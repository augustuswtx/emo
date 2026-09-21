import unittest

import torch

from audit_visual_directional_difference import (
    cached_text_embedding,
    cached_visual_embedding,
    central_visual_kl_derivative,
)
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


class ToyCrossAttention(torch.nn.Module):
    def forward(self, query, key, value):
        return query + key.mean(dim=0, keepdim=True)


class ToyCachedFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = torch.nn.Identity()
        self.proj_t = torch.nn.Identity()
        self.proj_v = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.promptv_m = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.float64))
        self.vision_with_text = ToyCrossAttention()
        self.text_dropout = 0.0
        self.use_dpg = False

    def forward(self, text, vision, audio, mode='test'):
        hidden = self.proj_t(self.text_encoder(text).permute(1, 0, 2))
        projected = self.proj_v(vision).permute(1, 0, 2)
        projected = projected + self.promptv_m.unsqueeze(1)
        visual = self.vision_with_text(hidden, projected, projected)[0]
        return None, (None, visual, None)


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

    def test_cached_visual_path_matches_full_probe(self):
        model = ToyCachedFusion().eval()
        text = torch.tensor(
            [[[0.1, 0.3]], [[-0.4, 0.2]]], dtype=torch.float64
        )
        vision = torch.tensor(
            [[[0.8, -0.2]], [[0.5, 0.7]]], dtype=torch.float64
        )
        inputs = (text, vision, None, None)
        target = torch.tensor(
            [[0.2, 0.4], [-0.3, 0.1]], dtype=torch.float64
        )
        direction = torch.ones_like(model.proj_v.weight)
        direction = direction / direction.norm()
        cached_text = cached_text_embedding(model, text)
        _, full_embeddings = model(text, vision, None, mode='test')
        cached_visual = cached_visual_embedding(model, cached_text, vision)
        self.assertTrue(torch.allclose(full_embeddings[1], cached_visual))
        original = model.proj_v.weight.detach().clone()
        full, _, _ = central_visual_kl_derivative(
            model, inputs, target, model.proj_v.weight, direction, 1e-4
        )
        cached, _, _ = central_visual_kl_derivative(
            model, inputs, target, model.proj_v.weight, direction, 1e-4,
            cached_text=cached_text
        )
        self.assertTrue(torch.allclose(full, cached, atol=1e-7, rtol=1e-5))
        self.assertTrue(torch.equal(model.proj_v.weight.detach(), original))


if __name__ == '__main__':
    unittest.main()
