import unittest

import torch

from audit_visual_representation_proxy import visual_kl_representation_gradient
from budgeted_auxiliary import per_sample_kl


class VisualRepresentationProxyTest(unittest.TestCase):
    def test_analytic_kl_gradient_matches_autograd(self):
        student = torch.tensor(
            [[0.3, -0.2, 0.7], [-0.5, 0.1, 0.4]],
            dtype=torch.float32, requires_grad=True
        )
        teacher = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, -0.1, 0.2]],
            dtype=torch.float32
        )
        expected = torch.autograd.grad(
            per_sample_kl(student, teacher).sum(), student
        )[0]
        actual = visual_kl_representation_gradient(student.detach(), teacher)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_rejects_mismatched_embeddings(self):
        with self.assertRaises(ValueError):
            visual_kl_representation_gradient(
                torch.zeros(2, 3), torch.zeros(2, 4)
            )


if __name__ == '__main__':
    unittest.main()
