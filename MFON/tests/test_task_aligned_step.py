import unittest

import torch

from audit_task_aligned_step import probe_one, task_mse, training_loader


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2.0]]))
        self.fail_on_shift = False

    def forward(self, text, vision, audio, mode='test'):
        if self.fail_on_shift and self.weight.item() != 2.0:
            raise RuntimeError('injected forward failure')
        return (vision @ self.weight).view(-1), None


class TaskAlignedStepTest(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.inputs = (
            None,
            torch.tensor([[1.0], [2.0]]),
            None,
            torch.tensor([1.0, 2.0]),
        )

    def test_predicted_direction_matches_exact_quadratic_probe(self):
        baseline_tensor = task_mse(self.model, self.inputs)
        meta_gradient = torch.autograd.grad(
            baseline_tensor, self.model.weight
        )[0]
        auxiliary_gradient = torch.tensor([[0.5]])
        dot = float((meta_gradient * auxiliary_gradient).sum())
        baseline = float(baseline_tensor)
        original = self.model.weight.detach().clone()

        result = probe_one(
            self.model, self.inputs, self.model.weight,
            auxiliary_gradient, dot, baseline, 0.001
        )

        self.assertEqual(result['probe_status'], 'ok')
        self.assertTrue(torch.equal(self.model.weight.detach(), original))
        self.assertLess(result['actual_minus_change'], 0)
        self.assertAlmostEqual(
            result['predicted_minus_change'],
            result['central_minus_change'], places=5
        )

    def test_parameter_restored_when_probe_forward_fails(self):
        original = self.model.weight.detach().clone()
        self.model.fail_on_shift = True
        with self.assertRaisesRegex(RuntimeError, 'injected forward failure'):
            probe_one(
                self.model, self.inputs, self.model.weight,
                torch.tensor([[0.5]]), 2.5, 2.5, 0.001
            )
        self.assertTrue(torch.equal(self.model.weight.detach(), original))

    def test_seeded_loader_is_reproducible_and_excludes_prior_indices(self):
        class FakeDataModule:
            @staticmethod
            def MOSEIDataset(mode, raw_data_path):
                return list(range(40))

        first = list(training_loader(FakeDataModule, '', 4, 17, 16))
        second = list(training_loader(FakeDataModule, '', 4, 17, 16))
        first_indices = [int(item) for batch in first for item in batch]
        second_indices = [int(item) for batch in second for item in batch]
        self.assertEqual(first_indices, second_indices)
        self.assertEqual(set(first_indices), set(range(16, 40)))
        self.assertNotEqual(first_indices, list(range(16, 40)))


if __name__ == '__main__':
    unittest.main()
