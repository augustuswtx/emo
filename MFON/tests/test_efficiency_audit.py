import unittest
from pathlib import Path

import torch
from torch import nn

from audit_efficiency import count_unique_parameters, parse_variants


class EfficiencyAuditTest(unittest.TestCase):
    def test_parse_variants_preserves_requested_order(self):
        self.assertEqual(
            parse_variants('learned,baseline,constant'),
            ['learned', 'baseline', 'constant'],
        )

    def test_parse_variants_rejects_unknown_or_duplicate_values(self):
        with self.assertRaises(ValueError):
            parse_variants('baseline,unknown')
        with self.assertRaises(ValueError):
            parse_variants('baseline,baseline')

    def test_parameter_count_deduplicates_shared_parameters(self):
        shared = nn.Linear(4, 3)
        expected = sum(parameter.numel() for parameter in shared.parameters())
        self.assertEqual(
            count_unique_parameters(modules=(shared, shared)),
            expected,
        )

    def test_parameter_count_combines_modules_and_standalone_parameters(self):
        module = nn.Linear(2, 2)
        standalone = nn.Parameter(torch.zeros(5))
        expected = sum(parameter.numel() for parameter in module.parameters()) + 5
        self.assertEqual(
            count_unique_parameters(
                modules=(module,), parameters=(standalone, standalone)
            ),
            expected,
        )

    def test_audit_cannot_step_optimizer_or_save_checkpoint(self):
        source = (Path(__file__).parents[1] / 'audit_efficiency.py').read_text()
        self.assertNotIn('optimizer.step(', source)
        self.assertNotIn('.save_model(', source)


if __name__ == '__main__':
    unittest.main()
