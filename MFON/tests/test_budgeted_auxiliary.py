import unittest

import torch
import torch.nn.functional as F

from budgeted_auxiliary import (
    apply_quality_control,
    build_budgeted_auxiliary,
    fixed_budget_weights,
    per_sample_infonce,
    per_sample_kl,
    sample_quality_score,
    weighted_batch_mean,
)


class BudgetedAuxiliaryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_per_sample_kl_matches_legacy_batchmean(self):
        student = torch.randn(5, 8, requires_grad=True)
        teacher = torch.randn(5, 8)
        expected = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction='batchmean',
        )
        actual = per_sample_kl(student, teacher)
        self.assertEqual(actual.shape, (5,))
        self.assertTrue(torch.allclose(actual.mean(), expected, atol=1e-7))

    def test_per_sample_infonce_matches_legacy_mean(self):
        input1 = torch.randn(6, 10)
        input2 = torch.randn(6, 10)
        x1 = input1 / input1.norm(dim=1, keepdim=True)
        x2 = input2 / input2.norm(dim=1, keepdim=True)
        expected = -(
            torch.sum(x1 * x2, dim=-1)
            - torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)
        ).mean()
        actual = per_sample_infonce(input1, input2)
        self.assertEqual(actual.shape, (6,))
        self.assertTrue(torch.allclose(actual.mean(), expected, atol=1e-7))

    def test_fixed_budget_has_exact_mean(self):
        reliability = torch.tensor([0.05, 0.2, 0.7, 1.0])
        weights = fixed_budget_weights(reliability, base_weight=0.5, progress=0.4)
        self.assertTrue(torch.allclose(weights.mean(), torch.tensor(0.2), atol=1e-7))
        self.assertGreater(weights[-1].item(), weights[0].item())

    def test_uniform_quality_recovers_original_auxiliary_scale(self):
        losses = torch.tensor([0.2, 0.5, 1.0, 1.7])
        weights = fixed_budget_weights(torch.ones(4), base_weight=0.5)
        actual = weighted_batch_mean(losses, weights)
        self.assertTrue(torch.allclose(actual, 0.5 * losses.mean(), atol=1e-7))

    def test_allocation_warmup_preserves_mean_budget(self):
        quality = torch.tensor([0.1, 0.2, 0.8, 1.0])
        initial = fixed_budget_weights(
            quality, base_weight=0.5, progress=0.0, warmup_mode='allocation'
        )
        middle = fixed_budget_weights(
            quality, base_weight=0.5, progress=0.4, warmup_mode='allocation'
        )
        final = fixed_budget_weights(
            quality, base_weight=0.5, progress=1.0, warmup_mode='allocation'
        )
        self.assertTrue(torch.allclose(initial, torch.full_like(initial, 0.5)))
        self.assertTrue(torch.allclose(middle.mean(), torch.tensor(0.5)))
        self.assertTrue(torch.allclose(final.mean(), torch.tensor(0.5)))
        self.assertLess(middle.std(unbiased=False), final.std(unbiased=False))

    def test_budget_warmup_mode_is_validated(self):
        with self.assertRaises(ValueError):
            fixed_budget_weights(
                torch.ones(3), base_weight=0.5, warmup_mode='unsupported'
            )

    def test_quality_permutation_changes_nonuniform_allocation(self):
        losses = torch.tensor([0.1, 0.2, 1.0, 2.0])
        quality = torch.tensor([1.0, 0.8, 0.2, 0.1])
        normal = weighted_batch_mean(losses, fixed_budget_weights(quality, 0.5))
        reversed_score = weighted_batch_mean(losses, fixed_budget_weights(quality.flip(0), 0.5))
        self.assertNotAlmostEqual(normal.item(), reversed_score.item(), places=5)

    def test_equal_budget_quality_controls(self):
        quality = torch.tensor([0.1, 0.4, 0.8, 0.9])
        oracle = torch.tensor([1.0, 0.7, 0.5, 0.2])

        learned = apply_quality_control(quality, 'learned')
        constant = apply_quality_control(quality, 'constant')
        permuted = apply_quality_control(quality, 'permuted')
        reversed_score = apply_quality_control(quality, 'reversed')
        oracle_score = apply_quality_control(quality, 'oracle', oracle)

        self.assertTrue(torch.equal(learned, quality))
        self.assertTrue(torch.equal(constant, torch.ones_like(quality)))
        self.assertTrue(torch.equal(permuted, quality.roll(1)))
        self.assertTrue(torch.equal(permuted.sort().values, quality.sort().values))
        self.assertTrue(torch.equal(reversed_score.sort().values, quality.sort().values))
        self.assertTrue(torch.equal(oracle_score, oracle))
        for controlled in [learned, constant, permuted, reversed_score, oracle_score]:
            weights = fixed_budget_weights(controlled, base_weight=0.5)
            self.assertTrue(
                torch.allclose(weights.mean(), torch.tensor(0.5), atol=1e-7)
            )

    def test_oracle_quality_control_requires_matching_scores(self):
        quality = torch.tensor([0.2, 0.8])
        with self.assertRaises(ValueError):
            apply_quality_control(quality, 'oracle')
        with self.assertRaises(ValueError):
            apply_quality_control(quality, 'oracle', torch.ones(3))

    def test_quality_score_is_per_sample_and_batch_composition_invariant(self):
        embed = torch.randn(3, 6)
        target = torch.randn(3, 6)
        initial = sample_quality_score(embed, target, q_type='align')
        extended = sample_quality_score(
            torch.cat([embed, torch.randn(1, 6)]),
            torch.cat([target, torch.randn(1, 6)]),
            q_type='align',
        )
        self.assertEqual(initial.shape, (3,))
        self.assertTrue(torch.allclose(initial, extended[:3], atol=1e-7))

    def test_gradients_flow_to_losses_but_not_quality_proxy(self):
        losses = torch.tensor([0.2, 0.5, 1.0], requires_grad=True)
        quality = torch.tensor([0.2, 0.5, 0.9], requires_grad=True)
        objective = weighted_batch_mean(losses, fixed_budget_weights(quality, 0.5))
        objective.backward()
        self.assertIsNotNone(losses.grad)
        self.assertIsNone(quality.grad)

    def test_complete_objective_preserves_each_budget(self):
        batch, dim = 4, 6
        embeds = [torch.randn(batch, dim) for _ in range(4)]
        losses = [torch.rand(batch, requires_grad=True) for _ in range(4)]
        result = build_budgeted_auxiliary(
            embeds[0],
            embeds[1],
            embeds[2],
            embeds[3],
            torch.randn(batch),
            losses[0],
            losses[1],
            losses[2],
            losses[3],
            q_type='align',
            temperature=1.0,
            delta_va=0.5,
            delta_nce=0.2,
            progress=0.6,
        )
        self.assertTrue(torch.allclose(result['w_v'].mean(), torch.tensor(0.3), atol=1e-7))
        self.assertTrue(torch.allclose(result['w_a'].mean(), torch.tensor(0.3), atol=1e-7))
        self.assertTrue(torch.allclose(result['w_nce_v'].mean(), torch.tensor(0.12), atol=1e-7))
        self.assertTrue(torch.allclose(result['w_nce_a'].mean(), torch.tensor(0.12), atol=1e-7))
        self.assertTrue(
            torch.allclose(
                result['loss'], result['loss_v'] + result['loss_a'] + result['loss_nce']
            )
        )
        result['loss'].backward()
        self.assertTrue(all(loss.grad is not None for loss in losses))

    def test_external_quality_override_controls_allocation(self):
        batch, dim = 4, 6
        embeds = [torch.randn(batch, dim) for _ in range(4)]
        losses = [torch.rand(batch) for _ in range(4)]
        quality_v = torch.tensor([0.1, 0.2, 0.8, 1.0], requires_grad=True)
        quality_a = torch.tensor([1.0, 0.8, 0.2, 0.1], requires_grad=True)
        result = build_budgeted_auxiliary(
            embeds[0],
            embeds[1],
            embeds[2],
            embeds[3],
            torch.randn(batch),
            losses[0],
            losses[1],
            losses[2],
            losses[3],
            q_type='align',
            temperature=1.0,
            delta_va=0.5,
            delta_nce=0.2,
            progress=1.0,
            quality_v=quality_v,
            quality_a=quality_a,
        )
        self.assertTrue(torch.equal(result['q_v'], quality_v))
        self.assertTrue(torch.equal(result['q_a'], quality_a))
        self.assertGreater(result['w_v'][-1].item(), result['w_v'][0].item())
        self.assertLess(result['w_a'][-1].item(), result['w_a'][0].item())
        self.assertFalse(result['w_v'].requires_grad)


if __name__ == '__main__':
    unittest.main()
