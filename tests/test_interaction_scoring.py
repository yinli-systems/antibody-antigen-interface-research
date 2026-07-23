#!/usr/bin/env python3
"""Focused tests for robust interaction scoring and marginal pooling."""

from __future__ import annotations

import importlib.util
import math
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class InteractionScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch

    def test_noisy_or_matches_independent_contact_probability(self) -> None:
        from interaction_first.scoring import pool_pair_marginals

        torch = self.torch
        probabilities = torch.tensor([[[0.20, 0.30], [0.40, 0.50]]])
        logits = torch.logit(probabilities)
        mask = torch.ones_like(logits, dtype=torch.bool)
        antigen, antibody = pool_pair_marginals(logits, mask, pooling="noisy_or")
        torch.testing.assert_close(antigen, torch.tensor([[0.44, 0.70]]))
        torch.testing.assert_close(antibody, torch.tensor([[0.52, 0.65]]))

    def test_noisy_or_is_stable_at_extreme_logits(self) -> None:
        from interaction_first.scoring import pool_pair_marginals

        torch = self.torch
        logits = torch.tensor([[[1000.0, -1000.0], [-1000.0, -1000.0]]])
        mask = torch.ones_like(logits, dtype=torch.bool)
        antigen, antibody = pool_pair_marginals(logits, mask, pooling="noisy_or")
        self.assertTrue(torch.isfinite(antigen).all())
        self.assertTrue(torch.isfinite(antibody).all())
        self.assertEqual(float(antigen[0, 0]), 1.0)
        self.assertEqual(float(antigen[0, 1]), 0.0)

    def test_marginal_pooling_excludes_padding_and_zeros_empty_axes(self) -> None:
        from interaction_first.scoring import pool_pair_marginals

        torch = self.torch
        logits = torch.tensor([[[0.0, 2.0, 100.0], [100.0, 100.0, 100.0]]])
        mask = torch.tensor([[[True, True, False], [False, False, False]]])
        antigen, antibody = pool_pair_marginals(
            logits, mask, pooling="topk", top_k=2
        )
        expected = (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))) / 2
        torch.testing.assert_close(antigen[0, 0], expected)
        self.assertEqual(float(antigen[0, 1]), 0.0)
        torch.testing.assert_close(antibody[0, :2], torch.sigmoid(logits[0, 0, :2]))
        self.assertEqual(float(antibody[0, 2]), 0.0)

    def test_explicit_decoder_biases_are_removed_exactly(self) -> None:
        from interaction_first.scoring import interaction_residual

        torch = self.torch
        interaction = torch.tensor([[[1.0, -1.0], [-2.0, 2.0]]])
        antigen_bias = torch.tensor([[0.5, 1.5]])
        antibody_bias = torch.tensor([[-0.25, 0.75]])
        global_bias = torch.tensor(-3.0)
        logits = (
            interaction
            + antigen_bias.unsqueeze(2)
            + antibody_bias.unsqueeze(1)
            + global_bias
        )
        mask = torch.ones_like(logits, dtype=torch.bool)
        residual = interaction_residual(
            logits,
            mask,
            antigen_bias=antigen_bias,
            antibody_bias=antibody_bias,
            global_bias=global_bias,
            center=False,
        )
        torch.testing.assert_close(residual, interaction)

    def test_double_center_removes_masked_unary_effects(self) -> None:
        from interaction_first.scoring import masked_double_center

        torch = self.torch
        row = torch.tensor([[[1.0], [3.0], [8.0]]])
        column = torch.tensor([[[2.0, -4.0, 5.0]]])
        values = row + column
        mask = torch.tensor(
            [[[True, True, False], [True, True, False], [False, False, False]]]
        )
        centered = masked_double_center(values, mask)
        torch.testing.assert_close(centered[mask], torch.zeros_like(centered[mask]))
        torch.testing.assert_close(centered[~mask], torch.zeros_like(centered[~mask]))

    def test_double_center_handles_sparse_internal_holes(self) -> None:
        from interaction_first.scoring import masked_double_center

        torch = self.torch
        row = torch.tensor([[[1.0], [4.0], [-2.0]]], dtype=torch.float64)
        column = torch.tensor([[[3.0, -5.0, 2.0]]], dtype=torch.float64)
        values = row + column
        mask = torch.tensor(
            [[[True, True, False], [False, True, True], [True, False, True]]]
        )
        centered = masked_double_center(values, mask, iterations=32)
        torch.testing.assert_close(
            centered[mask], torch.zeros_like(centered[mask]), atol=1e-10, rtol=0.0
        )

    def test_logmeanexp_score_is_invariant_to_repeated_grid_entries(self) -> None:
        from interaction_first.scoring import masked_logmeanexp_score

        torch = self.torch
        base = torch.tensor([[[0.0, 2.0], [-1.0, 1.0]]])
        repeated = base.repeat_interleave(3, dim=1).repeat_interleave(4, dim=2)
        base_score = masked_logmeanexp_score(
            base, torch.ones_like(base, dtype=torch.bool), temperature=0.7
        )
        repeated_score = masked_logmeanexp_score(
            repeated, torch.ones_like(repeated, dtype=torch.bool), temperature=0.7
        )
        torch.testing.assert_close(base_score, repeated_score)

    def test_top_fraction_score_is_invariant_to_proportional_repetition(self) -> None:
        from interaction_first.scoring import masked_top_fraction_score

        torch = self.torch
        base = torch.tensor([[[4.0, 4.0], [0.0, 0.0]]])
        repeated = base.repeat_interleave(3, dim=1).repeat_interleave(2, dim=2)
        base_score = masked_top_fraction_score(
            base, torch.ones_like(base, dtype=torch.bool), fraction=0.5
        )
        repeated_score = masked_top_fraction_score(
            repeated, torch.ones_like(repeated, dtype=torch.bool), fraction=0.5
        )
        torch.testing.assert_close(base_score, torch.tensor([4.0]))
        torch.testing.assert_close(base_score, repeated_score)

    def test_normalized_partner_score_has_finite_gradients(self) -> None:
        from interaction_first.scoring import normalized_partner_score

        torch = self.torch
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(2, 4, 3, generator=generator, requires_grad=True)
        mask = torch.tensor(
            [
                [[True, True, True]] * 4,
                [[True, True, False], [True, True, False], [True, True, False], [False, False, False]],
            ]
        )
        score = normalized_partner_score(
            logits, mask, pooling="logmeanexp", temperature=0.5
        )
        score.sum().backward()
        self.assertTrue(torch.isfinite(score).all())
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_strict_input_validation_rejects_invalid_inputs(self) -> None:
        from interaction_first.scoring import (
            interaction_residual,
            masked_logmeanexp_score,
            masked_top_fraction_score,
            pool_pair_marginals,
        )

        torch = self.torch
        values = torch.zeros(1, 2, 3)
        mask = torch.ones_like(values, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "same shape"):
            masked_logmeanexp_score(values, mask[:, :, :2])
        with self.assertRaisesRegex(ValueError, "boolean"):
            masked_logmeanexp_score(values, mask.float())
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            masked_logmeanexp_score(values, torch.zeros_like(mask))
        with self.assertRaisesRegex(ValueError, "finite"):
            masked_logmeanexp_score(values.masked_fill(mask, math.inf), mask)
        with self.assertRaisesRegex(ValueError, "temperature"):
            masked_logmeanexp_score(values, mask, temperature=0.0)
        with self.assertRaisesRegex(ValueError, "fraction"):
            masked_top_fraction_score(values, mask, fraction=1.1)
        with self.assertRaisesRegex(ValueError, "antigen_bias"):
            interaction_residual(
                values, mask, antigen_bias=torch.zeros(1, 3), center=False
            )
        with self.assertRaisesRegex(ValueError, "pooling"):
            pool_pair_marginals(values, mask, pooling="maximum")


if __name__ == "__main__":
    unittest.main()
