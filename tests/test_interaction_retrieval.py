#!/usr/bin/env python3
"""Focused tests for partner-retrieval representations and objectives."""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "interaction_first" / "retrieval.py"


class CompatibilityTests(unittest.TestCase):
    def test_module_uses_python_39_grammar(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(MODULE_PATH), feature_version=(3, 9))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        self.torch = torch
        torch.manual_seed(13)

    def _model(self):
        from interaction_first.retrieval import PartnerRetrievalModel

        return PartnerRetrievalModel(
            antigen_input_dim=6,
            antibody_input_dim=5,
            embedding_dim=4,
            attention_hidden_dim=7,
            projection_hidden_dim=9,
            dropout=0.0,
            initial_temperature=0.07,
            minimum_temperature=0.01,
            maximum_temperature=0.5,
        )

    def test_padding_is_ignored_and_outputs_are_normalized(self) -> None:
        torch = self.torch
        model = self._model().eval()
        antigen = torch.randn(2, 5, 6)
        antibody = torch.randn(3, 4, 5)
        antigen_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]]
        )
        antibody_mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )

        first = model(antigen, antibody, antigen_mask, antibody_mask)
        perturbed_antigen = antigen.clone()
        perturbed_antibody = antibody.clone()
        perturbed_antigen[~antigen_mask] = float("nan")
        perturbed_antibody[~antibody_mask] = 1e20
        second = model(
            perturbed_antigen,
            perturbed_antibody,
            antigen_mask,
            antibody_mask,
        )

        torch.testing.assert_close(
            first["antigen_embeddings"], second["antigen_embeddings"]
        )
        torch.testing.assert_close(
            first["antibody_embeddings"], second["antibody_embeddings"]
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(first["antigen_embeddings"], dim=1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(first["antibody_embeddings"], dim=1),
            torch.ones(3),
        )
        self.assertEqual(tuple(first["similarity_matrix"].shape), (3, 2))
        self.assertTrue(torch.all(first["similarity_matrix"].abs() <= 1.000001))
        self.assertTrue(torch.all(first["antigen_attention"][~antigen_mask] == 0.0))
        self.assertTrue(torch.all(first["antibody_attention"][~antibody_mask] == 0.0))
        torch.testing.assert_close(
            first["antigen_attention"].sum(dim=1), torch.ones(2)
        )
        torch.testing.assert_close(
            first["antibody_attention"].sum(dim=1), torch.ones(3)
        )

    def test_cosine_matrix_matches_explicit_normalized_product(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import cosine_similarity_matrix

        antibody = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        antigen = torch.tensor([[1.0, 0.0], [0.0, -3.0], [2.0, 2.0]])
        observed = cosine_similarity_matrix(antibody, antigen)
        expected = torch.nn.functional.normalize(antibody, dim=1) @ torch.nn.functional.normalize(
            antigen, dim=1
        ).transpose(0, 1)
        torch.testing.assert_close(observed, expected)

    def test_temperature_is_learnable_and_strictly_bounded(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import BoundedTemperature

        head = BoundedTemperature(0.07, 0.01, 0.5)
        initial = head()
        self.assertAlmostEqual(float(initial.detach()), 0.07, places=6)
        initial.backward()
        self.assertTrue(torch.isfinite(head.raw_temperature.grad))

        with torch.no_grad():
            head.raw_temperature.fill_(1000.0)
        self.assertLessEqual(float(head()), 0.5 + 1e-7)
        self.assertGreaterEqual(float(head()), 0.01 - 1e-7)
        with torch.no_grad():
            head.raw_temperature.fill_(-1000.0)
        self.assertLessEqual(float(head()), 0.5 + 1e-7)
        self.assertGreaterEqual(float(head()), 0.01 - 1e-7)

    def test_multi_positive_loss_matches_manual_probability_mass(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import symmetric_multi_positive_info_nce

        similarities = torch.tensor(
            [[2.0, 1.0, -1.0], [0.0, 3.0, 1.0]], dtype=torch.float64
        )
        positives = torch.tensor(
            [[True, True, False], [False, True, False]], dtype=torch.bool
        )
        observed = symmetric_multi_positive_info_nce(
            similarities, positives, temperature=1.0
        )

        row_losses = []
        for row, positive in zip(similarities, positives):
            row_losses.append(
                torch.logsumexp(row, dim=0) - torch.logsumexp(row[positive], dim=0)
            )
        column_losses = []
        for column, positive in zip(similarities.transpose(0, 1), positives.transpose(0, 1)):
            if positive.any():
                column_losses.append(
                    torch.logsumexp(column, dim=0)
                    - torch.logsumexp(column[positive], dim=0)
                )
        expected = 0.5 * (
            torch.stack(row_losses).mean() + torch.stack(column_losses).mean()
        )
        torch.testing.assert_close(observed, expected)

    def test_anchors_without_positives_are_excluded_in_both_directions(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import symmetric_multi_positive_info_nce

        base = torch.tensor(
            [[4.0, -1.0, 0.0], [-2.0, 3.0, 1.0], [5.0, 5.0, 5.0]],
            requires_grad=True,
        )
        positives = torch.tensor(
            [[True, False, False], [False, True, False], [False, False, False]]
        )
        first = symmetric_multi_positive_info_nce(base, positives, temperature=0.2)
        modified = base.detach().clone()
        modified[2, 2] = -1000.0
        second = symmetric_multi_positive_info_nce(
            modified, positives, temperature=0.2
        )
        self.assertTrue(torch.isfinite(first))
        torch.testing.assert_close(first.detach(), second)
        first.backward()
        self.assertTrue(torch.isfinite(base.grad).all())
        self.assertAlmostEqual(float(base.grad[2, 2]), 0.0, places=7)

    def test_well_aligned_pairs_have_lower_loss_than_swapped_pairs(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import symmetric_multi_positive_info_nce

        positives = torch.eye(3, dtype=torch.bool)
        aligned = torch.eye(3) * 4.0
        swapped = aligned[:, torch.tensor([1, 2, 0])]
        aligned_loss = symmetric_multi_positive_info_nce(aligned, positives, 0.5)
        swapped_loss = symmetric_multi_positive_info_nce(swapped, positives, 0.5)
        self.assertLess(float(aligned_loss), float(swapped_loss))

    def test_model_and_temperature_receive_finite_gradients(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import symmetric_multi_positive_info_nce

        model = self._model()
        antigen = torch.randn(3, 5, 6)
        antibody = torch.randn(3, 4, 5)
        antigen_mask = torch.ones(3, 5, dtype=torch.bool)
        antibody_mask = torch.ones(3, 4, dtype=torch.bool)
        outputs = model(antigen, antibody, antigen_mask, antibody_mask)
        positives = torch.eye(3, dtype=torch.bool)
        loss = symmetric_multi_positive_info_nce(
            outputs["similarity_matrix"], positives, outputs["temperature"]
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        torch = self.torch
        from interaction_first.retrieval import (
            BoundedTemperature,
            MaskedAttentionPooler,
            cosine_similarity_matrix,
            symmetric_multi_positive_info_nce,
        )

        pooler = MaskedAttentionPooler(4, 3)
        features = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(ValueError, "boolean"):
            pooler(features, torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            pooler(features, torch.tensor([[True, False, False], [False] * 3]))
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = features.clone()
            bad[0, 0, 0] = float("nan")
            pooler(bad, torch.ones(2, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            cosine_similarity_matrix(torch.zeros(1, 3), torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "same shape"):
            symmetric_multi_positive_info_nce(
                torch.ones(2, 2), torch.ones(2, 3, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            symmetric_multi_positive_info_nce(
                torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            symmetric_multi_positive_info_nce(
                torch.ones(2, 2), torch.eye(2, dtype=torch.bool), temperature=0.0
            )
        with self.assertRaisesRegex(ValueError, "satisfy"):
            BoundedTemperature(0.01, 0.01, 0.5)


if __name__ == "__main__":
    unittest.main()
