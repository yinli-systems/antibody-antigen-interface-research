"""Tests for the prospective D0/R1/R2 frontier models."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class FrontierModelTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[object, ...]:
        import torch

        torch.manual_seed(20260724)
        antibody = torch.randn(3, 7, 12)
        antigen = torch.randn(3, 9, 16)
        antibody_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        )
        antigen_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        )
        chain_ids = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 1, 1, 1, -1],
            ],
            dtype=torch.long,
        )
        return antibody, antigen, antibody_mask, antigen_mask, chain_ids

    @staticmethod
    def _coarse() -> object:
        from interaction_first.frontier_model import ProtocolD0Retriever

        return ProtocolD0Retriever(
            antibody_input_dim=12,
            antigen_input_dim=16,
            embedding_dim=8,
            projection_hidden_dim=16,
            dropout=0.0,
        )

    def test_d0_paired_score_matches_similarity_diagonal(self) -> None:
        import torch

        coarse = self._coarse().eval()
        antibody, antigen, antibody_mask, antigen_mask, chain_ids = self._inputs()
        paired = coarse.paired_scores(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        full = coarse(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        torch.testing.assert_close(
            paired["coarse_score"],
            full["similarity_matrix"].diagonal(),
        )
        torch.testing.assert_close(
            full["temperature"],
            torch.tensor(0.07),
        )
        self.assertFalse(full["temperature"].requires_grad)

    def test_d0_uses_length_mean_and_learned_chain_type_vectors(self) -> None:
        import torch

        coarse = self._coarse().eval()
        antibody, antigen, antibody_mask, antigen_mask, chain_ids = self._inputs()
        original = coarse(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        valid_weights = original["antibody_weights"][0][antibody_mask[0]]
        torch.testing.assert_close(
            valid_weights,
            torch.full_like(valid_weights, 1.0 / int(antibody_mask[0].sum())),
        )
        swapped = chain_ids.clone()
        swapped[antibody_mask] = 1 - swapped[antibody_mask]
        changed = coarse(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            swapped,
        )
        self.assertFalse(
            torch.allclose(
                original["antibody_embeddings"],
                changed["antibody_embeddings"],
            )
        )

    def test_r1_shapes_and_frozen_coarse_model(self) -> None:
        import torch

        from interaction_first.frontier_model import ProtocolReranker

        model = ProtocolReranker(
            self._coarse(),
            antibody_input_dim=12,
            antigen_input_dim=16,
            d_model=16,
            num_heads=4,
            num_layers=2,
            ffn_dim=32,
            dropout=0.0,
        ).train()
        self.assertEqual(model.stage, "R1")
        self.assertFalse(model.coarse_retriever.training)
        self.assertFalse(any(p.requires_grad for p in model.coarse_retriever.parameters()))
        output = model(*self._inputs())
        self.assertEqual(tuple(output["retrieval_score"].shape), (3,))
        self.assertEqual(tuple(output["antibody_residues"].shape), (3, 7, 16))
        self.assertEqual(tuple(output["antigen_residues"].shape), (3, 9, 16))
        self.assertNotIn("contact_logits", output)
        self.assertTrue(torch.isfinite(output["retrieval_score"]).all())

    def test_r2_contact_orientation_and_padding_mask(self) -> None:
        import torch

        from interaction_first.frontier_model import ProtocolReranker

        model = ProtocolReranker(
            self._coarse(),
            antibody_input_dim=12,
            antigen_input_dim=16,
            d_model=16,
            num_heads=4,
            num_layers=2,
            ffn_dim=32,
            dropout=0.0,
            contact_pair_dim=8,
        ).eval()
        antibody, antigen, antibody_mask, antigen_mask, chain_ids = self._inputs()
        output = model(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        expected_mask = antigen_mask.unsqueeze(-1) & antibody_mask.unsqueeze(-2)
        self.assertEqual(model.stage, "R2")
        self.assertEqual(tuple(output["contact_logits"].shape), (3, 9, 7))
        torch.testing.assert_close(output["contact_pair_mask"], expected_mask)
        self.assertTrue(
            torch.equal(
                output["contact_probabilities"][~expected_mask],
                torch.zeros_like(output["contact_probabilities"][~expected_mask]),
            )
        )

    def test_padding_values_cannot_change_valid_outputs(self) -> None:
        import torch

        from interaction_first.frontier_model import ProtocolReranker

        model = ProtocolReranker(
            self._coarse(),
            antibody_input_dim=12,
            antigen_input_dim=16,
            d_model=16,
            num_heads=4,
            num_layers=2,
            ffn_dim=32,
            dropout=0.0,
            contact_pair_dim=8,
        ).eval()
        antibody, antigen, antibody_mask, antigen_mask, chain_ids = self._inputs()
        first = model(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        antibody_changed = antibody.clone()
        antigen_changed = antigen.clone()
        antibody_changed[~antibody_mask] = 1e6
        antigen_changed[~antigen_mask] = -1e6
        second = model(
            antibody_changed,
            antigen_changed,
            antibody_mask,
            antigen_mask,
            chain_ids,
        )
        torch.testing.assert_close(first["retrieval_score"], second["retrieval_score"])
        pair_mask = first["contact_pair_mask"]
        torch.testing.assert_close(
            first["contact_logits"][pair_mask],
            second["contact_logits"][pair_mask],
        )

    def test_contact_targets_are_not_model_inputs(self) -> None:
        import inspect

        from interaction_first.frontier_model import ProtocolReranker

        parameters = inspect.signature(ProtocolReranker.forward).parameters
        self.assertNotIn("contact_targets", parameters)
        self.assertNotIn("contact_labels", parameters)

    def test_indexed_pair_scoring_matches_direct_selected_inputs(self) -> None:
        import torch

        from interaction_first.frontier_model import ProtocolReranker

        model = ProtocolReranker(
            self._coarse(),
            antibody_input_dim=12,
            antigen_input_dim=16,
            d_model=16,
            num_heads=4,
            num_layers=2,
            ffn_dim=32,
            dropout=0.0,
        ).eval()
        antibody, antigen, antibody_mask, antigen_mask, chain_ids = self._inputs()
        pair_indices = torch.tensor([[0, 0, 2, 1], [2, 1, 0, 1]])
        indexed = model.score_indexed_pairs(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            chain_ids,
            pair_indices,
        )
        direct = model(
            antibody[pair_indices[0]],
            antigen[pair_indices[1]],
            antibody_mask[pair_indices[0]],
            antigen_mask[pair_indices[1]],
            chain_ids[pair_indices[0]],
        )
        torch.testing.assert_close(
            indexed["retrieval_score"],
            direct["retrieval_score"],
        )

    def test_indexed_pair_scoring_rejects_invalid_edges(self) -> None:
        import torch

        from interaction_first.frontier_model import ProtocolReranker

        model = ProtocolReranker(
            self._coarse(),
            antibody_input_dim=12,
            antigen_input_dim=16,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ffn_dim=32,
            dropout=0.0,
        )
        inputs = self._inputs()
        with self.assertRaisesRegex(IndexError, "antigen"):
            model.score_indexed_pairs(
                *inputs,
                torch.tensor([[0], [3]], dtype=torch.long),
            )

    def test_balanced_contact_bce_matches_class_means(self) -> None:
        import torch
        import torch.nn.functional as functional

        from interaction_first.frontier_model import balanced_contact_bce

        logits = torch.tensor([[[2.0, -1.0], [0.5, -0.5]]])
        targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        expected = 0.5 * (
            functional.binary_cross_entropy_with_logits(
                logits[targets == 1.0],
                targets[targets == 1.0],
            )
            + functional.binary_cross_entropy_with_logits(
                logits[targets == 0.0],
                targets[targets == 0.0],
            )
        )
        torch.testing.assert_close(
            balanced_contact_bce(logits, targets, mask),
            expected,
        )

    def test_balanced_contact_bce_excludes_single_class_complexes(self) -> None:
        import torch

        from interaction_first.frontier_model import balanced_contact_bce

        logits = torch.zeros(2, 2, 2, requires_grad=True)
        targets = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        )
        mask = torch.ones_like(targets, dtype=torch.bool)
        loss = balanced_contact_bce(logits, targets, mask)
        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros_like(logits.grad[0])))

    def test_balanced_contact_bce_exposes_exact_global_reduction_statistics(
        self,
    ) -> None:
        import torch

        from interaction_first.frontier_model import balanced_contact_bce

        logits = torch.zeros(3, 2, 2, requires_grad=True)
        targets = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        )
        mask = torch.ones_like(targets, dtype=torch.bool)
        details = balanced_contact_bce(
            logits,
            targets,
            mask,
            return_details=True,
        )
        self.assertEqual(details["valid_complex_count"], 2)
        self.assertEqual(details["observed_complex_count"], 3)
        self.assertEqual(details["reduction"], "equal_per_dual_class_complex")
        torch.testing.assert_close(
            details["loss"],
            details["loss_sum"] / details["valid_complex_count"],
        )

    def test_balanced_contact_bce_rejects_hidden_unresolved_labels(self) -> None:
        import torch

        from interaction_first.frontier_model import balanced_contact_bce

        with self.assertRaisesRegex(ValueError, "zero placeholders"):
            balanced_contact_bce(
                torch.zeros(1, 1, 2),
                torch.tensor([[[1.0, 0.0]]]),
                torch.tensor([[[False, True]]]),
            )


if __name__ == "__main__":
    unittest.main()
