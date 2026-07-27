#!/usr/bin/env python3
"""Unit tests for the interaction-first research components."""

from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from interaction_first.partners import PartnerRecord, PartnerSampler
from interaction_first.splits import (
    SimilarityEdge,
    audit_component_split,
    build_components,
    deterministic_component_split,
)


class SplitTests(unittest.TestCase):
    def test_components_join_all_relationship_types(self) -> None:
        components = build_components(
            ["a", "b", "c", "d", "e", "f"],
            similarity_edges=[
                SimilarityEdge("a", "b", "antibody_full", 0.8, 0.9, 0.9),
                SimilarityEdge("c", "d", "antigen_full", 0.7, 0.8, 0.8),
            ],
            equivalence_groups=[["b", "c"], ["e", "f"]],
        )
        self.assertEqual(components, [("a", "b", "c", "d"), ("e", "f")])

    def test_component_split_is_deterministic_and_isolated(self) -> None:
        components = [
            ("a", "b", "c"),
            ("d", "e"),
            ("f",),
            ("g",),
            ("h",),
            ("i",),
        ]
        fractions = {"train": 0.6, "validation": 0.2, "test": 0.2}
        first = deterministic_component_split(components, fractions, seed=17)
        second = deterministic_component_split(components, fractions, seed=17)
        self.assertEqual(first, second)
        audit = audit_component_split(components, first)
        self.assertTrue(audit["valid"])
        self.assertEqual(set(audit["split_instance_counts"]), set(fractions))
        for component in components:
            self.assertEqual(len({first[member] for member in component}), 1)

    def test_large_component_does_not_starve_holdout_splits(self) -> None:
        components = [tuple(f"large-{index}" for index in range(80))]
        components.extend((f"small-{index}",) for index in range(20))
        assignments = deterministic_component_split(
            components,
            {"train": 0.8, "validation": 0.1, "test": 0.1},
            seed=42,
        )
        counts = {
            split_name: sum(value == split_name for value in assignments.values())
            for split_name in ("train", "validation", "test")
        }
        self.assertEqual(counts, {"train": 80, "validation": 10, "test": 10})

    def test_unknown_similarity_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown instance"):
            build_components(
                ["a", "b"],
                similarity_edges=[SimilarityEdge("a", "missing", "test")],
            )


class PartnerSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            PartnerRecord("a", "ab-a", "ag-a", 120, 400, "ab-g1", "ag-g1"),
            PartnerRecord("b", "ab-b", "ag-b", 121, 410, "ab-g1", "ag-g2"),
            PartnerRecord("c", "ab-c", "ag-c", 160, 700, "ab-g2", "ag-g3"),
            PartnerRecord("d", "ab-d", "ag-d", 122, 405, "ab-g1", "ag-g1"),
        ]

    def test_sampler_prefers_group_and_length_matched_real_sequences(self) -> None:
        sampler = PartnerSampler(self.records, seed=11)
        candidates = sampler.candidates("a", swap="antibody")
        self.assertEqual(candidates[0].instance_id, "b")
        self.assertNotIn("d", [record.instance_id for record in candidates])

    def test_sampler_is_epoch_reproducible(self) -> None:
        sampler = PartnerSampler(self.records, seed=11)
        first = sampler.sample("a", count=2, epoch=3, hard_pool_size=3)
        second = sampler.sample("a", count=2, epoch=3, hard_pool_size=3)
        self.assertEqual(first, second)
        self.assertTrue(all(record.antibody_key != "ab-a" for record in first))

    def test_known_positive_cross_pair_is_never_sampled(self) -> None:
        sampler = PartnerSampler(
            self.records,
            known_positive_pairs=[("ab-b", "ag-a")],
            seed=5,
        )
        candidates = sampler.candidates(
            "a", swap="antibody", require_different_antigen_group=False
        )
        self.assertNotIn("b", [record.instance_id for record in candidates])

    def test_sampler_returns_all_available_unique_partners(self) -> None:
        records = [
            PartnerRecord("a", "ab-a", "ag-a", 120, 400),
            PartnerRecord("b", "ab-b", "ag-b", 121, 410),
            PartnerRecord("c", "ab-b", "ag-c", 121, 420),
        ]
        sampler = PartnerSampler(records, seed=7)
        selected = sampler.sample("a", count=2, epoch=1, hard_pool_size=2)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].antibody_key, "ab-b")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class ContactMapTests(unittest.TestCase):
    def test_sparse_round_trip_and_marginals(self) -> None:
        import numpy as np

        from interaction_first.contacts import (
            SparseContactMap,
            load_contact_map,
            load_contact_map_bytes,
            save_contact_map,
        )

        contact = SparseContactMap.from_rows(
            "example",
            antibody_length=10,
            antigen_length=12,
            antibody_indices=[2, 2, 5, 5],
            antigen_indices=[1, 7, 1, 7],
            labels=[1, 0, 0, 1],
        )
        labels, mask = contact.dense()
        self.assertEqual(labels.shape, (2, 2))
        self.assertTrue(mask.all())
        np.testing.assert_array_equal(contact.node_labels()[0], [1.0, 1.0])
        np.testing.assert_array_equal(contact.node_labels()[1], [1.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contact.npz"
            save_contact_map(contact, path)
            loaded = load_contact_map(path, expected_instance_id="example")
            loaded_from_bytes = load_contact_map_bytes(
                path.read_bytes(),
                expected_instance_id="example",
                source_name="unit-test bytes",
            )
        np.testing.assert_array_equal(loaded.pair_labels, contact.pair_labels)
        np.testing.assert_array_equal(
            loaded_from_bytes.pair_labels,
            contact.pair_labels,
        )

    def test_duplicate_candidate_pair_is_rejected(self) -> None:
        from interaction_first.contacts import SparseContactMap

        with self.assertRaisesRegex(ValueError, "duplicate"):
            SparseContactMap.from_rows(
                "duplicate",
                antibody_length=5,
                antigen_length=5,
                antibody_indices=[1, 1],
                antigen_indices=[2, 2],
                labels=[0, 1],
            )


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is unavailable")
class MetricTests(unittest.TestCase):
    def test_known_binary_ranking_metrics(self) -> None:
        from interaction_first.metrics import average_precision, binary_metrics, roc_auc

        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        self.assertAlmostEqual(roc_auc(labels, scores), 0.75)
        self.assertAlmostEqual(average_precision(labels, scores), 5.0 / 6.0)
        metrics = binary_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mcc"], 1.0 / math.sqrt(3.0))

    def test_partner_ranking_uses_hardest_negative(self) -> None:
        from interaction_first.metrics import partner_ranking_metrics

        metrics = partner_ranking_metrics(
            matched_scores=[3.0, 1.0, 2.0],
            negative_scores=[[1.0, 2.0], [1.5, 0.5], [0.0, 1.0]],
        )
        self.assertAlmostEqual(metrics["recall_at_1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mean_rank"], 4.0 / 3.0)

    def test_cluster_bootstrap_is_reproducible(self) -> None:
        from interaction_first.metrics import cluster_bootstrap_mean_ci

        values = {"a": 1.0, "b": 3.0, "c": 8.0}
        groups = {"a": "g1", "b": "g1", "c": "g2"}
        first = cluster_bootstrap_mean_ci(values, groups, iterations=100, seed=7)
        second = cluster_bootstrap_mean_ci(values, groups, iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], 4.0)


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is unavailable")
class AntibodyMetadataTests(unittest.TestCase):
    def test_cdrh3_edges_work_without_germline_annotations(self) -> None:
        import pandas as pd

        from sabdab2_interaction_first import cdrh3_similarity_edges

        frame = pd.DataFrame(
            {
                "instance_id": ["a", "b", "c"],
                "cdrh3": ["ABCDE", "ABCDF", "QRST"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cdrh3.csv"
            frame.to_csv(path, index=False)
            edges = cdrh3_similarity_edges(path, {"a", "b", "c"}, threshold=0.70)
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0].left, edges[0].right}, {"a", "b"})
        self.assertEqual(edges[0].source, "cdrh3_length")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class ModelTests(unittest.TestCase):
    @staticmethod
    def _candidate_mask_model() -> object:
        from interaction_first.model import PartnerSpecificInteractionModel

        return PartnerSpecificInteractionModel(
            antigen_input_dim=8,
            antibody_input_dim=8,
            d_model=16,
            pair_dim=8,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
            score_pooling="top_fraction",
            score_top_fraction=0.01,
            score_top_k=3,
        )

    def test_candidate_pair_mask_controls_every_pair_derived_output(self) -> None:
        import torch

        from interaction_first.scoring import (
            normalized_partner_score,
            pool_pair_marginals,
        )

        torch.manual_seed(29)
        model = self._candidate_mask_model().eval()
        antigen = torch.randn(1, 4, 8)
        antibody = torch.randn(1, 3, 8)
        candidate_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False],
                ]
            ]
        )
        outputs = model(
            antigen,
            antibody,
            candidate_pair_mask=candidate_mask,
        )

        torch.testing.assert_close(outputs["pair_mask"], candidate_mask)
        self.assertTrue(outputs["axis_pair_mask"].all())
        self.assertTrue(
            torch.equal(
                outputs["pair_probabilities"][~candidate_mask],
                torch.zeros_like(outputs["pair_probabilities"][~candidate_mask]),
            )
        )
        torch.testing.assert_close(
            outputs["epitope_supervision_mask"],
            candidate_mask.any(dim=2),
        )
        torch.testing.assert_close(
            outputs["paratope_supervision_mask"],
            candidate_mask.any(dim=1),
        )
        self.assertTrue(
            torch.equal(
                outputs["epitope_probability"][:, 2:],
                torch.zeros_like(outputs["epitope_probability"][:, 2:]),
            )
        )
        self.assertEqual(
            outputs["paratope_probability"][0, 2].detach().item(),
            0.0,
        )

        expected_marginals = pool_pair_marginals(
            outputs["pair_logits"],
            candidate_mask,
            pooling=model.marginal_pooling,
            top_k=model.marginal_top_k,
            temperature=model.marginal_temperature,
        )
        torch.testing.assert_close(
            outputs["pair_epitope_probability"],
            expected_marginals[0],
        )
        torch.testing.assert_close(
            outputs["pair_paratope_probability"],
            expected_marginals[1],
        )
        expected_score = normalized_partner_score(
            outputs["interaction_logits"],
            candidate_mask,
            pooling="top_fraction",
            temperature=model.score_temperature,
            top_fraction=model.score_top_fraction,
            min_count=model.score_top_k,
            center=model.center_partner_score,
        )
        torch.testing.assert_close(outputs["complex_score"], expected_score)

    def test_candidate_mask_rejects_padded_pairs(self) -> None:
        import torch

        model = self._candidate_mask_model().eval()
        antigen = torch.randn(1, 3, 8)
        antibody = torch.randn(1, 2, 8)
        antigen_mask = torch.tensor([[True, True, False]])
        antibody_mask = torch.ones(1, 2, dtype=torch.bool)
        candidate_mask = torch.ones(1, 3, 2, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "padded residue"):
            model(
                antigen,
                antibody,
                antigen_mask,
                antibody_mask,
                candidate_mask,
            )

    def test_loss_ignores_candidate_unsupported_node_targets(self) -> None:
        import torch

        from interaction_first.losses import (
            InteractionLossWeights,
            interaction_first_loss,
        )

        torch.manual_seed(31)
        model = self._candidate_mask_model().eval()
        antigen = torch.randn(1, 4, 8)
        antibody = torch.randn(1, 3, 8)
        candidate_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False],
                ]
            ]
        )
        outputs = model(
            antigen,
            antibody,
            candidate_pair_mask=candidate_mask,
        )
        pair_targets = torch.zeros_like(outputs["pair_logits"])
        antigen_mask = torch.ones(1, 4, dtype=torch.bool)
        antibody_mask = torch.ones(1, 3, dtype=torch.bool)
        epitope_targets = torch.zeros(1, 4)
        paratope_targets = torch.zeros(1, 3)
        altered_epitope = epitope_targets.clone()
        altered_paratope = paratope_targets.clone()
        altered_epitope[:, 2:] = 1.0
        altered_paratope[:, 2:] = 1.0
        weights = InteractionLossWeights(
            edge=0.0,
            epitope_bce=1.0,
            paratope_bce=1.0,
            epitope_dice=0.0,
            paratope_dice=0.0,
            partner_rank=0.0,
        )
        baseline, _ = interaction_first_loss(
            outputs,
            pair_targets,
            epitope_targets,
            paratope_targets,
            antigen_mask,
            antibody_mask,
            weights=weights,
        )
        altered, _ = interaction_first_loss(
            outputs,
            pair_targets,
            altered_epitope,
            altered_paratope,
            antigen_mask,
            antibody_mask,
            weights=weights,
        )
        torch.testing.assert_close(baseline, altered)

    def test_pair_marginal_consistency_backpropagates_to_both_views(self) -> None:
        import torch

        from interaction_first.losses import (
            InteractionLossWeights,
            interaction_first_loss,
        )

        torch.manual_seed(37)
        model = self._candidate_mask_model()
        antigen = torch.randn(1, 4, 8)
        antibody = torch.randn(1, 3, 8)
        candidate_mask = torch.tensor(
            [
                [
                    [True, True, False],
                    [True, False, True],
                    [False, True, True],
                    [True, False, False],
                ]
            ]
        )
        outputs = model(
            antigen,
            antibody,
            candidate_pair_mask=candidate_mask,
        )
        pair_targets = torch.zeros_like(outputs["pair_logits"])
        epitope_targets = torch.zeros(1, 4)
        paratope_targets = torch.zeros(1, 3)
        weights = InteractionLossWeights(
            edge=0.0,
            epitope_bce=0.0,
            paratope_bce=0.0,
            epitope_dice=0.0,
            paratope_dice=0.0,
            epitope_consistency=1.0,
            paratope_consistency=1.0,
            partner_rank=0.0,
        )
        loss, components = interaction_first_loss(
            outputs,
            pair_targets,
            epitope_targets,
            paratope_targets,
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            weights=weights,
        )
        loss.backward()
        self.assertGreater(
            components["epitope_consistency"].detach().item(),
            0.0,
        )
        self.assertGreater(
            components["paratope_consistency"].detach().item(),
            0.0,
        )
        self.assertIsNotNone(model.decoder.ag_query.weight.grad)
        self.assertIsNotNone(model.epitope_head.weight.grad)
        self.assertIsNotNone(model.paratope_head.weight.grad)

    def test_score_top_k_must_be_a_positive_integer(self) -> None:
        from interaction_first.model import PartnerSpecificInteractionModel

        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "score_top_k"):
                    PartnerSpecificInteractionModel(
                        antigen_input_dim=8,
                        antibody_input_dim=8,
                        d_model=16,
                        pair_dim=8,
                        num_heads=4,
                        num_layers=0,
                        score_top_k=invalid,
                    )

    def test_partner_score_is_invariant_to_explicit_unary_biases(self) -> None:
        import torch

        from interaction_first.model import PartnerSpecificInteractionModel

        torch.manual_seed(19)
        model = PartnerSpecificInteractionModel(
            antigen_input_dim=8,
            antibody_input_dim=8,
            d_model=16,
            pair_dim=8,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
        ).eval()
        antigen = torch.randn(2, 5, 8)
        antibody = torch.randn(2, 4, 8)
        antigen_mask = torch.ones(2, 5, dtype=torch.bool)
        antibody_mask = torch.ones(2, 4, dtype=torch.bool)
        before = model(antigen, antibody, antigen_mask, antibody_mask)
        with torch.no_grad():
            model.decoder.ag_bias.weight.add_(3.0)
            model.decoder.ab_bias.weight.sub_(2.0)
            model.decoder.global_bias.add_(7.0)
        after = model(antigen, antibody, antigen_mask, antibody_mask)
        self.assertFalse(torch.allclose(before["pair_logits"], after["pair_logits"]))
        torch.testing.assert_close(before["complex_score"], after["complex_score"])

    def test_independent_node_heads_are_not_pair_probability_means(self) -> None:
        import torch

        from interaction_first.model import PartnerSpecificInteractionModel

        torch.manual_seed(23)
        model = PartnerSpecificInteractionModel(
            antigen_input_dim=8,
            antibody_input_dim=8,
            d_model=16,
            pair_dim=8,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        ).eval()
        antigen = torch.randn(1, 6, 8)
        antibody = torch.randn(1, 5, 8)
        outputs = model(antigen, antibody)
        self.assertEqual(outputs["epitope_probability"].shape, (1, 6))
        self.assertEqual(outputs["pair_epitope_probability"].shape, (1, 6))
        self.assertFalse(
            torch.allclose(
                outputs["epitope_probability"],
                outputs["pair_epitope_probability"],
            )
        )

    def test_joint_loss_has_finite_gradients(self) -> None:
        import torch

        from interaction_first.losses import interaction_first_loss
        from interaction_first.model import PartnerSpecificInteractionModel

        torch.manual_seed(3)
        model = PartnerSpecificInteractionModel(
            antigen_input_dim=16,
            antibody_input_dim=16,
            d_model=32,
            pair_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        antigen = torch.randn(2, 6, 16)
        antibody = torch.randn(2, 4, 16)
        antigen_mask = torch.tensor([[1] * 6, [1] * 5 + [0]], dtype=torch.bool)
        antibody_mask = torch.tensor([[1] * 4, [1] * 3 + [0]], dtype=torch.bool)
        pair_mask = antigen_mask.unsqueeze(2) & antibody_mask.unsqueeze(1)
        targets = torch.zeros(2, 6, 4)
        targets[0, 1, 2] = 1.0
        targets[1, 3, 1] = 1.0
        outputs = model(antigen, antibody, antigen_mask, antibody_mask, pair_mask)
        negative = model(antigen, antibody.flip(1), antigen_mask, antibody_mask)
        loss, _ = interaction_first_loss(
            outputs,
            targets,
            (targets.sum(dim=2) > 0).float(),
            (targets.sum(dim=1) > 0).float(),
            antigen_mask,
            antibody_mask,
            edge_positive_weight=10.0,
            epitope_positive_weight=4.0,
            paratope_positive_weight=2.0,
            negative_scores=negative["complex_score"].unsqueeze(1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
