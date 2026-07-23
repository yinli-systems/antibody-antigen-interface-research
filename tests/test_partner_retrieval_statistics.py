#!/usr/bin/env python3
"""Focused tests for validation-only partner-retrieval statistics."""

from __future__ import annotations

import itertools
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sabdab2_partner_retrieval_statistics import (
    _best_positive_ranks,
    _stable_rank_matrix,
    anchor_bootstrap_intervals,
    exact_random_direction,
    fixed_degree_label_permutation_test,
    _load_development_dataset,
)


class ExactFiniteLibraryTests(unittest.TestCase):
    def test_multi_positive_expectation_matches_enumeration(self) -> None:
        result = exact_random_direction(4, [2])
        best_ranks = []
        for positive_positions in itertools.combinations(range(1, 5), 2):
            best_ranks.append(min(positive_positions))
        self.assertAlmostEqual(
            result["mrr"], np.mean([1.0 / rank for rank in best_ranks])
        )
        self.assertAlmostEqual(result["mean_rank"], np.mean(best_ranks))
        self.assertAlmostEqual(
            result["recall_at_1"], np.mean([rank <= 1 for rank in best_ranks])
        )
        self.assertAlmostEqual(
            result["recall_at_5"], 1.0
        )

    def test_heterogeneous_degrees_are_anchor_averaged(self) -> None:
        result = exact_random_direction(5, [1, 5])
        self.assertAlmostEqual(result["recall_at_1"], 0.6)
        self.assertAlmostEqual(result["mean_positives_per_anchor"], 3.0)
        self.assertEqual(result["positive_degree_histogram"], {"1": 1, "5": 1})


class RankAndInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.similarities = np.asarray(
            [
                [0.9, 0.1, 0.3],
                [0.2, 0.8, 0.4],
                [0.5, 0.6, 0.7],
            ],
            dtype=np.float64,
        )
        self.positives = np.asarray(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ],
            dtype=bool,
        )

    def test_stable_rank_matrix_uses_candidate_index_for_ties(self) -> None:
        similarities = np.asarray([[0.7, 0.7, 0.2]], dtype=np.float64)
        ranks = _stable_rank_matrix(similarities)
        np.testing.assert_array_equal(ranks, [[1, 2, 3]])
        positives = np.asarray([[False, True, False]])
        np.testing.assert_array_equal(_best_positive_ranks(ranks, positives), [2])

    def test_permutation_analysis_is_deterministic(self) -> None:
        first = fixed_degree_label_permutation_test(
            self.similarities, self.positives, 200, seed=17
        )
        second = fixed_degree_label_permutation_test(
            self.similarities, self.positives, 200, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["permutation_count"], 200)
        self.assertGreater(
            first["results"]["bidirectional"]["mrr"]["p_value"], 0.0
        )

    def test_anchor_bootstrap_is_deterministic_and_bounded(self) -> None:
        first = anchor_bootstrap_intervals(
            self.similarities, self.positives, 500, seed=23
        )
        second = anchor_bootstrap_intervals(
            self.similarities, self.positives, 500, seed=23
        )
        self.assertEqual(first, second)
        for direction in (
            "antibody_to_antigen",
            "antigen_to_antibody",
            "bidirectional",
        ):
            for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
                interval = first[direction][metric]
                self.assertLessEqual(interval["lower"], interval["estimate"])
                self.assertLessEqual(interval["estimate"], interval["upper"])


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
    "Pandas and PyArrow are required",
)
class FrozenTestIsolationTests(unittest.TestCase):
    def test_loader_filters_out_frozen_test_pair_rows(self) -> None:
        import pandas as pd

        rows = []
        splits = {"train": ["train"], "validation": ["validation"], "test": ["test"]}
        for index, instance_id in enumerate(("train", "validation", "test")):
            rows.append(
                {
                    "instance_id": instance_id,
                    "ab_sequence_sha256": hashlib.sha256(
                        f"ab:{instance_id}".encode("ascii")
                    ).hexdigest(),
                    "ag_sequence_sha256": hashlib.sha256(
                        f"ag:{instance_id}".encode("ascii")
                    ).hexdigest(),
                    "ab_length": 10 + index,
                    "ag_length": 20 + index,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_dir = root / "prepared"
            prepare_dir.mkdir()
            pd.DataFrame(rows).to_parquet(
                prepare_dir / "instances.parquet", index=False
            )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_interaction_first_v1",
                        "intended_use": "development_only",
                        "splits": splits,
                    }
                ),
                encoding="utf-8",
            )
            original_read = pd.read_parquet
            observed_filters = []

            def recording_read(*args, **kwargs):
                observed_filters.append(kwargs.get("filters"))
                return original_read(*args, **kwargs)

            with mock.patch("pandas.read_parquet", side_effect=recording_read):
                records, metadata = _load_development_dataset(
                    prepare_dir, split_path
                )

        self.assertEqual(set(records), {"train", "validation"})
        self.assertEqual(records["train"][0].instance_id, "train")
        self.assertEqual(records["validation"][0].instance_id, "validation")
        self.assertEqual(metadata["raw_instance_counts"]["test"], 1)
        self.assertEqual(len(observed_filters), 1)
        filtered_ids = set(observed_filters[0][0][2])
        self.assertEqual(filtered_ids, {"train", "validation"})
        self.assertNotIn("test", filtered_ids)


if __name__ == "__main__":
    unittest.main()
