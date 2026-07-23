#!/usr/bin/env python3
"""Safety and serialization tests for the antigen-only NW fold audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_train_only_antigen_folds as module
import build_train_only_antigen_folds as builder


class SourceTrainSequenceTests(unittest.TestCase):
    """Verify sequence parsing is gated by source-train membership."""

    def test_malicious_sealed_sequence_is_never_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.csv"
            path.write_text(
                "INSTANCE,ag_seq\ntrain_a,ACDE\nsealed_test,INVALID!\n",
                encoding="utf-8",
            )
            records = module.load_source_train_antigens(path, {"train_a"})
            self.assertEqual(records, {"train_a": "ACDE"})


class FoldBarrierTests(unittest.TestCase):
    """Verify the fold index pin and sealed-ID rejection."""

    def _write_fixture(self, root: Path, include_sealed: bool) -> Path:
        validation = ["b", "sealed"] if include_sealed else ["b"]
        folds = []
        memberships = [
            (["a", "c"], validation),
            (["a", "b"], ["c"]),
            (["b", "c"], ["a"]),
        ]
        for index, (train, held_out) in enumerate(memberships):
            path = root / f"fold_{index}.json"
            payload = {
                "format_version": builder.FOLD_FORMAT_VERSION,
                "splits": {"train": train, "validation": held_out},
            }
            path.write_bytes(builder.canonical_json_bytes(payload))
            folds.append(
                {
                    "fold_index": index,
                    "path": path.name,
                    "sha256": builder.sha256_file(path),
                }
            )
        index_path = root / "index.json"
        index_path.write_bytes(
            builder.canonical_json_bytes(
                {"format_version": builder.INDEX_FORMAT_VERSION, "folds": folds}
            )
        )
        return index_path

    def test_valid_three_fold_partition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fixture(Path(temporary), include_sealed=False)
            folds, summary = module.load_verified_folds(
                path,
                builder.sha256_file(path),
                {"a", "b", "c"},
                {"sealed"},
            )
            self.assertEqual(len(folds), 3)
            self.assertTrue(summary["validation_membership_exactly_once"])

    def test_malicious_sealed_fold_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fixture(Path(temporary), include_sealed=True)
            with self.assertRaisesRegex(PermissionError, "sealed"):
                module.load_verified_folds(
                    path,
                    builder.sha256_file(path),
                    {"a", "b", "c"},
                    {"sealed"},
                )


class EdgeTests(unittest.TestCase):
    """Verify cumulative edge merging and deterministic serialization."""

    def test_strongest_duplicate_edge_is_retained(self) -> None:
        weak = builder.SimilarityHit("a", "b", 0.81, 0.9, 0.9, "weak")
        strong = builder.SimilarityHit("a", "b", 0.92, 0.9, 0.9, "strong")
        merged = module.merge_edges([weak], [strong])
        self.assertEqual(merged, [strong])
        fields = module.edge_tsv_bytes(merged).decode("ascii").strip().split("\t")
        self.assertEqual(fields[:2], ["a", "b"])
        self.assertAlmostEqual(float(fields[2]), 0.92)


if __name__ == "__main__":
    unittest.main()
