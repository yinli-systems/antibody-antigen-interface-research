"""Tests for the development-only Needleman-Wunsch leakage audit."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "needleman_wunsch_development_audit.py"
SPEC = importlib.util.spec_from_file_location("needleman_wunsch_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NeedlemanWunschDevelopmentAuditTest(unittest.TestCase):
    """Verify metric definitions, thresholding, and the sealed-data barrier."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_metric_denominators_and_scaled_identity(self) -> None:
        metrics = MODULE.metrics_from_aligned_blocks(
            "AACCGG",
            "AATGG",
            [(0, 3), (4, 6)],
            [(0, 3), (3, 5)],
            score=7.0,
        )
        self.assertEqual(metrics.aligned_residue_pair_count, 5)
        self.assertEqual(metrics.identical_residue_pair_count, 4)
        self.assertEqual(metrics.alignment_column_count, 6)
        self.assertAlmostEqual(metrics.aligned_region_identity, 4 / 5)
        self.assertAlmostEqual(metrics.left_reciprocal_coverage, 5 / 6)
        self.assertAlmostEqual(metrics.right_reciprocal_coverage, 1.0)
        self.assertAlmostEqual(metrics.calm_scaled_similarity, 4 / 6)
        self.assertAlmostEqual(metrics.full_alignment_column_identity, 4 / 6)

    def test_rules_are_separate_and_explicit(self) -> None:
        policy = MODULE.AuditPolicy(0.8, 0.8, 0.8, 10.0, 1.0, 5)
        metrics = MODULE.AlignmentMetrics(
            score=1.0,
            identical_residue_pair_count=64,
            aligned_residue_pair_count=80,
            alignment_column_count=100,
            aligned_region_identity=0.8,
            left_reciprocal_coverage=0.8,
            right_reciprocal_coverage=0.8,
            minimum_reciprocal_coverage=0.8,
            calm_scaled_similarity=0.64,
            full_alignment_column_identity=0.64,
        )
        result = MODULE.classify_pair("a", "A" * 100, "b", "A" * 100, metrics, policy)
        self.assertTrue(result.violates_identity_and_coverage_rule)
        self.assertFalse(result.violates_calm_scaled_rule)

    def test_length_ratio_is_a_safe_upper_bound(self) -> None:
        self.assertAlmostEqual(MODULE.pair_length_upper_bound("A" * 80, "C" * 100), 0.8)
        self.assertLess(MODULE.pair_length_upper_bound("A" * 79, "C" * 100), 0.8)

    def test_sealed_identifier_is_rejected(self) -> None:
        source = self.root / "source.json"
        source.write_text(json.dumps({
            "splits": {
                "train": ["a"],
                "validation": ["b"],
                "test": ["sealed"],
            }
        }), encoding="utf-8")
        audited = self.root / "audited.json"
        audited.write_text(json.dumps({
            "splits": {
                "train": ["a", "sealed"],
                "validation": ["b"],
            }
        }), encoding="utf-8")
        development, sealed, counts = MODULE.establish_development_barrier(
            source, ["train", "validation"], ["test"]
        )
        self.assertEqual(counts, {"test": 1})
        with self.assertRaisesRegex(ValueError, "sealed identifiers"):
            MODULE.validate_audited_split(
                audited, development, sealed, "train", "validation"
            )

    def test_incomplete_development_pool_is_rejected(self) -> None:
        audited = self.root / "audited.json"
        audited.write_text(json.dumps({
            "splits": {"train": ["a"], "validation": ["b"]}
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            MODULE.validate_audited_split(
                audited, {"a", "b", "c"}, {"sealed"}, "train", "validation"
            )

    def test_named_paths_reject_duplicates_and_unsafe_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.parse_named_paths(["active=a.json", "active=b.json"])
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            MODULE.parse_named_paths(["../bad=a.json"])

    def test_inventory_hash_does_not_contain_raw_sequences(self) -> None:
        items = ((MODULE.sequence_digest("AAAA"), "AAAA"),)
        digest = MODULE._inventory_sha256(items)
        self.assertEqual(len(digest), 64)
        self.assertNotIn("AAAA", digest)

    def test_chain_loader_excludes_sealed_rows_and_keeps_chains_separate(self) -> None:
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["INSTANCE", "vh_seq", "vl_seq", "ag_seq"]
            )
            writer.writeheader()
            writer.writerow({
                "INSTANCE": "dev_a", "vh_seq": "AAAA", "vl_seq": "CCCC", "ag_seq": "GG"
            })
            writer.writerow({
                "INSTANCE": "dev_b", "vh_seq": "AAAA", "vl_seq": "", "ag_seq": "TT"
            })
            writer.writerow({
                "INSTANCE": "sealed", "vh_seq": "SSSS", "vl_seq": "QQQQ", "ag_seq": "WW"
            })
        records = MODULE.load_development_chain_records(
            manifest, {"dev_a", "dev_b"}, "INSTANCE", "vh_seq", "vl_seq"
        )
        self.assertEqual(set(records), {"dev_a", "dev_b"})
        self.assertEqual(records["dev_a"].heavy, "AAAA")
        self.assertEqual(records["dev_a"].light, "CCCC")
        self.assertEqual(records["dev_b"].light, "")
        vh = MODULE._unique_chain_sequences(records, ["dev_a", "dev_b"], "vh")
        vl = MODULE._unique_chain_sequences(records, ["dev_a", "dev_b"], "vl")
        self.assertEqual(len(vh), 1)
        self.assertEqual(len(vl), 1)
        self.assertNotEqual(vh[0][0], vl[0][0])

    def test_complete_edge_artifact_contains_no_raw_sequence(self) -> None:
        metrics = MODULE.AlignmentMetrics(
            10.0, 8, 10, 10, 0.8, 1.0, 1.0, 1.0, 0.8, 0.8
        )
        edge = MODULE.PairResult(
            "a" * 64, "b" * 64, 10, 10, metrics, True, True
        )
        role_result = {
            "_primary_crossing_edges": [edge],
            "identity_and_reciprocal_coverage_violation_count": 1,
        }
        artifact = MODULE.write_primary_edge_artifact(
            role_result, self.root, "active", "vh"
        )
        path = Path(artifact["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["train_digest"], "a" * 64)
        self.assertNotIn("AAAAAAAAAA", path.read_text(encoding="utf-8"))
        self.assertNotIn("_primary_crossing_edges", role_result)


if __name__ == "__main__":
    unittest.main()
