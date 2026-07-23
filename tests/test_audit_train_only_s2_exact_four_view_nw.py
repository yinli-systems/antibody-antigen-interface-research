#!/usr/bin/env python3
"""Synthetic tests for the independent train-only S2 exact NW auditor."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_train_only_s2_exact_four_view_nw.py"
SPEC = importlib.util.spec_from_file_location("s2_exact_nw_auditor", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeAlignment:
    """Minimal full-length alignment compatible with the production metric reader."""

    def __init__(self, left: str, right: str) -> None:
        width = min(len(left), len(right))
        self.aligned = (
            ((0, width),),
            ((0, width),),
        )
        self.score = float(sum(a == b for a, b in zip(left, right)))


class FakeAligner:
    """Deterministic synthetic aligner for publication and accounting tests."""

    def align(self, left: str, right: str) -> list[FakeAlignment]:
        """Return one full-length synthetic alignment."""
        return [FakeAlignment(left, right)]


class ExactFourViewAuditorTest(unittest.TestCase):
    """Verify firewall, parity, threshold, tamper, and publication invariants."""

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        """Ambiguous and non-finite JSON must fail before semantic parsing."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"outer":{"x":1,"x":2}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                MODULE.strict_json_object(duplicate, "duplicate")
            constant = root / "constant.json"
            constant.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.strict_json_object(constant, "constant")
            overflow = root / "overflow.json"
            overflow.write_text('{"value":1e999}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.strict_json_object(overflow, "overflow")

    def test_firewall_skips_invalid_sealed_sequence_fields(self) -> None:
        """Sealed rows must be excluded before any sequence field is parsed."""
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            manifest.write_text(
                "INSTANCE,vh_seq,vl_seq,ag_seq,score\n"
                "train_a,AAAA,CCCC,GGGG,not-a-score\n"
                "sealed_validation,INVALID,***,NaN,NaN\n"
                "train_b,TTTT,,WWWW,also-not-a-score\n"
                "sealed_test,???,BROKEN,INF,INF\n",
                encoding="utf-8",
            )
            records, firewall = MODULE.load_authorized_sequences(
                manifest,
                {"train_a", "train_b"},
                {"sealed_validation", "sealed_test"},
            )
            self.assertEqual(set(records), {"train_a", "train_b"})
            self.assertEqual(records["train_b"].vl, "")
            self.assertEqual(firewall["authorized_rows_materialized"], 2)
            self.assertEqual(firewall["sealed_rows_materialized"], 0)
            self.assertEqual(firewall["validation_sequence_fields_parsed"], 0)
            self.assertEqual(firewall["test_sequence_fields_parsed"], 0)
            self.assertEqual(
                firewall["non_authorized_rows_skipped_before_sequence_parse"], 2
            )

    def test_scientific_digest_detects_assignment_tampering(self) -> None:
        """Scientific content and the external digest pin must both authenticate."""
        scientific = {"format_version": "synthetic", "count": 3}
        digest = MODULE.canonical_payload_sha256(scientific)
        assignment = {**scientific, "scientific_payload_sha256": digest}
        self.assertEqual(
            MODULE._validate_assignment_scientific_digest(assignment, digest), digest
        )
        tampered = dict(assignment)
        tampered["count"] = 4
        with self.assertRaisesRegex(ValueError, "mismatch"):
            MODULE._validate_assignment_scientific_digest(tampered, digest)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            MODULE._validate_assignment_scientific_digest(assignment, "0" * 64)

    def test_physical_input_hash_detects_tampering(self) -> None:
        """Every source and input hash must be checked against physical bytes."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.bin"
            path.write_bytes(b"original")
            expected = MODULE.sha256_file(path)
            reference = MODULE.verify_file(path, expected, "synthetic input")
            self.assertEqual(reference["sha256"], expected)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                MODULE.verify_file(path, expected, "synthetic input")

    def test_metric_denominators_match_locked_definitions(self) -> None:
        """Gap columns must be excluded from identity and retained in raw lengths."""
        metrics = MODULE.metrics_from_aligned_blocks(
            "AACCGG",
            "AATGG",
            ((0, 3), (4, 6)),
            ((0, 3), (3, 5)),
            7.0,
        )
        self.assertEqual(metrics.identical_residue_pair_count, 4)
        self.assertEqual(metrics.aligned_residue_pair_count, 5)
        self.assertEqual(metrics.alignment_column_count, 6)
        self.assertAlmostEqual(metrics.aligned_region_identity, 4 / 5)
        self.assertAlmostEqual(metrics.left_reciprocal_coverage, 5 / 6)
        self.assertAlmostEqual(metrics.right_reciprocal_coverage, 1.0)

    def test_threshold_equality_passes_and_one_count_below_fails(self) -> None:
        """The 0.8 boundary must be inclusive and compared with exact integers."""
        exact = MODULE.AlignmentMetrics(0.0, 80, 100, 100, 0.8, 0.8, 0.8, 0.8)
        self.assertTrue(MODULE.meets_locked_threshold(exact, 100, 100))
        below_identity = MODULE.AlignmentMetrics(
            0.0, 79, 100, 100, 0.79, 0.8, 0.8, 0.8
        )
        self.assertFalse(
            MODULE.meets_locked_threshold(below_identity, 100, 100)
        )
        below_coverage = MODULE.AlignmentMetrics(
            0.0, 79, 99, 121, 79 / 99, 0.99, 0.792, 0.792
        )
        self.assertFalse(
            MODULE.meets_locked_threshold(below_coverage, 100, 125)
        )

    def test_length_upper_bound_prunes_only_strictly_below_threshold(self) -> None:
        """A length ratio equal to 0.8 remains dynamically alignable."""
        self.assertTrue(MODULE.is_safe_length_prune(79, 100))
        self.assertFalse(MODULE.is_safe_length_prune(80, 100))
        self.assertFalse(MODULE.is_safe_length_prune(100, 100))

    def test_missing_vl_is_excluded_without_placeholder(self) -> None:
        """Empty VL contributes no VL sequence but does not remove other views."""
        records = {
            "a": MODULE.SequenceRecord("a", "AAAA", "", "GGGG"),
            "b": MODULE.SequenceRecord("b", "CCCC", "TTTT", "WWWW"),
            "c": MODULE.SequenceRecord("c", "DDDD", "SSSS", "YYYY"),
            "d": MODULE.SequenceRecord("d", "EEEE", "NNNN", "QQQQ"),
        }
        inventories = MODULE.build_fold_inventories(
            records, {"a": 0, "b": 0, "c": 1, "d": 2}
        )
        self.assertEqual(len(inventories["vl"][0]), 1)
        self.assertEqual(len(inventories["vh"][0]), 2)
        self.assertEqual(len(inventories["concatenated_antibody"][0]), 2)

    def test_edge_payload_is_digest_only(self) -> None:
        """Crossing artifacts must never contain sequences or source identifiers."""
        metrics = MODULE.AlignmentMetrics(10.0, 4, 5, 5, 0.8, 1.0, 1.0, 1.0)
        payload = MODULE.edge_payload(
            "vh",
            0,
            1,
            MODULE.sequence_sha256("AAAAA"),
            MODULE.sequence_sha256("AAAAT"),
            "AAAAA",
            "AAAAT",
            metrics,
        )
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("AAAAA", encoded)
        self.assertNotIn("AAAAT", encoded)
        self.assertNotIn("source_instance_id", payload)
        self.assertNotIn("instance_id", payload)
        self.assertFalse(payload["raw_sequences_or_instance_ids_included"])

    def test_exhaustive_streaming_accounting_and_crossing_publication(self) -> None:
        """Every synthetic cross-fold pair must be aligned and counted once."""
        folds = (
            ((MODULE.sequence_sha256("AAAAA"), "AAAAA"),),
            ((MODULE.sequence_sha256("AAAAT"), "AAAAT"),),
            ((MODULE.sequence_sha256("CCCCC"), "CCCCC"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vh.jsonl"
            with mock.patch.object(
                MODULE, "build_locked_aligner", return_value=FakeAligner()
            ):
                summary, artifact = MODULE.audit_view_to_staged_artifact(
                    "vh", folds, target, workers=1, chunk_size=1
                )
            try:
                self.assertEqual(
                    summary["total_cross_fold_unique_sequence_pair_count"], 3
                )
                self.assertEqual(
                    summary["total_dynamic_programming_alignment_count"], 3
                )
                self.assertEqual(
                    summary["total_provably_pruned_by_length_upper_bound_count"], 0
                )
                self.assertEqual(summary["total_threshold_crossing_edge_count"], 1)
                rows = artifact.temporary_path.read_text(encoding="ascii").splitlines()
                self.assertEqual(len(rows), 1)
                self.assertNotIn("AAAAA", rows[0])
            finally:
                artifact.temporary_path.unlink(missing_ok=True)

    def test_approval_requires_zero_crossings_in_all_exact_views(self) -> None:
        """Missing views or one positive crossing must prevent both approvals."""
        zero = {
            view: {
                "valid_zero_crossing_edges": True,
                "total_threshold_crossing_edge_count": 0,
            }
            for view in MODULE.REQUIRED_VIEWS
        }
        self.assertTrue(MODULE.approval_from_results(zero))
        missing = dict(zero)
        missing.pop("vl")
        self.assertFalse(MODULE.approval_from_results(missing))
        crossing = {view: dict(result) for view, result in zero.items()}
        crossing["antigen"] = {
            "valid_zero_crossing_edges": False,
            "total_threshold_crossing_edge_count": 1,
        }
        self.assertFalse(MODULE.approval_from_results(crossing))

    def test_no_clobber_publication_preserves_existing_bytes(self) -> None:
        """A second publisher must fail without changing the first artifact."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.json"
            first = MODULE.stage_bytes(target, b"first\n")
            second = MODULE.stage_bytes(target, b"second\n")
            try:
                MODULE.publish_staged_no_clobber(first)
                with self.assertRaises(FileExistsError):
                    MODULE.publish_staged_no_clobber(second)
                self.assertEqual(target.read_bytes(), b"first\n")
            finally:
                first.temporary_path.unlink(missing_ok=True)
                second.temporary_path.unlink(missing_ok=True)

    @unittest.skipUnless(
        importlib.util.find_spec("Bio") is not None,
        "Biopython is not installed in the local test runtime",
    )
    def test_pinned_biopython_synthetic_traceback_parity(self) -> None:
        """Production alignment must match manually obvious synthetic metrics."""
        aligner = MODULE.build_locked_aligner()
        metrics = MODULE.global_alignment_metrics("AAAAA", "AAAAT", aligner)
        self.assertEqual(metrics.identical_residue_pair_count, 4)
        self.assertEqual(metrics.aligned_residue_pair_count, 5)
        self.assertAlmostEqual(metrics.aligned_region_identity, 0.8)
        self.assertEqual(metrics.left_reciprocal_coverage, 1.0)
        self.assertEqual(metrics.right_reciprocal_coverage, 1.0)
        self.assertTrue(MODULE.meets_locked_threshold(metrics, 5, 5))


if __name__ == "__main__":
    unittest.main()
