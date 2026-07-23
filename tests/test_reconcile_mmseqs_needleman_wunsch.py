"""Tests for MMseqs2 and Needleman-Wunsch edge reconciliation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconcile_mmseqs_needleman_wunsch.py"
SPEC = importlib.util.spec_from_file_location("nw_mmseqs_reconcile", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReconcileNeedlemanWunschTest(unittest.TestCase):
    """Exercise every mutually exclusive reconciliation category."""

    def setUp(self) -> None:
        self.edge = {
            "train_digest": "a",
            "validation_digest": "b",
            "metrics": {"calm_scaled_similarity": 0.7},
        }

    def classify(self, rows, digests={"a", "b"}):
        """Classify one toy edge at the production thresholds."""
        return MODULE.reconcile_edge(self.edge, digests, rows, 0.8, 0.8)

    def test_no_row_and_mapping_failure(self) -> None:
        self.assertEqual(
            self.classify([])["reconciliation_category"],
            "no_nonself_mmseqs_row",
        )
        self.assertEqual(
            self.classify([], {"a"})["reconciliation_category"],
            "digest_mapping_failure",
        )

    def test_identity_and_coverage_failures(self) -> None:
        low_identity = MODULE.MmseqsRow("a", "b", 0.79, 1.0, 1.0)
        low_coverage = MODULE.MmseqsRow("a", "b", 0.9, 0.79, 1.0)
        self.assertEqual(
            self.classify([low_identity])["reconciliation_category"],
            "mmseqs_identity_below_threshold",
        )
        self.assertEqual(
            self.classify([low_coverage])["reconciliation_category"],
            "mmseqs_reciprocal_coverage_below_threshold",
        )

    def test_qualified_row_and_direction(self) -> None:
        forward = MODULE.MmseqsRow("a", "b", 0.8, 0.8, 0.8)
        reverse = MODULE.MmseqsRow("b", "a", 0.8, 0.8, 0.8)
        result = self.classify([forward, reverse])
        self.assertEqual(
            result["reconciliation_category"],
            "threshold_qualified_row_present_inconsistency",
        )
        self.assertEqual(result["direction_status"], "both_directions_present")


if __name__ == "__main__":
    unittest.main()
