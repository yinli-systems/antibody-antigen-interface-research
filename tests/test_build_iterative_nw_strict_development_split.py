"""Tests for iterative exact-alignment development splitting."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_iterative_nw_strict_development_split.py"
SPEC = importlib.util.spec_from_file_location("iterative_nw_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IterativeNeedlemanWunschSplitTest(unittest.TestCase):
    """Verify graph evidence and feasible holdout accounting."""

    def test_hit_merging_is_orientation_independent(self) -> None:
        weak = MODULE.SimilarityHit("b", "a", 0.8, 0.8, 0.8)
        strong = MODULE.SimilarityHit("a", "b", 0.9, 0.9, 0.9)
        merged = MODULE._merge_hits([weak, strong])
        self.assertEqual(set(merged), {("a", "b")})
        self.assertEqual(merged[("a", "b")].identity, 0.9)

    def test_reachable_sizes_report_exact_support(self) -> None:
        components = [("a", "b", "c"), ("d", "e"), ("f",)]
        summary = MODULE.reachable_component_sizes(components, minimum_train_count=2)
        self.assertEqual(summary["minimum_feasible_validation_instance_count"], 1)
        self.assertEqual(summary["maximum_feasible_validation_instance_count"], 4)
        self.assertEqual(summary["feasible_values"], [1, 2, 3, 4])

    def test_pair_result_conversion_preserves_declared_denominators(self) -> None:
        metrics = MODULE.AlignmentMetrics(
            1.0, 8, 10, 12, 0.8, 1.0, 0.8, 0.8, 0.64, 8 / 12
        )
        edge = MODULE.PairResult("a", "b", 10, 12, metrics, True, False)
        hit = MODULE.pair_result_to_hit(edge)
        self.assertEqual(hit.identity, 0.8)
        self.assertEqual(hit.query_coverage, 1.0)
        self.assertEqual(hit.target_coverage, 0.8)

    def test_chain_constraints_join_instances_independently_of_concat(self) -> None:
        records = {
            "a": MODULE.SequenceRecord("a", "AAAACCCC", "KKKK"),
            "b": MODULE.SequenceRecord("b", "GGGGTTTT", "LLLL"),
        }
        chains = {
            "a": SimpleNamespace(heavy="AAAA", light="CCCC"),
            "b": SimpleNamespace(heavy="GGGG", light="TTTT"),
        }
        vh_hit = MODULE.SimilarityHit(
            MODULE.sequence_digest("AAAA"),
            MODULE.sequence_digest("GGGG"),
            0.8,
            1.0,
            1.0,
        )
        components, summary = MODULE.build_extended_components(
            records, [], [], chains, [vh_hit], []
        )
        self.assertEqual(components, [("a", "b")])
        self.assertTrue(summary["includes_vh_and_vl_constraints"])


if __name__ == "__main__":
    unittest.main()
