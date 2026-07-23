#!/usr/bin/env python3
"""Synthetic tests for the exact S2 closure infeasibility receipt builder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_train_only_s2_iterative_closure.py"
SPEC = importlib.util.spec_from_file_location("s2_iterative_closure", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
AUDITOR = MODULE.auditor


class IterativeClosureTest(unittest.TestCase):
    """Verify exact-edge ingestion, transitive lifting, and infeasibility evidence."""

    def test_union_find_uses_deterministic_lexical_root(self) -> None:
        """Union order must not change the canonical component root."""
        first = MODULE.UnionFind(["c", "a", "b"])
        first.union("c", "b")
        first.union("b", "a")
        second = MODULE.UnionFind(["a", "b", "c"])
        second.union("a", "b")
        second.union("a", "c")
        self.assertEqual({first.find(value) for value in ("a", "b", "c")}, {"a"})
        self.assertEqual({second.find(value) for value in ("a", "b", "c")}, {"a"})

    def test_vh_edge_is_lifted_to_concatenated_antibody_entities(self) -> None:
        """A chain-level edge must become a same-fold antibody-entity component."""
        records = {
            "a": AUDITOR.SequenceRecord("a", "AAAA", "CCCC", "GGGG"),
            "b": AUDITOR.SequenceRecord("b", "AAAT", "DDDD", "HHHH"),
            "c": AUDITOR.SequenceRecord("c", "TTTT", "EEEE", "IIII"),
        }
        antibody_digests = {
            key: AUDITOR.sequence_sha256(record.concatenated_antibody)
            for key, record in records.items()
        }
        antigen_digests = {
            key: AUDITOR.sequence_sha256(record.antigen)
            for key, record in records.items()
        }
        antibody_assignment = {
            digest: {"cluster_id": f"antibody_cluster_{index:06d}"}
            for index, digest in enumerate(sorted(antibody_digests.values()))
        }
        antigen_assignment = {
            digest: {"cluster_id": f"antigen_cluster_{index:06d}"}
            for index, digest in enumerate(sorted(antigen_digests.values()))
        }
        vh_a = AUDITOR.sequence_sha256(records["a"].vh)
        vh_b = AUDITOR.sequence_sha256(records["b"].vh)
        edges = {
            "vh": {tuple(sorted((vh_a, vh_b)))},
            "vl": set(),
            "concatenated_antibody": set(),
            "antigen": set(),
        }
        antibody_root, antigen_root, exact_pairs, evidence = (
            MODULE.build_exact_closure_components(
                records, antibody_assignment, antigen_assignment, edges
            )
        )
        self.assertEqual(
            antibody_root[antibody_digests["a"]],
            antibody_root[antibody_digests["b"]],
        )
        self.assertNotEqual(
            antibody_root[antibody_digests["a"]],
            antibody_root[antibody_digests["c"]],
        )
        self.assertEqual(len(set(antigen_root.values())), 3)
        summary, witness = MODULE.component_summary(
            antibody_root, exact_pairs, "antibody"
        )
        self.assertEqual(summary["component_count"], 2)
        self.assertEqual(witness["degree"], 2)
        self.assertEqual(len(witness["members"]), 2)
        self.assertEqual(evidence["ingested_exact_edge_counts"]["vh"], 1)

    def test_exact_chain_digest_equivalence_is_a_hard_component_constraint(self) -> None:
        """Two concat entities sharing an exact VH must remain in one component."""
        records = {
            "a": AUDITOR.SequenceRecord("a", "AAAA", "CCCC", "GGGG"),
            "b": AUDITOR.SequenceRecord("b", "AAAA", "DDDD", "HHHH"),
            "c": AUDITOR.SequenceRecord("c", "TTTT", "EEEE", "IIII"),
        }
        antibody_assignment = {
            AUDITOR.sequence_sha256(record.concatenated_antibody): {
                "cluster_id": f"antibody_cluster_{index:06d}"
            }
            for index, record in enumerate(records.values())
        }
        antigen_assignment = {
            AUDITOR.sequence_sha256(record.antigen): {
                "cluster_id": f"antigen_cluster_{index:06d}"
            }
            for index, record in enumerate(records.values())
        }
        empty_edges = {view: set() for view in MODULE.REQUIRED_VIEWS}
        antibody_root, _, _, _ = MODULE.build_exact_closure_components(
            records, antibody_assignment, antigen_assignment, empty_edges
        )
        digest_a = AUDITOR.sequence_sha256(records["a"].concatenated_antibody)
        digest_b = AUDITOR.sequence_sha256(records["b"].concatenated_antibody)
        self.assertEqual(antibody_root[digest_a], antibody_root[digest_b])

    def _edge_row(self) -> dict[str, object]:
        """Return one exact threshold synthetic VH edge."""
        metrics = AUDITOR.AlignmentMetrics(10.0, 4, 5, 5, 0.8, 1.0, 1.0, 1.0)
        return AUDITOR.edge_payload(
            "vh",
            0,
            1,
            AUDITOR.sequence_sha256("AAAAA"),
            AUDITOR.sequence_sha256("AAAAT"),
            "AAAAA",
            "AAAAT",
            metrics,
        )

    def test_edge_loader_rejects_duplicates_and_below_threshold_tamper(self) -> None:
        """Every ingested relation must be unique and meet exact integer thresholds."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.jsonl"
            valid_path.write_text(
                json.dumps(self._edge_row(), sort_keys=True) + "\n", encoding="ascii"
            )
            edges, reference = MODULE.load_edge_artifact(
                valid_path, "vh", AUDITOR.sha256_file(valid_path), 1
            )
            self.assertEqual(len(edges), 1)
            self.assertTrue(reference["all_rows_digest_only_and_threshold_verified"])

            duplicate_path = root / "duplicate.jsonl"
            row = json.dumps(self._edge_row(), sort_keys=True) + "\n"
            duplicate_path.write_text(row + row, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate relation"):
                MODULE.load_edge_artifact(
                    duplicate_path, "vh", AUDITOR.sha256_file(duplicate_path), 2
                )

            tampered = self._edge_row()
            tampered["alignment"]["identical_residue_pair_count"] = 3
            tampered_path = root / "tampered.jsonl"
            tampered_path.write_text(
                json.dumps(tampered, sort_keys=True) + "\n", encoding="ascii"
            )
            with self.assertRaisesRegex(ValueError, "fails the locked exact threshold"):
                MODULE.load_edge_artifact(
                    tampered_path, "vh", AUDITOR.sha256_file(tampered_path), 1
                )

    def test_config_rejects_bound_relaxation(self) -> None:
        """Integer bounds and the no-relaxation flag are immutable."""
        source = (
            Path(__file__).parents[2]
            / "outputs/interaction_first/research/train_only_s2_iterative_closure/closure_config_v2.json"
        )
        config = MODULE.load_config(source)
        self.assertEqual(config["integer_axis_degree_lower_bound"], 580)
        self.assertEqual(config["integer_axis_degree_upper_bound"], 640)
        with tempfile.TemporaryDirectory() as directory:
            relaxed = dict(config)
            relaxed["integer_axis_degree_upper_bound"] = 1704
            path = Path(directory) / "relaxed.json"
            path.write_text(json.dumps(relaxed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "580..640"):
                MODULE.load_config(path)


if __name__ == "__main__":
    unittest.main()
