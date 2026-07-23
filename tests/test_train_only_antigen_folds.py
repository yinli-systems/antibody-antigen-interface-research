#!/usr/bin/env python3
"""Tests for the train-only antigen-cluster fold builder."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_train_only_antigen_folds as module


class AuthoritySafetyTests(unittest.TestCase):
    """Verify the source-train authorization and sealed-data boundary."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.authority = self.root / "authority.json"
        self.authority.write_text(
            json.dumps(
                {
                    "intended_use": "development_only",
                    "splits": {
                        "train": ["a", "b", "c"],
                        "validation": ["sealed_validation"],
                        "test": ["sealed_test"],
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authority_hash_and_membership_are_pinned(self) -> None:
        digest = module.sha256_file(self.authority)
        allowed, sealed, summary = module.load_authority(self.authority, digest)
        self.assertEqual(allowed, {"a", "b", "c"})
        self.assertEqual(sealed, {"sealed_validation", "sealed_test"})
        self.assertEqual(summary["sealed_split_counts"], {"validation": 1, "test": 1})
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            module.load_authority(self.authority, "0" * 64)

    def test_malicious_sealed_id_is_rejected_before_record_access(self) -> None:
        records = {
            "a": module.PairRecord("a", "ab_a", "ag_a"),
            "b": module.PairRecord("b", "ab_b", "ag_b"),
            "c": module.PairRecord("c", "ab_c", "ag_c"),
        }
        components = {"ag_a": "one", "ag_b": "two", "ag_c": "three"}
        with self.assertRaisesRegex(PermissionError, "sealed"):
            module.audit_fold(
                ["a", "sealed_test"],
                ["b", "c"],
                {"a", "b", "c"},
                {"sealed_validation", "sealed_test"},
                records,
                components,
                [],
                [],
            )

    def test_canonical_parser_skips_malicious_sealed_sequences(self) -> None:
        manifest = self.root / "canonical.csv"
        manifest.write_text(
            "INSTANCE,vh_seq,vl_seq,ag_seq\n"
            "a,ACD,EFG,HIK\n"
            "sealed_test,INVALID!,INVALID!,INVALID!\n",
            encoding="utf-8",
        )
        records = module.load_source_train_records(manifest, {"a"})
        self.assertEqual(set(records), {"a"})

    def test_similarity_parser_filters_before_sealed_metrics(self) -> None:
        evidence = self.root / "evidence.tsv"
        evidence.write_text(
            "ag_a\tag_a\t1.0\t1.0\t1.0\n"
            "sealed_digest\tsealed_digest\tnot-a-number\tnot-a-number\tnot-a-number\n",
            encoding="utf-8",
        )
        edges, summary = module.load_similarity_edges(
            evidence,
            {"ag_a"},
            0.8,
            0.8,
            "test",
            require_complete_inventory=True,
        )
        self.assertEqual(edges, [])
        self.assertEqual(summary["authorized_row_count"], 1)
        self.assertTrue(summary["filter_before_metric_parsing"])


class ComponentAndFoldTests(unittest.TestCase):
    """Verify component isolation, balancing, and antibody-overlap auditing."""

    def setUp(self) -> None:
        self.records = {
            "a1": module.PairRecord("a1", "shared_ab", "ag_1"),
            "a2": module.PairRecord("a2", "shared_ab", "ag_1"),
            "b1": module.PairRecord("b1", "shared_ab", "ag_2"),
            "b2": module.PairRecord("b2", "ab_2", "ag_2"),
            "c1": module.PairRecord("c1", "shared_ab", "ag_3"),
            "c2": module.PairRecord("c2", "ab_3", "ag_3"),
            "d1": module.PairRecord("d1", "ab_4", "ag_4"),
            "d2": module.PairRecord("d2", "ab_4", "ag_4"),
        }

    def test_three_folds_exactly_cover_pool_without_antigen_crossing(self) -> None:
        components, mapping = module.build_antigen_components(self.records, [])
        assignments, summary = module.balance_components(components, 3, 42)
        self.assertEqual(set(assignments), {component.key for component in components})
        self.assertEqual(len(summary["validation_exact_pair_counts"]), 3)

        allowed = set(self.records)
        seen_validation: set[str] = set()
        component_by_key = {component.key: component for component in components}
        antibody_edge = module.SimilarityHit(
            "ab_2", "shared_ab", 0.9, 1.0, 1.0, "test"
        )
        for fold_index in range(3):
            validation = sorted(
                instance_id
                for key, assigned in assignments.items()
                if assigned == fold_index
                for instance_id in component_by_key[key].instance_ids
            )
            train = sorted(allowed - set(validation))
            audit = module.audit_fold(
                train,
                validation,
                allowed,
                {"sealed"},
                self.records,
                mapping,
                [],
                [antibody_edge],
            )
            self.assertTrue(audit["valid"])
            self.assertEqual(
                audit["antigen_isolation"]["exact_sequence_crossing_digest_count"],
                0,
            )
            self.assertEqual(
                audit["antigen_isolation"]["threshold_qualified_crossing_edge_count"],
                0,
            )
            self.assertTrue(audit["antibody_overlap"]["allowed_by_policy"])
            seen_validation.update(validation)
        self.assertEqual(seen_validation, allowed)

    def test_similarity_connected_antigens_stay_in_one_component(self) -> None:
        edge = module.SimilarityHit("ag_1", "ag_2", 0.9, 0.9, 0.9, "test")
        components, mapping = module.build_antigen_components(self.records, [edge])
        self.assertEqual(mapping["ag_1"], mapping["ag_2"])
        connected = next(value for value in components if value.key == mapping["ag_1"])
        self.assertEqual(set(connected.instance_ids), {"a1", "a2", "b1", "b2"})


class OutputContractTests(unittest.TestCase):
    """Verify immutable metadata publication semantics."""

    def test_existing_output_requires_identical_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold.json"
            module.atomic_create_or_validate(path, b"first", resume=False)
            with self.assertRaises(FileExistsError):
                module.atomic_create_or_validate(path, b"first", resume=False)
            module.atomic_create_or_validate(path, b"first", resume=True)
            with self.assertRaisesRegex(ValueError, "differs"):
                module.atomic_create_or_validate(path, b"second", resume=True)
            self.assertEqual(path.read_bytes(), b"first")

    def test_cli_rejects_non_three_fold_protocol(self) -> None:
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--authority-split",
                    "authority.json",
                    "--canonical-manifest",
                    "canonical.csv",
                    "--antigen-similarity-tsv",
                    "antigen.tsv",
                    "--output-dir",
                    "output",
                    "--fold-count",
                    "2",
                ]
            )


if __name__ == "__main__":
    unittest.main()
