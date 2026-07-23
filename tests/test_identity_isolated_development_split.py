"""Tests for the sequence-identity-isolated development split utility."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_identity_isolated_development_split.py"
SPEC = importlib.util.spec_from_file_location("identity_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class IdentityIsolatedSplitTest(unittest.TestCase):
    """Exercise deterministic splitting and independent leakage detection."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = {
            "a": MODULE.SequenceRecord("a", "AAAA", "KKKK"),
            "b": MODULE.SequenceRecord("b", "AAAA", "LLLL"),
            "c": MODULE.SequenceRecord("c", "CCCC", "MMMM"),
            "d": MODULE.SequenceRecord("d", "DDDD", "MMMM"),
            "e": MODULE.SequenceRecord("e", "EEEE", "NNNN"),
            "f": MODULE.SequenceRecord("f", "FFFF", "PPPP"),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_components_join_on_either_partner(self) -> None:
        antibody_hit = MODULE.SimilarityHit(
            self.records["c"].antibody_digest,
            self.records["e"].antibody_digest,
            0.8,
            1.0,
            1.0,
        )
        components, summary = MODULE.build_components(
            self.records, [antibody_hit], []
        )
        observed = {frozenset(component) for component in components}
        self.assertIn(frozenset({"a", "b"}), observed)
        self.assertIn(frozenset({"c", "d", "e"}), observed)
        self.assertEqual(summary["largest_component_size"], 3)

    def test_subset_sum_is_deterministic_and_hits_target(self) -> None:
        components = [("a", "b"), ("c", "d", "e"), ("f",), ("g",)]
        first, summary = MODULE.choose_validation_components(
            components, 3, 42, 1, 1
        )
        second, _ = MODULE.choose_validation_components(components, 3, 42, 1, 1)
        self.assertEqual(first, second)
        self.assertEqual(summary["selected_validation_instance_count"], 3)

    def test_independent_audit_detects_crossing_hit(self) -> None:
        assignments = {
            "a": "train",
            "b": "train",
            "c": "validation",
            "d": "validation",
            "e": "validation",
            "f": "validation",
        }
        crossing = MODULE.SimilarityHit(
            self.records["a"].antibody_digest,
            self.records["c"].antibody_digest,
            0.85,
            1.0,
            1.0,
        )
        audit = MODULE.audit_assignments(self.records, assignments, [crossing], [])
        self.assertFalse(audit["valid"])
        self.assertEqual(
            audit["roles"]["antibody"]["threshold_qualified_crossing_digest_pair_count"],
            1,
        )

    def test_build_command_excludes_sealed_identifiers(self) -> None:
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["INSTANCE", "vh_seq", "vl_seq", "ag_seq"],
            )
            writer.writeheader()
            for record in self.records.values():
                writer.writerow({
                    "INSTANCE": record.instance_id,
                    "vh_seq": record.antibody[:2],
                    "vl_seq": record.antibody[2:],
                    "ag_seq": record.antigen,
                })
            writer.writerow({
                "INSTANCE": "sealed",
                "vh_seq": "GG",
                "vl_seq": "GG",
                "ag_seq": "QQQQ",
            })
        source = self.root / "source.json"
        source.write_text(json.dumps({
            "splits": {
                "train": ["a", "b", "c", "d"],
                "validation": ["e", "f"],
                "test": ["sealed"],
            }
        }), encoding="utf-8")
        antibody_hits = self.root / "antibody.tsv"
        antigen_hits = self.root / "antigen.tsv"
        for role, path in (("antibody", antibody_hits), ("antigen", antigen_hits)):
            groups = MODULE.group_by_digest(self.records, role)
            with path.open("w", encoding="utf-8") as handle:
                for digest in groups:
                    handle.write(f"{digest}\t{digest}\t1.0\t1.0\t1.0\n")
        output = self.root / "output.json"
        args = MODULE.build_parser().parse_args([
            "build",
            "--manifest", str(manifest),
            "--source-split", str(source),
            "--antibody-hits", str(antibody_hits),
            "--antigen-hits", str(antigen_hits),
            "--minimum-validation-count", "1",
            "--minimum-train-count", "1",
            "--validation-fraction", "0.34",
            "--output", str(output),
        ])
        args.handler(args)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assigned = set(payload["splits"]["train"] + payload["splits"]["validation"])
        self.assertEqual(assigned, set(self.records))
        self.assertNotIn("sealed", assigned)
        self.assertTrue(payload["audit"]["valid"])
        self.assertFalse(payload["frozen_test_labels_or_predictions_read"])


if __name__ == "__main__":
    unittest.main()
