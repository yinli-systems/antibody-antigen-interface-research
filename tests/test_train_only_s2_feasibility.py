#!/usr/bin/env python3
"""Tests for the train-only S2 feasibility diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from analyze_train_only_s2_feasibility import canonical_sha256, run, sha256_file


def sequence_digest(value: str) -> str:
    """Return the synthetic sequence digest used by the production code."""
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class TrainOnlyS2FeasibilityTests(unittest.TestCase):
    """Exercise the data firewall, graph calculation, and immutable output."""

    def build_fixture(self, root: Path) -> Namespace:
        """Create a small two-axis graph with deliberately invalid sealed rows."""
        authority = root / "authority.json"
        authority.write_text(
            json.dumps(
                {
                    "splits": {
                        "train": ["t1", "t2", "t3"],
                        "validation": ["v1"],
                        "test": ["x1"],
                    }
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["INSTANCE", "vh_seq", "vl_seq", "ag_seq"])
            writer.writerow(["t1", "AAAA", "CCCC", "GGGG"])
            writer.writerow(["t2", "AAAA", "CCCC", "TTTT"])
            writer.writerow(["t3", "ACDE", "", "GGGG"])
            writer.writerow(["v1", "***", "***", "***"])
            writer.writerow(["x1", "***", "***", "***"])

        antibody_one = sequence_digest("AAAACCCC")
        antibody_two = sequence_digest("ACDE")
        antigen_one = sequence_digest("GGGG")
        antigen_two = sequence_digest("TTTT")
        antibody_similarity = root / "antibody.tsv"
        antibody_similarity.write_text(
            "\n".join(
                [
                    f"{antibody_one}\t{antibody_one}\t1\t1\t1",
                    f"{antibody_two}\t{antibody_two}\t1\t1\t1",
                    f"{antibody_one}\t{antibody_two}\t0.9\t1\t1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        antigen_similarity = root / "antigen.tsv"
        antigen_similarity.write_text(
            "\n".join(
                [
                    f"{antigen_one}\t{antigen_one}\t1\t1\t1",
                    f"{antigen_two}\t{antigen_two}\t1\t1\t1",
                    f"{antigen_one}\t{antigen_two}\t0.9\t1\t1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return Namespace(
            authority_split=authority,
            authority_sha256=sha256_file(authority),
            canonical_manifest=manifest,
            canonical_manifest_sha256=sha256_file(manifest),
            antibody_similarity_tsv=antibody_similarity,
            antibody_similarity_sha256=sha256_file(antibody_similarity),
            antigen_similarity_tsv=antigen_similarity,
            antigen_similarity_sha256=sha256_file(antigen_similarity),
            identity_threshold=0.8,
            coverage_threshold=0.8,
            output=root / "diagnostic.json",
        )

    def test_whole_component_rejection_and_train_only_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            payload = run(args)
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["data_firewall"]["authorized_rows_materialized"], 3)
            self.assertEqual(payload["data_firewall"]["sealed_rows_materialized"], 0)
            analysis = payload["bipartite_component_analysis"]
            self.assertEqual(analysis["component_count_with_interactions"], 1)
            self.assertEqual(analysis["largest_component"]["source_row_count"], 3)
            self.assertEqual(analysis["largest_component"]["unique_exact_pair_count"], 3)
            self.assertFalse(analysis["whole_component_three_fold_feasible"])
            scientific_payload = dict(payload)
            observed_digest = scientific_payload.pop("scientific_payload_sha256")
            self.assertEqual(observed_digest, canonical_sha256(scientific_payload))

    def test_input_hash_tampering_fails_before_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            args.canonical_manifest_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run(args)
            self.assertFalse(args.output.exists())

    def test_output_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            run(args)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
