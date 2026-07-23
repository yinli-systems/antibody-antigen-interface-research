#!/usr/bin/env python3
"""Synthetic tests for the provisional train-only two-axis S2 builder."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import analyze_train_only_s2_feasibility as base
import build_train_only_s2_folds as builder
import optimize_train_only_s2_feasibility as optimizer


def sequence_digest(sequence: str) -> str:
    """Return the production sequence digest for a synthetic sequence."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


class TrainOnlyS2FoldBuilderTests(unittest.TestCase):
    """Verify the firewall, determinism, immutability, and full accounting."""

    def build_fixture(self, root: Path, output_name: str = "folds.json") -> Namespace:
        """Create a balanced six-edge cycle with one duplicate source row."""
        rows = [
            ("t00", "AAAA", "CCCC", "TTTT"),
            ("t00_dup", "AAAA", "CCCC", "TTTT"),
            ("t01", "AAAA", "CCCC", "VVVV"),
            ("t02", "ACDE", "FGHI", "VVVV"),
            ("t03", "ACDE", "FGHI", "WWWW"),
            ("t04", "KLMN", "PQRS", "WWWW"),
            ("t05", "KLMN", "PQRS", "TTTT"),
        ]
        authority = root / "authority.json"
        authority.write_text(
            json.dumps(
                {
                    "splits": {
                        "train": [row[0] for row in rows],
                        "validation": ["v_invalid"],
                        "test": ["x_invalid"],
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
            writer.writerows(rows)
            writer.writerow(["v_invalid", "***", "***", "***"])
            writer.writerow(["x_invalid", "***", "***", "***"])

        antibody_digests = sorted(
            {
                sequence_digest(heavy + light)
                for _, heavy, light, _ in rows
            }
        )
        antigen_digests = sorted({sequence_digest(antigen) for *_, antigen in rows})
        unauthorized_one = "0" * 64
        unauthorized_two = "f" * 64
        antibody_similarity = root / "antibody.tsv"
        antibody_similarity.write_text(
            "\n".join(
                [f"{digest}\t{digest}\t1\t1\t1" for digest in antibody_digests]
                + [
                    f"{unauthorized_one}\t{unauthorized_two}\tinvalid\tinvalid\tinvalid"
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        antigen_similarity = root / "antigen.tsv"
        antigen_similarity.write_text(
            "\n".join(
                [f"{digest}\t{digest}\t1\t1\t1" for digest in antigen_digests]
                + [
                    f"{unauthorized_one}\t{unauthorized_two}\tinvalid\tinvalid\tinvalid"
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "format_version": builder.CONFIG_FORMAT_VERSION,
                    "identity_threshold": 0.8,
                    "reciprocal_coverage_threshold": 0.8,
                    "fold_count": 3,
                    "axis_balance_tolerance": 0.0,
                    "time_limit_seconds": 30.0,
                    "similarity_backend": "precomputed_mmseqs2",
                    "optimized_sequence_views": [
                        "concatenated_antibody",
                        "antigen",
                    ],
                    "deterministic_policy": (
                        "sorted_inputs_fixed_highs_then_canonical_fold_relabel_v1"
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return Namespace(
            authority_split=authority,
            authority_sha256=base.sha256_file(authority),
            canonical_manifest=manifest,
            canonical_manifest_sha256=base.sha256_file(manifest),
            antibody_similarity_tsv=antibody_similarity,
            antibody_similarity_sha256=base.sha256_file(antibody_similarity),
            antigen_similarity_tsv=antigen_similarity,
            antigen_similarity_sha256=base.sha256_file(antigen_similarity),
            config_json=config,
            config_sha256=base.sha256_file(config),
            expected_builder_source_sha256=base.sha256_file(Path(builder.__file__)),
            expected_base_source_sha256=base.sha256_file(Path(base.__file__)),
            expected_optimizer_source_sha256=base.sha256_file(
                Path(optimizer.__file__)
            ),
            output=root / output_name,
        )

    def test_firewall_skips_invalid_sealed_sequences_and_metrics(self) -> None:
        """Sealed sequence fields and unauthorized metrics must remain unparsed."""
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            payload = builder.run(args)
            firewall = payload["data_firewall"]
            self.assertEqual(firewall["authorized_rows_materialized"], 7)
            self.assertEqual(firewall["sealed_rows_materialized"], 0)
            self.assertEqual(firewall["validation_sequence_fields_parsed"], 0)
            self.assertEqual(firewall["test_sequence_fields_parsed"], 0)
            self.assertFalse(
                firewall["accepted_label_prediction_checkpoint_or_score_inputs"]
            )
            self.assertFalse(payload["authorized_for_training"])
            self.assertFalse(payload["authorized_for_model_selection"])
            self.assertFalse(
                payload["required_downstream_gate"]["current_gate_passed"]
            )

    def test_repeated_build_is_deterministic(self) -> None:
        """Two independent publications from identical bytes must match exactly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_args = self.build_fixture(root, "first.json")
            second_args = Namespace(**vars(first_args))
            second_args.output = root / "second.json"
            first = builder.run(first_args)
            second = builder.run(second_args)
            self.assertEqual(first, second)
            self.assertEqual(first_args.output.read_bytes(), second_args.output.read_bytes())
            scientific = dict(first)
            observed = scientific.pop("scientific_payload_sha256")
            self.assertEqual(observed, base.canonical_sha256(scientific))

    def test_existing_output_is_never_overwritten(self) -> None:
        """Create-only publication must preserve an existing target byte-for-byte."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.build_fixture(root)
            builder.run(args)
            original = args.output.read_bytes()
            with self.assertRaises(FileExistsError):
                builder.run(args)
            self.assertEqual(args.output.read_bytes(), original)
            self.assertEqual(list(root.glob(f".{args.output.name}.*.staged")), [])

    def test_discarded_edge_and_source_row_ledgers_are_exact(self) -> None:
        """The six-edge cycle must retain three edges and account for all rows."""
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            payload = builder.run(args)
            ledger = payload["discarded_cross_fold_ledger"]
            accounting = payload["ledger_accounting"]
            retained_pairs = [
                pair
                for fold in payload["retained_folds"]
                for pair in fold["retained_exact_pairs"]
            ]
            retained_source_ids = {
                instance_id
                for fold in payload["retained_folds"]
                for instance_id in fold["retained_source_instance_ids"]
            }
            discarded_source_ids = {
                row["source_instance_id"]
                for row in ledger["discarded_source_rows"]
            }
            self.assertEqual(len(retained_pairs), 3)
            self.assertEqual(ledger["discarded_unique_exact_pair_count"], 3)
            self.assertEqual(
                ledger["discarded_source_row_count"],
                sum(
                    edge["source_row_count"]
                    for edge in ledger["discarded_cross_fold_edges"]
                ),
            )
            self.assertEqual(len(retained_source_ids | discarded_source_ids), 7)
            self.assertFalse(retained_source_ids & discarded_source_ids)
            discarded_pair_ids = {
                edge["exact_pair_id_sha256"]
                for edge in ledger["discarded_cross_fold_edges"]
            }
            self.assertTrue(
                all(
                    row["exact_pair_id_sha256"] in discarded_pair_ids
                    for row in ledger["discarded_source_rows"]
                )
            )
            self.assertTrue(accounting["valid"])
            self.assertEqual(accounting["all_unique_exact_pair_count"], 6)
            self.assertEqual(accounting["all_authority_train_source_row_count"], 7)
            self.assertEqual(
                accounting["discarded_unique_exact_pair_count"],
                payload["solution_summary"]["discarded_unique_exact_pair_count"],
            )
            self.assertEqual(
                accounting["discarded_source_row_count"],
                payload["solution_summary"]["discarded_source_row_count"],
            )

    def test_config_rejects_duplicate_and_nonfinite_json(self) -> None:
        """Strict JSON parsing must reject ambiguous or non-finite configurations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"format_version":"x","format_version":"y"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                builder.load_config(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text(
                '{"identity_threshold":NaN}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                builder.load_config(nonfinite)

    def test_source_or_input_hash_mismatch_fails_without_output(self) -> None:
        """All declared input and source bindings must be physically enforced."""
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_fixture(Path(directory))
            args.expected_optimizer_source_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                builder.run(args)
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
