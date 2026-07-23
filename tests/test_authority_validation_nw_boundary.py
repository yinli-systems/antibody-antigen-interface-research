#!/usr/bin/env python3
"""Unit tests for the authority-validation-only boundary builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_authority_validation_nw_boundary import run


def _sha256(path: Path) -> str:
    """Return a small test file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuthorityValidationBoundaryTest(unittest.TestCase):
    """Exercise deterministic component inclusion and sealed-row gating."""

    def test_all_pure_components_are_retained(self) -> None:
        """A component with a train edge is excluded while every pure one remains."""
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            authority = directory / "authority.json"
            authority.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": ["train_1"],
                            "validation": ["validation_clean", "validation_crossing"],
                            "test": ["sealed_1"],
                        },
                        "components": [
                            {"component_id": "c0", "members": ["train_1"]},
                            {
                                "component_id": "c1",
                                "members": ["validation_clean"],
                            },
                            {
                                "component_id": "c2",
                                "members": ["validation_crossing"],
                            },
                            {"component_id": "c3", "members": ["sealed_1"]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = directory / "manifest.csv"
            manifest.write_text(
                "INSTANCE,vh_seq,vl_seq,ag_seq\n"
                "train_1,AAAA,,GGGG\n"
                "validation_clean,CCCC,,TTTT\n"
                "validation_crossing,AAAT,,TTTA\n"
                "sealed_1,123,456,789\n",
                encoding="utf-8",
            )
            edge_dir = directory / "edges"
            edge_dir.mkdir()
            train_digest = hashlib.sha256(b"AAAA").hexdigest()
            crossing_digest = hashlib.sha256(b"AAAT").hexdigest()
            crossing_edge = edge_dir / "active.concat_antibody.identity_coverage_edges.jsonl"
            crossing_edge.write_text(
                json.dumps(
                    {
                        "train_digest": train_digest,
                        "validation_digest": crossing_digest,
                        "violates_identity_and_coverage_rule": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            empty_edge = edge_dir / "active.antigen.identity_coverage_edges.jsonl"
            empty_edge.write_text("", encoding="utf-8")
            report = directory / "audit.json"
            report.write_text(
                json.dumps(
                    {
                        "frozen_test_sequences_aligned": False,
                        "scientific_payload_sha256": "audit-science",
                        "provenance": {
                            "development_source_split": {"sha256": _sha256(authority)}
                        },
                        "results": {
                            "active": {
                                "roles": {
                                    "antibody": {
                                        "identity_and_reciprocal_coverage_edge_artifact": {
                                            "path": str(crossing_edge),
                                            "sha256": _sha256(crossing_edge),
                                            "edge_count": 1,
                                        }
                                    },
                                    "antigen": {
                                        "identity_and_reciprocal_coverage_edge_artifact": {
                                            "path": str(empty_edge),
                                            "sha256": _sha256(empty_edge),
                                            "edge_count": 0,
                                        }
                                    },
                                }
                            }
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            protocol = directory / "protocol.md"
            protocol.write_text("locked\n", encoding="utf-8")
            output = directory / "boundary.json"
            args = argparse.Namespace(
                authority_split=authority,
                expected_authority_sha256=_sha256(authority),
                sequence_manifest=manifest,
                authority_nw_audit=report,
                edge_dir=edge_dir,
                protocol=protocol,
                constraint_mode="concat_antibody_and_antigen",
                train_split="train",
                validation_split="validation",
                sealed_splits=["test"],
                id_column="INSTANCE",
                heavy_column="vh_seq",
                light_column="vl_seq",
                antigen_column="ag_seq",
                output=output,
            )
            payload = run(args)
            self.assertEqual(payload["splits"]["train"], ["train_1"])
            self.assertEqual(payload["splits"]["validation"], ["validation_clean"])
            self.assertEqual(
                payload["splits"]["excluded_validation"], ["validation_crossing"]
            )
            self.assertEqual(
                payload["evidence"]["sequence_access_summary"][
                    "sealed_identifier_rows_skipped_before_sequence_access"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
