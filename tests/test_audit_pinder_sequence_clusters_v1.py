#!/usr/bin/env python3
"""Focused contracts for the PINDER train-only sequence-cluster audit."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from remote_tools.abf_paircontact_v1.pinder_scale_v1 import (
    audit_pinder_sequence_clusters_v1 as audit,
)


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


class PinderSequenceClusterAuditTests(unittest.TestCase):
    def test_fasta_identifier_must_equal_sequence_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequences.fasta"
            sequence = "ACDEFGHIK"
            path.write_text(f">{digest(sequence)}\n{sequence}\n", encoding="utf-8")
            self.assertEqual(audit.read_fasta(path), {digest(sequence): sequence})
            path.write_text(f">{'0' * 64}\n{sequence}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not the sequence SHA256"):
                audit.read_fasta(path)

    def test_mmseqs_fraction_fields_are_validated_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = digest("AAAA"), digest("AAAT")
            path = root / "edges.tsv"
            path.write_text(
                f"{first}\t{second}\t0.50\t0.80\t0.90\t4\t1e-5\t20\n"
                f"{second}\t{first}\t0.55\t0.90\t0.80\t4\t1e-5\t22\n",
                encoding="utf-8",
            )
            edges, stats = audit.load_edges(path, {first, second}, 30.0, 80.0)
            self.assertEqual(edges, [(min(first, second), max(first, second), 55.0)])
            self.assertEqual(stats["directional_row_count"], 2)
            self.assertEqual(stats["unique_undirected_nonself_edge_count"], 1)
            path.write_text(
                f"{first}\t{second}\t0.50\t0.79\t0.90\t4\t1e-5\t20\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "below reciprocal coverage"):
                audit.load_edges(path, {first, second}, 30.0, 80.0)

    def test_threshold_components_are_nested_transitive_closures(self) -> None:
        nodes = {digest("AAAA"): "AAAA", digest("AAAT"): "AAAT", digest("AATT"): "AATT"}
        first, second, third = sorted(nodes)
        assignments, summary = audit.cluster_role(
            "antibody",
            nodes,
            [(first, second, 95.0), (second, third, 55.0)],
        )
        self.assertEqual(summary["30"]["component_count"], 1)
        self.assertEqual(summary["50"]["component_count"], 1)
        self.assertEqual(summary["70"]["component_count"], 2)
        self.assertEqual(summary["90"]["component_count"], 2)
        self.assertEqual(assignments[30][first], assignments[30][third])
        self.assertNotEqual(assignments[70][first], assignments[70][third])

    def test_streaming_edge_clustering_matches_nested_transitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = {
                digest("CCCC"): "CCCC",
                digest("CCCD"): "CCCD",
                digest("CCDD"): "CCDD",
            }
            first, second, third = sorted(nodes)
            path = root / "edges.tsv"
            path.write_text(
                f"{first}\t{second}\t0.95\t0.80\t0.80\t4\t1e-5\t20\n"
                f"{second}\t{third}\t0.55\t0.80\t0.80\t4\t1e-5\t20\n",
                encoding="utf-8",
            )
            assignments, summary, stats = audit.cluster_role_from_edge_file(
                "antigen", nodes, path, 30.0, 80.0
            )
            self.assertEqual(stats["directional_nonself_row_count"], 2)
            self.assertEqual(summary["50"]["component_count"], 1)
            self.assertEqual(summary["70"]["component_count"], 2)
            self.assertEqual(assignments[30][first], assignments[30][third])

    def test_quality_filter_matches_frozen_pinder_criteria(self) -> None:
        frame = pd.DataFrame(
            {
                "method": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
                "resolution": [3.5, 2.0],
                "length1": [40, 100],
                "length2": [40, 100],
                "num_atom_types": [3, 4],
                "interface_atom_gaps_4A": [0, 0],
                "max_var_1": [0.98, 0.1],
                "max_var_2": [0.98, 0.1],
                "number_of_components_1": [1, 1],
                "number_of_components_2": [1, 1],
            }
        )
        self.assertEqual(audit.official_like_xray_quality(frame).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
