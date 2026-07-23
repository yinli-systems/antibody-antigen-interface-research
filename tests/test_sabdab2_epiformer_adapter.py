#!/usr/bin/env python3
"""Focused unit tests for the SAbDab2 EpiFormer projection adapter."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sabdab2_epiformer_adapter as adapter


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            adapter.ResidueRecord("A", 1, " ", "A"),
            adapter.ResidueRecord("A", 2, " ", "C"),
            adapter.ResidueRecord("A", 3, "A", "D"),
            adapter.ResidueRecord("A", 4, " ", "E"),
        ]

    def test_parse_chain_ids_accepts_supported_separators(self):
        self.assertEqual(adapter.parse_chain_ids("H,L"), ["H", "L"])
        self.assertEqual(adapter.parse_chain_ids("H + L"), ["H", "L"])
        self.assertEqual(adapter.parse_chain_ids("A|B;C"), ["A", "B", "C"])

    def test_antibody_chain_types_come_from_canonical_metadata(self):
        mapping = adapter.antibody_chain_type_map(
            {"Hchain": "B", "Lchain": "A"},
            ["B", "A"],
        )
        self.assertEqual(mapping, {"B": "H", "A": "L"})

    def test_single_domain_antibody_ignores_plus_light_chain_marker(self):
        mapping = adapter.antibody_chain_type_map(
            {"Hchain": "A", "Lchain": "+"},
            ["A"],
        )
        self.assertEqual(mapping, {"A": "H"})

    def test_delivered_imgt_mapping_selects_cdrs_with_nonstandard_numbering(self):
        heavy = [
            adapter.ResidueRecord("D", 1501 + index, " ", "A")
            for index in range(120)
        ]
        light = [
            adapter.ResidueRecord("C", 1001 + index, " ", "A")
            for index in range(110)
        ]
        mapping = {
            "vh_imgt_positions": [str(index + 1) for index in range(120)],
            "vl_imgt_positions": [str(index + 1) for index in range(110)],
            "vh_to_hseq": list(range(120)),
            "vl_to_lseq": list(range(110)),
        }
        selected = adapter.benchmark_cdr_residue_keys(
            heavy + light,
            mapping,
            "A" * 230,
            {"D": "H", "C": "L"},
        )
        self.assertIn(("D", 1501 + 26, " "), selected)
        self.assertIn(("C", 1001 + 104, " "), selected)
        self.assertNotIn(("D", 1501 + 20, " "), selected)

    def test_antibody_mapping_falls_back_after_a_structure_deletion(self):
        source_sequence = "ACDEFG"
        canonical_sequence = "ACMDEFG"
        source = [
            adapter.ResidueRecord("D", 1501 + index, " ", amino_acid)
            for index, amino_acid in enumerate(source_sequence)
        ]
        mapping, method, coverage, identity = adapter.canonical_to_source_residue_map(
            source,
            canonical_sequence,
            list(range(len(canonical_sequence))),
        )
        self.assertEqual(method, "local_sequence_alignment")
        self.assertNotIn(2, mapping)
        self.assertEqual(mapping[3], 2)
        self.assertGreater(coverage, 0.8)
        self.assertEqual(identity, 1.0)

    def test_delivered_mapping_projects_and_fills_with_instance_minimum(self):
        result = {
            "residue_details": [
                {
                    "chain_id": "A",
                    "residue_id": 2,
                    "insertion_code": " ",
                    "probability": 0.8,
                },
                {
                    "chain_id": "A",
                    "residue_id": 3,
                    "insertion_code": "A",
                    "probability": 0.2,
                },
            ]
        }
        scores, observed, audit = adapter.project_surface_scores(
            instance_id="example",
            canonical_sequence="ACDEF",
            source_residues=self.source,
            mapping_payload={"ag_to_agseq": [0, 1, 2, 3]},
            inference_result=result,
            elapsed_seconds=1.5,
        )
        np.testing.assert_allclose(scores, [0.2, 0.8, 0.2, 0.2, 0.2])
        np.testing.assert_array_equal(observed, [False, True, True, False, False])
        self.assertEqual(audit.mapping_method, "delivered_ag_to_agseq")
        self.assertEqual(audit.filled_with_minimum_count, 3)
        self.assertEqual(audit.mapped_surface_fraction, 1.0)

    def test_alignment_fallback_handles_a_canonical_insertion(self):
        mapping, method, coverage, identity = adapter.canonical_index_map(
            self.source,
            "ACXDE",
            {"ag_to_agseq": [100]},
        )
        self.assertEqual(method, "global_sequence_alignment")
        self.assertEqual(mapping[("A", 3, "A")], 3)
        self.assertAlmostEqual(coverage, 0.8)
        self.assertAlmostEqual(identity, 1.0)

    def test_score_archive_validation_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path = root / "scores.npz"
            audit_path = root / "audit.json"
            adapter.atomic_write_npz(
                score_path,
                canonical_scores=np.array([0.1, np.nan], dtype=np.float32),
                observed_mask=np.array([True, False]),
            )
            audit_path.write_text(json.dumps({"canonical_length": 2}))
            self.assertFalse(
                adapter.validate_score_archive(score_path, audit_path, 2)
            )

    def test_row_group_index_uses_exact_instance_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.parquet"
            schema = pa.schema(
                [
                    ("instance_id", pa.string()),
                    ("ab_res_idx", pa.int64()),
                    ("ag_res_idx", pa.int64()),
                    ("label", pa.bool_()),
                ]
            )
            with pq.ParquetWriter(path, schema) as writer:
                for instance_id in ("first", "second"):
                    writer.write_table(
                        pa.table(
                            {
                                "instance_id": [instance_id, instance_id],
                                "ab_res_idx": [0, 1],
                                "ag_res_idx": [0, 0],
                                "label": [False, True],
                            },
                            schema=schema,
                        )
                    )
            index = adapter.row_group_instance_index(
                pq.ParquetFile(path),
                {"first", "second"},
            )
            self.assertEqual(index, {"first": 0, "second": 1})


if __name__ == "__main__":
    unittest.main()
