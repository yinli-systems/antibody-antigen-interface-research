from __future__ import annotations

import unittest

import pandas as pd

from remote_tools.abf_paircontact_v1.pinder_scale_v1.build_pinder_antibody_adaptation_quality_v1 import (
    antibody_antigen_edges,
    official_like_xray_quality,
    weighted_rows,
)


class AntibodyAdaptationQualityTests(unittest.TestCase):
    def test_quality_filter_matches_frozen_edge_policy(self) -> None:
        base = {
            "method": "X-RAY DIFFRACTION",
            "resolution": 3.5,
            "length1": 40,
            "length2": 40,
            "num_atom_types": 3,
            "interface_atom_gaps_4A": 0,
            "max_var_1": 0.98,
            "max_var_2": 0.98,
            "number_of_components_1": 1,
            "number_of_components_2": 1,
        }
        bad = dict(base, resolution=3.5001)
        observed = official_like_xray_quality(pd.DataFrame([base, bad])).tolist()
        self.assertEqual(observed, [True, False])

    def test_hl_support_is_not_treated_as_antibody_antigen_label_edge(self) -> None:
        row = {
            "system_id": "s",
            "support_hl_edge_ids": "hl",
            "support_heavy_antigen_edge_ids": "ha",
            "support_light_antigen_edge_ids": "la",
            "support_fused_antigen_edge_ids": "",
        }
        self.assertEqual(antibody_antigen_edges(row), ("ha", "la"))

    def test_three_level_multiplicity_weight_is_normalized(self) -> None:
        rows = [
            {
                "system_id": "s1",
                "pdb_id": "p1",
                "antibody_axis_sequence_sha256": "a1",
                "antigen_sequence_sha256": "g1",
                "support_interface_cluster_ids": "c1",
            },
            {
                "system_id": "s2",
                "pdb_id": "p1",
                "antibody_axis_sequence_sha256": "a1",
                "antigen_sequence_sha256": "g1",
                "support_interface_cluster_ids": "c1",
            },
            {
                "system_id": "s3",
                "pdb_id": "p2",
                "antibody_axis_sequence_sha256": "a2",
                "antigen_sequence_sha256": "g2",
                "support_interface_cluster_ids": "c2",
            },
        ]
        enriched, summary = weighted_rows(rows)
        weights = [float(row["system_loss_weight"]) for row in enriched]
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=11)
        self.assertGreater(weights[2], weights[0])
        self.assertEqual(enriched[0]["exact_sequence_pair_multiplicity"], 2)
        self.assertEqual(enriched[0]["pdb_multiplicity"], 2)
        self.assertEqual(enriched[0]["interface_cluster_signature_multiplicity"], 2)
        self.assertAlmostEqual(summary["final_weight_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
