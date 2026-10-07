from __future__ import annotations

import unittest

from remote_tools.abf_paircontact_v1.pinder_scale_v1.materialize_pinder_antibody_s2_assets_v1 import (
    noncanonical_symbols,
    sequence_sha256,
    sequence_views,
)


class PinderAntibodyS2AssetTests(unittest.TestCase):
    def test_fab_view_is_heavy_then_light(self) -> None:
        vh, vl, antigen = sequence_views(
            {
                "topology": "fab",
                "heavy_sequence": "ACD",
                "light_sequence": "EFG",
                "fused_sequence": "",
                "antigen_sequence": "HIK",
            }
        )
        self.assertEqual((vh, vl, antigen), ("ACD", "EFG", "HIK"))
        self.assertEqual(sequence_sha256(vh + vl), sequence_sha256("ACDEFG"))

    def test_heavy_only_and_scfv_have_empty_vl_view(self) -> None:
        heavy = sequence_views(
            {
                "topology": "heavy_only_or_vhh",
                "heavy_sequence": "ACD",
                "light_sequence": "",
                "fused_sequence": "",
                "antigen_sequence": "EFG",
            }
        )
        scfv = sequence_views(
            {
                "topology": "scfv",
                "heavy_sequence": "",
                "light_sequence": "",
                "fused_sequence": "HIK",
                "antigen_sequence": "LMN",
            }
        )
        self.assertEqual(heavy, ("ACD", "", "EFG"))
        self.assertEqual(scfv, ("HIK", "", "LMN"))

    def test_noncanonical_symbols_are_explicit(self) -> None:
        self.assertEqual(noncanonical_symbols("ACX", "", "BZ"), ("B", "X", "Z"))
        self.assertEqual(noncanonical_symbols("ACD", "EFG"), ())


if __name__ == "__main__":
    unittest.main()
