from __future__ import annotations

import unittest

from remote_tools.abf_paircontact_v1.pinder_scale_v1.build_pinder_antibody_s2_folds_v1 import (
    SourceRow,
    exact_pair_id,
    interaction_components,
)


class PinderAntibodyS2FoldTests(unittest.TestCase):
    def test_exact_pair_and_pdb_links_form_indivisible_components(self) -> None:
        rows = [
            SourceRow("s1", "p1", "fab", "a1", "g1"),
            SourceRow("s2", "p1", "fab", "a2", "g2"),
            SourceRow("s3", "p2", "fab", "a2", "g2"),
            SourceRow("s4", "p3", "fab", "a3", "g3"),
        ]
        antibody_roots = {"a1": "ac1", "a2": "ac2", "a3": "ac3"}
        antigen_roots = {"g1": "gc1", "g2": "gc2", "g3": "gc3"}
        components, pairs, component_by_system = interaction_components(
            rows, antibody_roots, antigen_roots
        )
        self.assertEqual(len(components), 2)
        self.assertEqual(component_by_system["s1"], component_by_system["s3"])
        self.assertNotEqual(component_by_system["s1"], component_by_system["s4"])
        self.assertEqual(len(pairs), 3)
        self.assertIn(exact_pair_id("a2", "g2"), pairs)


if __name__ == "__main__":
    unittest.main()
