from __future__ import annotations

import unittest

from remote_tools.abf_paircontact_v1.pinder_scale_v1.finalize_pinder_antibody_s2_mmseqs_v1 import (
    fraction,
)


class PinderAntibodyS2MmseqsFinalizerTests(unittest.TestCase):
    def test_fraction_accepts_percent_and_unit_interval(self) -> None:
        self.assertEqual(fraction("80"), 0.8)
        self.assertEqual(fraction("0.8"), 0.8)

    def test_fraction_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            fraction("101")


if __name__ == "__main__":
    unittest.main()
