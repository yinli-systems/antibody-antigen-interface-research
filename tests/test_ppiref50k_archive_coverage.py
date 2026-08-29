from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_ppiref50k_archive_coverage_v2.py"


def module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("ppiref_coverage", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_split_members_are_partitioned_into_exact_casefold_reverse_and_absent() -> None:
    split = [("1abc", "A", "B"), ("2xyz", "C", "D"), ("3def", "E", "F"), ("4ghi", "G", "H")]
    members = {
        ("1abc", "A", "B"): "ab/1abc_A_B.pdb",
        ("2xyz", "D", "C"): "xy/2xyz_D_C.pdb",
        ("4ghi", "g", "h"): "gh/4ghi_g_h.pdb",
    }
    counts, aliases, reverse, absent = module().classify(split, members)
    assert counts == {
        "absent_interfaces": 1,
        "archive_pdb_members": 3,
        "casefold_same_order_interfaces": 1,
        "exact_order_interfaces": 1,
        "reverse_order_only_interfaces": 1,
        "split_interfaces": 4,
    }
    assert reverse[0]["requested_interface_id"] == "2xyz_C_D"
    assert aliases[0]["archive_interface_id"] == "4ghi_g_h"
    assert absent == ["3def_E_F"]
