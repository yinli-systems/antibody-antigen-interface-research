from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).parents[1]
    / "remote_tools/abf_paircontact_v1/pinder_scale_v1"
    / "diagnose_pinder_antibody_cdr_numbering_v1.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("cdr_numbering_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _numbering(positions: range):
    return [((position, " "), "A") for position in positions]


def test_summarize_numbering_reports_missing_cdr2() -> None:
    result = {
        "chain_type": "L",
        "error": None,
        "numbering": _numbering(range(1, 56)) + _numbering(range(66, 118)),
    }
    summary = MODULE.summarize_numbering(result, {"K", "L"})
    assert summary["status"] == "FAIL"
    assert summary["missing_cdr_ranges"] == ["56-65"]
    assert "missing IMGT CDR range 56-65" in summary["failures"]


def test_summarize_numbering_accepts_all_three_cdr_ranges() -> None:
    result = {"chain_type": "H", "error": None, "numbering": _numbering(range(1, 119))}
    summary = MODULE.summarize_numbering(result, {"H"})
    assert summary["status"] == "PASS"
    assert summary["missing_cdr_ranges"] == []
    assert summary["cdr_length"] == 35


def test_resolve_scfv_source_key_accepts_original_and_domain_keys() -> None:
    keys = {"scfv-000019", "scfv-000035"}
    assert MODULE.resolve_scfv_source_key("scfv-000019", keys) == "scfv-000019"
    assert MODULE.resolve_scfv_source_key("scfv-000019-1", keys) == "scfv-000019"
    assert MODULE.resolve_scfv_source_key("unknown-1", keys) is None
