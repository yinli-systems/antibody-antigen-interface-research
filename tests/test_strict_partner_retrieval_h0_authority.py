from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/h0.py"
spec = importlib.util.spec_from_file_location("h0_authority", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _base() -> dict:
    return {
        "format_version": "strict_partner_retrieval_h0_cache_v1",
        "role": "h0_evaluation_only",
        "h0_execution_authorized": True,
        "test_access": False,
        "ood_access": False,
        "training_antibodies": {"a0": {"vh": "ACD", "vl": ""}},
        "training_antigens": {"g0": "ACD"},
        "validation_antibodies": {"a1": {"vh": "ACE", "vl": ""}},
        "validation_antigens": {"g1": "ACE"},
        "training_pairs": [{"antibody_id": "a0", "antigen_id": "g0"}],
        "validation_pairs": [{"antibody_id": "a1", "antigen_id": "g1"}],
    }


def test_h0_authority_required() -> None:
    value = _base()
    value["h0_execution_authorized"] = False
    try:
        module.validate_cache(value)
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized H0 cache was accepted")


def test_authorized_h0_cache_passes() -> None:
    module.validate_cache(_base())
