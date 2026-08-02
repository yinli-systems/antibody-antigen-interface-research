"""Test the numbered correction to Python cyclic-GC audit semantics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_compile_numerics_diagnostic_v2.py"
PARENT_AUDITOR = ROOT / "scripts/audit_antibody_compile_numerics_diagnostic.py"
PARENT_PROTOCOL = ROOT / "protocols/antibody_compile_numerics_diagnostic_v1.json"
AMENDMENT = ROOT / "protocols/antibody_compile_numerics_audit_amendment_v2.json"


def _load():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(
        "compile_numerics_auditor_v2",
        AUDITOR,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_amendment_binds_both_auditors_and_parent_protocol() -> None:
    module = _load()
    value, amendment_sha256 = module._load_amendment(
        AMENDMENT,
        parent_protocol_path=PARENT_PROTOCOL,
    )
    assert amendment_sha256 == _sha(AMENDMENT)
    assert value["amended_auditor_sha256"] == _sha(AUDITOR)
    assert value["parent_auditor_sha256"] == _sha(PARENT_AUDITOR)
    assert value["parent_protocol_file_sha256"] == _sha(PARENT_PROTOCOL)
    assert set(value["authorization"].values()) == {False}


def test_corrected_auditor_allows_explicit_collection_while_disabled(
    tmp_path: Path,
) -> None:
    module = _load()
    mode_root = tmp_path / "compile_gc_off"
    run = mode_root / "run"
    run.mkdir(parents=True)
    for name in ("receipt.json", "trainer_state.json", "parity_extension_receipt.json"):
        (run / name).write_bytes(b"{}\n")
    flags = {
        "accuracy_first_bf16_reduction": False,
        "compile_blocks": True,
        "disable_python_gc": True,
    }
    protocol_sha256 = "1" * 64
    protocol = {
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        },
        "inputs": {"diagnostic_wrapper_sha256": "2" * 64},
        "modes": {"compile_gc_off": flags},
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    before = [
        {"collected": 1, "collections": 2, "uncollectable": 0},
        {"collected": 3, "collections": 4, "uncollectable": 0},
        {"collected": 5, "collections": 6, "uncollectable": 0},
    ]
    delta = [
        {"collected": 0, "collections": 0, "uncollectable": 0},
        {"collected": 9, "collections": 3, "uncollectable": 0},
        {"collected": 0, "collections": 0, "uncollectable": 0},
    ]
    after = [
        {key: left[key] + difference[key] for key in left}
        for left, difference in zip(before, delta)
    ]
    sidecar = {
        "authorization": protocol["authorization"],
        "base_receipt_file_sha256": _sha(run / "receipt.json"),
        "base_trainer_state_file_sha256": _sha(run / "trainer_state.json"),
        "bf16_reduced_precision_reduction": {
            "after_restore": True,
            "before": True,
            "during_run": True,
        },
        "format_version": "antibody_compile_numerics_diagnostic_receipt_v1",
        "mode": "compile_gc_off",
        "mode_flags": flags,
        "parity_extension_receipt_file_sha256": _sha(
            run / "parity_extension_receipt.json"
        ),
        "protocol_sha256": protocol_sha256,
        "python_gc": {
            "enabled_after_restore": True,
            "enabled_before": True,
            "enabled_during_run": False,
            "precollection_reclaimed": 0,
            "stats_after": after,
            "stats_before": before,
            "stats_delta": delta,
        },
        "test_access": protocol["test_access"],
        "wrapper_sha256": "2" * 64,
    }
    sidecar["receipt_sha256"] = hashlib.sha256(_canonical(sidecar)).hexdigest()
    (run / "compile_numerics_diagnostic_receipt.json").write_bytes(
        _canonical(sidecar)
    )
    with pytest.raises(module.v1.CompileNumericsAuditError, match="collected while disabled"):
        module.v1._diagnostic_sidecar(
            mode_root,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            mode="compile_gc_off",
        )
    observed = module._corrected_sidecar(
        mode_root,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        mode="compile_gc_off",
    )
    assert observed["automatic_gc_disabled_during_run"] is True
    assert observed["collection_counters_include_explicit_calls"] is True
    assert observed["observed_collection_counter_delta"] == [0, 3, 0]
