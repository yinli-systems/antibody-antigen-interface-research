"""Contract tests for the static-194 four-GPU efficiency gate v3."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_efficiency_gate_v3.py"
RECOVERY_AUDITOR = (
    ROOT / "scripts/recover_antibody_four_gpu_efficiency_gate_v3_audit.py"
)


def _load(name: str, path: Path):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_v3_exports_every_master_protocol_contract() -> None:
    freezer = _load("efficiency_v3_contract_test", FREEZER)
    for name in (
        "ACCEPTANCE",
        "AUTHORIZATION",
        "BATCH",
        "HARDWARE",
        "MEASUREMENT",
        "TEST_ACCESS",
    ):
        assert getattr(freezer, name) == getattr(freezer.parent, name)
    assert freezer.VALIDATION_PANEL["indices_sha256"] == (
        "ba24268e3897320aafb49c8b557d559a73386242e47b666d5fc862fdf453cd38"
    )


def test_recovery_auditor_restores_legacy_missing_contract() -> None:
    recovery = _load("efficiency_v3_recovery_test", RECOVERY_AUDITOR)
    for name in recovery.CONTRACT_NAMES:
        if hasattr(recovery.freezer, name):
            delattr(recovery.freezer, name)
    recovery.restore_delegated_contract()
    for name in recovery.CONTRACT_NAMES:
        assert getattr(recovery.freezer, name) == getattr(
            recovery.freezer.parent, name
        )


def test_v3_recomputes_hashes_after_boundary_transform(monkeypatch) -> None:
    freezer = _load("efficiency_v3_hash_test", FREEZER)
    gates = {
        mode: {
            "execution": {
                "length_bucket_boundaries": freezer.BUCKETED_EAGER_BOUNDARIES
            }
        }
        for mode in ("compiled", "safe_eager")
    }
    monkeypatch.setattr(
        freezer.parent,
        "freeze_bundle",
        lambda **kwargs: ({"mode_gate_file_sha256": {}}, gates),
    )
    master, transformed = freezer.freeze_bundle()
    for mode, gate in transformed.items():
        expected = hashlib.sha256(
            freezer.parent._canonical_json_bytes(gate)
        ).hexdigest()
        assert master["mode_gate_file_sha256"][mode] == expected
    assert transformed["compiled"]["execution"]["length_bucket_boundaries"] == "194"


def test_recovery_amendment_proves_only_boundary_transform(tmp_path: Path) -> None:
    recovery = _load("efficiency_v3_amendment_test", RECOVERY_AUDITOR)
    recovery.restore_delegated_contract()
    gates = {
        "compiled": {"execution": {"length_bucket_boundaries": "194"}},
        "safe_eager": {
            "execution": {"length_bucket_boundaries": "96,128,160,194"}
        },
    }
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    legacy_hashes = {}
    for mode, gate in gates.items():
        (bundle / f"{mode}_gate.json").write_bytes(
            recovery.parent._canonical_json_bytes(gate)
        )
        reconstructed = json.loads(json.dumps(gate))
        reconstructed["execution"]["length_bucket_boundaries"] = (
            recovery.freezer.BUCKETED_EAGER_BOUNDARIES
        )
        legacy_hashes[mode] = hashlib.sha256(
            recovery.parent._canonical_json_bytes(reconstructed)
        ).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(
        recovery.parent._canonical_json_bytes(
            {"mode_gate_file_sha256": legacy_hashes}
        )
    )
    amended = tmp_path / "amended.json"
    receipt = recovery.amend_legacy_mode_hashes(
        protocol_path=protocol,
        bundle_root=bundle,
        output=amended,
    )
    amended_value = json.loads(amended.read_text(encoding="ascii"))
    assert amended_value["mode_gate_file_sha256"] == {
        mode: hashlib.sha256(
            recovery.parent._canonical_json_bytes(gate)
        ).hexdigest()
        for mode, gate in gates.items()
    }
    assert receipt["original_protocol_file_sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()
