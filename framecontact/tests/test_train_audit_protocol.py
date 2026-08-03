from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "framecontact_train_audit_v1.json"
RECEIPT = ROOT / "protocols" / "FRAMECONTACT_TRAIN_AUDIT_V1_RECEIPT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def test_train_audit_protocol_is_hash_bound_and_development_only() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["protocol_file_sha256"] == _sha(PROTOCOL)
    assert receipt["protocol_canonical_sha256"] == hashlib.sha256(
        _canonical(protocol)
    ).hexdigest()
    assert protocol["eligible_checkpoint"]["selection_source"] == "validation only"
    assert protocol["calibration"]["fit_on_train"] is False
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["data"]["external_role_forbidden"] is True
    assert receipt["test_and_external_roles_forbidden"] is True


def test_train_audit_receipt_binds_evaluator_and_guard_tests() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    implementation = protocol["implementation"]
    for name in ("evaluator", "test"):
        assert _sha(ROOT / implementation[f"{name}_path"]) == implementation[
            f"{name}_sha256"
        ]
        assert receipt["implementation_sha256"][name] == implementation[
            f"{name}_sha256"
        ]
