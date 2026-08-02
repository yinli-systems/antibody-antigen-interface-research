from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interaction_first.v6_b_direct_highs_audit import canonical_json_bytes
from interaction_first.v6_b_direct_highs_audit import (
    validate_direct_highs_registration,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (
    ROOT / "protocols" / "v6_b_direct_highs_audit_registration_v1.json"
)
REGISTRATION_SHA256 = (
    "71dc1d05955218f4796f0455c3dc64cc736ae4ace7dfee637cbb51ec9ef96fe2"
)


def test_production_registration_is_canonical_and_valid() -> None:
    payload = REGISTRATION.read_bytes()
    registration = json.loads(payload.decode("ascii"))

    assert hashlib.sha256(payload).hexdigest() == REGISTRATION_SHA256
    assert payload == canonical_json_bytes(registration)
    checked = validate_direct_highs_registration(registration)
    assert checked == registration


def test_production_registration_remains_science_denied() -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="ascii"))
    checked = validate_direct_highs_registration(registration)

    assert set(checked["authorization"].values()) == {False}
    assert checked["expected"] == {
        "cp_consensus_status": "INFEASIBLE",
        "dimensions": {
            "columns": 163464,
            "nonzeros": 10365708,
            "rows": 2693786,
        },
        "ir_sha256": (
            "3fff0ee63e0346034ba0bc9ce306212523adc786f013aaf37a73bf57848ca05d"
        ),
        "source_linear_projection_sha256": (
            "854f144775f928993e61b3c075df377361ead1aa327473a0b21fd039e8e5f66a"
        ),
    }
    assert checked["source"]["implementation_commit"] == (
        "5f833c91966660641a85c3b3cf750f4c5496c75b"
    )
    assert checked["slurm"]["accelerator_visible_to_solver"] is False
    assert checked["slurm"]["solver_threads"] == 1
