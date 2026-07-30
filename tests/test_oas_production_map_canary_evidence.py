"""Test the authenticated real-OAS production mapper canary evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pretraining/oas_production_map_canary_v1.json"


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def test_real_canary_is_self_anchored_and_does_not_authorize_training() -> None:
    evidence = json.loads(RESULT.read_text(encoding="ascii"))
    receipt = evidence["mapper_receipt"]
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")

    assert hashlib.sha256(_canonical(unsigned)).hexdigest() == claimed
    assert receipt["counts"] == {
        "accepted_records": 3,
        "plan_fraction_selected": 3,
        "source_rows": 3,
    }
    assert evidence["independent_audit"]["audit_passed"] is True
    assert evidence["independent_audit"]["imported_mapper"] is False
    assert evidence["scheduler_probe"]["cpu_only_submission_accepted"] is False
    assert evidence["claim_boundary"]["large_scale_pretraining_authorized"] is False
    assert evidence["test_access"] == {
        "downstream_metrics_used": False,
        "sabdab2_heldout_accessed": False,
    }
