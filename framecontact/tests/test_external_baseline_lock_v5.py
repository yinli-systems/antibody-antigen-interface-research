"""Contract tests for the additive 2026 frontier-baseline ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_baseline_lock_is_additive_and_fail_closed() -> None:
    path = ROOT / "baselines" / "BASELINE_LOCK_V5.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = ROOT / payload["parent"]["path"]
    assert payload["format_version"] == "framecontact_baseline_lock_v5"
    assert _sha256(parent) == payload["parent"]["sha256"]
    assert payload["comparison_contract"][
        "endpoint_and_antigen_structure_disjoint_roles_required"
    ] is True
    assert payload["comparison_contract"]["five_paired_seeds"] == [
        17,
        42,
        73,
        101,
        137,
    ]
    assert payload["comparison_contract"][
        "published_numbers_are_literature_context_only"
    ] is True
    assert set(payload["frontier_additions"]) == {
        "deepssinter",
        "handshake",
        "prostt5_cross_contact_control",
        "protact",
    }
    for authority in payload["frontier_additions"].values():
        assert authority["identical_split_runs"] == 0
        assert authority["result_status"] == "not_run"
        assert authority["comparability_boundary"]
        assert authority["license_boundary"]


def test_v5_frontier_revisions_and_checkpoint_are_pinned() -> None:
    payload = json.loads(
        (ROOT / "baselines" / "BASELINE_LOCK_V5.json").read_text(
            encoding="utf-8"
        )
    )["frontier_additions"]
    assert payload["deepssinter"]["source_revision"] == (
        "d24e9e02b35c857a553e5a9d7d7def51c633d166"
    )
    assert payload["deepssinter"]["checkpoint"]["sha256"] == (
        "7ea97d065210c0fa4f6ceaed2c6f64caaba027bf661684affb3261b96103765d"
    )
    assert payload["handshake"]["source_revision"] == (
        "cfb10ffcf0d5745473fe1deb3a6e849b9f5986f4"
    )
    assert payload["prostt5_cross_contact_control"]["model_revision"] == (
        "d7d097d5bf9a993ab8f68488b4681d6ca70db9e5"
    )
    assert payload["protact"]["source_revision"] == (
        "ed0d75e2adf159915e5a1d2fb4ead73f8d278702"
    )
