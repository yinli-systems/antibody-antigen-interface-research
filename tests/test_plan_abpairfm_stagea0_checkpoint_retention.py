from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_abpairfm_stagea0_checkpoint_retention import (
    retention_plan,
    verified_checkpoints,
)


def make_checkpoint(root: Path, index: int, tokens: int) -> None:
    generation = f"segment-{index:06d}"
    checkpoint = root / "checkpoints" / generation
    checkpoint.mkdir(parents=True)
    (checkpoint / "rank.distcp").write_bytes(b"x" * (index + 1))
    training_state = {
        "consumed_residue_tokens": tokens,
        "test_access": False,
    }
    receipt = {
        "content_sha256": f"sha-{index}",
        "training_state": training_state,
    }
    (checkpoint / "receipt.json").write_text(json.dumps(receipt))
    result_root = root / "results"
    result_root.mkdir(exist_ok=True)
    result = {
        "checkpoint": {
            "content_sha256": f"sha-{index}",
            "generation": generation,
        },
        "segment_index": index,
        "status": "PASS_CHECKPOINT_VERIFIED",
        "test_access": False,
        "training_state": training_state,
    }
    (result_root / f"{generation}.json").write_text(json.dumps(result))


def test_plan_keeps_targets_periodic_anchors_and_latest(tmp_path: Path) -> None:
    for index, tokens in enumerate((100_000_000, 250_000_000, 400_000_000, 510_000_000)):
        make_checkpoint(tmp_path, index, tokens)
    plan = retention_plan(
        verified_checkpoints(tmp_path), keep_latest=1, periodic_interval=3
    )
    retained = {row["generation"]: row["reasons"] for row in plan["retained"]}
    assert retained == {
        "segment-000000": ["periodic_anchor"],
        "segment-000001": ["target_250000000"],
        "segment-000003": ["latest_resume", "periodic_anchor", "target_500000000"],
    }
    assert [row["generation"] for row in plan["eligible_for_removal"]] == [
        "segment-000002"
    ]
    assert plan["authorization"] == {"delete": False, "test_access": False}


def test_plan_fails_closed_on_receipt_mismatch(tmp_path: Path) -> None:
    make_checkpoint(tmp_path, 0, 100_000_000)
    receipt = tmp_path / "checkpoints/segment-000000/receipt.json"
    value = json.loads(receipt.read_text())
    value["content_sha256"] = "wrong"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Checkpoint closure differs"):
        verified_checkpoints(tmp_path)
