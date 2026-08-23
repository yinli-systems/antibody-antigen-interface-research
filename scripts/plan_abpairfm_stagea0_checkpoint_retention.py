#!/usr/bin/env python3
"""Plan verified Stage-A0 DCP retention without deleting any artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = (250_000_000, 500_000_000, 1_000_000_000)
MAXIMUM_TARGET_OVERSHOOT = 20_000_000


def directory_size(path: Path) -> int:
    total = 0
    for member in path.rglob("*"):
        if member.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {member}")
        if member.is_file():
            total += member.stat().st_size
    return total


def verified_checkpoints(campaign_root: Path) -> list[dict[str, Any]]:
    checkpoint_root = campaign_root / "checkpoints"
    result_root = campaign_root / "results"
    checkpoints = []
    for checkpoint_path in sorted(checkpoint_root.glob("segment-*")):
        if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
            raise ValueError(f"Invalid checkpoint directory: {checkpoint_path}")
        generation = checkpoint_path.name
        try:
            segment_index = int(generation.removeprefix("segment-"))
        except ValueError as error:
            raise ValueError(f"Invalid checkpoint generation: {generation}") from error
        result_path = result_root / f"{generation}.json"
        receipt_path = checkpoint_path / "receipt.json"
        if not result_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Unclosed checkpoint: {generation}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checkpoint = result.get("checkpoint", {})
        training_state = result.get("training_state", {})
        if (
            result.get("status") != "PASS_CHECKPOINT_VERIFIED"
            or result.get("segment_index") != segment_index
            or result.get("test_access") is not False
            or checkpoint.get("generation") != generation
            or receipt.get("content_sha256") != checkpoint.get("content_sha256")
            or receipt.get("training_state") != training_state
            or training_state.get("test_access") is not False
        ):
            raise ValueError(f"Checkpoint closure differs: {generation}")
        checkpoints.append(
            {
                "consumed_residue_tokens": training_state["consumed_residue_tokens"],
                "content_sha256": checkpoint["content_sha256"],
                "generation": generation,
                "segment_index": segment_index,
                "size_bytes": directory_size(checkpoint_path),
            }
        )
    if not checkpoints:
        raise ValueError("No verified checkpoints found")
    return checkpoints


def retention_plan(
    checkpoints: list[dict[str, Any]],
    *,
    keep_latest: int,
    periodic_interval: int,
) -> dict[str, Any]:
    if keep_latest <= 0 or periodic_interval <= 0:
        raise ValueError("Retention counts must be positive")
    reasons: dict[str, set[str]] = {
        checkpoint["generation"]: set() for checkpoint in checkpoints
    }
    for checkpoint in checkpoints[-keep_latest:]:
        reasons[checkpoint["generation"]].add("latest_resume")
    for checkpoint in checkpoints:
        if checkpoint["segment_index"] % periodic_interval == 0:
            reasons[checkpoint["generation"]].add("periodic_anchor")
    reached_targets = {}
    for target in TARGETS:
        candidates = [
            checkpoint
            for checkpoint in checkpoints
            if target
            <= checkpoint["consumed_residue_tokens"]
            <= target + MAXIMUM_TARGET_OVERSHOOT
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda checkpoint: (
                    checkpoint["consumed_residue_tokens"],
                    checkpoint["segment_index"],
                ),
            )
            reasons[selected["generation"]].add(f"target_{target}")
            reached_targets[str(target)] = selected["generation"]
    retained = []
    eligible = []
    for checkpoint in checkpoints:
        row = checkpoint | {"reasons": sorted(reasons[checkpoint["generation"]])}
        (retained if row["reasons"] else eligible).append(row)
    return {
        "authorization": {"delete": False, "test_access": False},
        "eligible_for_removal": eligible,
        "format_version": "abpairfm_stagea0_checkpoint_retention_plan_v1",
        "policy": {
            "keep_latest": keep_latest,
            "maximum_target_overshoot": MAXIMUM_TARGET_OVERSHOOT,
            "periodic_interval": periodic_interval,
            "targets": list(TARGETS),
        },
        "reached_targets": reached_targets,
        "retained": retained,
        "status": "PASS_READ_ONLY_RETENTION_PLAN",
        "summary": {
            "checkpoint_count": len(checkpoints),
            "eligible_bytes": sum(row["size_bytes"] for row in eligible),
            "eligible_count": len(eligible),
            "retained_bytes": sum(row["size_bytes"] for row in retained),
            "retained_count": len(retained),
            "total_bytes": sum(row["size_bytes"] for row in checkpoints),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--periodic-interval", type=int, default=10)
    arguments = parser.parse_args()
    plan = retention_plan(
        verified_checkpoints(arguments.campaign_root),
        keep_latest=arguments.keep_latest,
        periodic_interval=arguments.periodic_interval,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
