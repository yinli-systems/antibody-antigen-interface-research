#!/usr/bin/env python3
"""Independently audit exact 650M FSDP2 checkpoint recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PARAMETER_COUNT = 634_513_920
EXPECTED_FORMAT_VERSION = "antibody_pretraining_pilot_v4"
EXPECTED_RECOVERY_FORMAT_VERSION = "antibody_training_recovery_v3"
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
EXPECTED_SHARD_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
TRACE_KEYS = (
    "consumed_residue_tokens",
    "gradient_norm",
    "learning_rate",
    "loss",
    "residue_tokens",
    "selected_targets",
    "update",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_equal(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    return len(left) == len(right) and all(
        all(left_item[key] == right_item[key] for key in TRACE_KEYS)
        for left_item, right_item in zip(left, right)
    )


def _validate_training_receipt(
    receipt: dict[str, object],
    *,
    resume_used: bool,
    start_update: int,
    updates: list[int],
    checkpoint_format: str | None,
) -> None:
    if (
        receipt["backend"] != "fsdp2"
        or receipt["preset"] != "abf_650m"
        or receipt["world_size"] != 8
        or receipt["format_version"] != EXPECTED_FORMAT_VERSION
        or receipt["parameter_moments"]["count"] != PARAMETER_COUNT
        or receipt["resume_used"] is not resume_used
        or receipt["start_update"] != start_update
        or receipt["checkpoint_format"] != checkpoint_format
        or receipt["model_family_sha256"] != EXPECTED_MODEL_FAMILY_SHA256
        or receipt["training_protocol_sha256"]
        != EXPECTED_TRAINING_PROTOCOL_SHA256
        or receipt["shard_manifest_sha256"]
        != EXPECTED_SHARD_MANIFEST_SHA256
        or receipt["test_access"] != EXPECTED_TEST_ACCESS
        or [item["update"] for item in receipt["trace"]] != updates
    ):
        raise RuntimeError("A 650M recovery training receipt failed validation")


def _validate_state(state: dict[str, object], next_update: int) -> None:
    if (
        state["backend"] != "fsdp2"
        or state["preset"] != "abf_650m"
        or state["world_size"] != 8
        or state["next_update"] != next_update
        or state["model_family_sha256"] != EXPECTED_MODEL_FAMILY_SHA256
        or state["training_protocol_sha256"]
        != EXPECTED_TRAINING_PROTOCOL_SHA256
        or state["shard_manifest_sha256"] != EXPECTED_SHARD_MANIFEST_SHA256
    ):
        raise RuntimeError("A 650M recovery trainer state failed validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "continuous_receipt": args.continuous / "receipt.json",
        "continuous_state": args.continuous / "trainer_state.json",
        "stage1_receipt": args.stage1 / "receipt.json",
        "stage1_state": args.stage1 / "trainer_state.json",
        "resumed_receipt": args.resumed / "receipt.json",
        "resumed_state": args.resumed / "trainer_state.json",
        "recovery_receipt": args.recovery_receipt,
    }
    continuous = _load(paths["continuous_receipt"])
    stage1 = _load(paths["stage1_receipt"])
    resumed = _load(paths["resumed_receipt"])
    continuous_state = _load(paths["continuous_state"])
    stage1_state = _load(paths["stage1_state"])
    resumed_state = _load(paths["resumed_state"])
    recovery = _load(paths["recovery_receipt"])

    _validate_training_receipt(
        continuous,
        resume_used=False,
        start_update=0,
        updates=[1, 2, 3, 4],
        checkpoint_format=None,
    )
    _validate_training_receipt(
        stage1,
        resume_used=False,
        start_update=0,
        updates=[1, 2],
        checkpoint_format="torch_distributed_checkpoint",
    )
    _validate_training_receipt(
        resumed,
        resume_used=True,
        start_update=2,
        updates=[3, 4],
        checkpoint_format=None,
    )
    _validate_state(continuous_state, 4)
    _validate_state(stage1_state, 2)
    _validate_state(resumed_state, 4)

    exact = {
        "model": continuous["final_model_sha256"]
        == resumed["final_model_sha256"]
        == recovery["final_model_sha256"],
        "optimizer": continuous["final_optimizer_sha256"]
        == resumed["final_optimizer_sha256"]
        == recovery["final_optimizer_sha256"],
        "prefix_trace": _trace_equal(continuous["trace"][:2], stage1["trace"]),
        "resumed_trace": _trace_equal(continuous["trace"][2:], resumed["trace"]),
        "trainer_state": continuous_state == resumed_state,
    }
    if (
        recovery["backend"] != "fsdp2"
        or recovery["world_size"] != 8
        or recovery["checkpoint_format"] != "torch_distributed_checkpoint"
        or recovery["format_version"] != EXPECTED_RECOVERY_FORMAT_VERSION
        or recovery["test_access"] != EXPECTED_TEST_ACCESS
        or recovery["equal"]
        != {"model": True, "optimizer": True, "trainer_state": True}
        or not all(exact.values())
    ):
        raise RuntimeError(f"650M exact recovery audit failed: {exact}")

    payload = {
        "authorization": {
            "exact_recovery_backend": "fsdp2",
            "pilot_1_3b": False,
            "production_pretraining": False,
            "scale_32_gpu": False,
            "scale_64_gpu": False,
            "scientific_promotion_required": True,
        },
        "claim_boundary": (
            "650M same-world-size eight-GPU engineering recovery on repeated pilot "
            "data; no scientific, production-corpus, or strong-scaling claim"
        ),
        "exact_equal": exact,
        "format_version": "antibody_650m_recovery_audit_v1",
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "job_id": args.job_id,
        "parameter_count": PARAMETER_COUNT,
        "preset": "abf_650m",
        "test_access": EXPECTED_TEST_ACCESS,
        "world_size": 8,
    }
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
