"""Validate the preserved failed compile-only confirmation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results/pretraining/antibody_compile_only_confirmation_v1_1343024"
    / "antibody_compile_only_confirmation_v1_1343024.json"
)


def _canonical_bytes(value: object) -> bytes:
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


def test_failed_confirmation_is_canonical_and_self_hashed() -> None:
    raw = RESULT.read_bytes()
    value = json.loads(raw.decode("ascii"))
    assert raw == _canonical_bytes(value)
    recorded = value.pop("receipt_sha256")
    assert hashlib.sha256(_canonical_bytes(value)).hexdigest() == recorded
    assert hashlib.sha256(raw).hexdigest() == (
        "06cb74083bd79eec8a5f5abefc99ebc4c2d90634d9d4cba96d1d5dd973028c4d"
    )


def test_failed_confirmation_preserves_gate_boundaries() -> None:
    value = json.loads(RESULT.read_text(encoding="ascii"))
    assert value["gate_passed"] is False
    assert value["authorization"] == {
        "configuration_promotion": False,
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
        "scientific_claim": False,
    }
    assert value["gates"]["evaluation_noninferiority"] is False
    assert value["gates"]["step_stability"] is False
    assert value["pooled"]["compiled_relative_to_eager"] == 1.3430927409284954
    assert value["pooled"]["modes"]["fp32_compiled"]["approximate_6n_mfu"] == (
        0.4606550601042514
    )
    assert value["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }


def test_frozen_run_artifacts_match_the_audited_manifest() -> None:
    run_root = RESULT.parent / "slurm-1343024"
    lines = (run_root / "artifact_sha256.txt").read_text(encoding="ascii").splitlines()
    observed = {}
    for line in lines:
        digest, original = line.split("  ", 1)
        marker = "/slurm-1343024/"
        assert marker in original
        relative = original.split(marker, 1)[1]
        path = run_root / relative
        observed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed[relative] == digest
    assert len(observed) == 23
    assert hashlib.sha256((run_root / "artifact_sha256.txt").read_bytes()).hexdigest() == (
        "e7d3f8497cc5113241bd4610c890c0fdfcd3641da372d7a545c52d01769d63d3"
    )
