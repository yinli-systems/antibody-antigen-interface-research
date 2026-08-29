"""Fail-closed checks for the external-baseline evidence ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPOSITORY_ROOT / "baselines/BASELINE_LOCK_V4.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    return (REPOSITORY_ROOT / value).resolve()


def test_external_baseline_lock_is_parent_bound_and_claim_safe() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["format_version"] == "framecontact_baseline_lock_v4"
    parent = _resolve(lock["parent"]["path"])
    assert _sha256(parent) == lock["parent"]["sha256"]
    for baseline in lock["external_baselines"].values():
        assert baseline["result_status"] == "not_run"
        assert baseline["identical_split_runs"] == 0
    pairformer = lock["external_baselines"]["boltz1_pairformer"]
    assert pairformer["real_weight_smoke"]["benchmark_result"] is False
    assert pairformer["native_boltz_pipeline_reproduced"] is False


def test_present_external_sources_match_their_pinned_revisions() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    for baseline in lock["external_baselines"].values():
        source = _resolve(baseline["local_source_path"])
        if not source.is_dir():
            continue
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert revision == baseline["source_revision"]


def test_present_pairformer_artifacts_match_every_bound_hash_and_receipt() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pairformer = lock["external_baselines"]["boltz1_pairformer"]
    checkpoint = pairformer["official_checkpoint"]
    artifact = pairformer["extracted_pairformer_artifact"]
    smoke = pairformer["real_weight_smoke"]
    requirements = pairformer["adapter"]
    for descriptor in (checkpoint, artifact):
        path = _resolve(descriptor["local_path"])
        if path.is_file():
            assert path.stat().st_size == descriptor["bytes"]
            assert _sha256(path) == descriptor["sha256"]
    for descriptor in (artifact, smoke):
        path = _resolve(descriptor["receipt_path"])
        assert _sha256(path) == descriptor["receipt_sha256"]
    requirements_path = _resolve(requirements["requirements_lock"])
    assert _sha256(requirements_path) == requirements["requirements_lock_sha256"]
    extraction = json.loads(
        _resolve(artifact["receipt_path"]).read_text(encoding="utf-8")
    )
    assert extraction["artifact_sha256"] == artifact["sha256"]
    assert extraction["state_tensor_count"] == artifact["state_tensor_count"]
    assert extraction["architecture"] == artifact["architecture"]
    smoke_payload = json.loads(
        _resolve(smoke["receipt_path"]).read_text(encoding="utf-8")
    )
    assert smoke_payload["artifact_sha256"] == artifact["sha256"]
    assert smoke_payload["strict_state_load"] is True
    assert smoke_payload["real_weight_forward"] is True
    assert smoke_payload["biological_data_read"] is False
    assert smoke_payload["benchmark_result"] is False
