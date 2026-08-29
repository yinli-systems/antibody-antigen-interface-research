#!/usr/bin/env python3
"""Freeze a create-only closure receipt for a node-specific non-finite incident."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


FORMAT_VERSION = "antibody_node_specific_nonfinite_incident_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in result:
            result[key] = value
    return result


def verify_manifest(root: Path) -> dict[str, object]:
    manifest = root / "artifact_sha256.txt"
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError(f"Artifact manifest is invalid: {manifest}")
    checked = 0
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or len(digest) != 64:
            raise RuntimeError(f"Malformed artifact manifest row: {line!r}")
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        if sha256_file(path) != digest:
            raise RuntimeError(f"Artifact hash differs: {path}")
        checked += 1
    if checked == 0:
        raise RuntimeError(f"Artifact manifest is empty: {manifest}")
    return {
        "artifact_count": checked,
        "manifest_sha256": sha256_file(manifest),
    }


def require_text(path: Path, required: tuple[str, ...], forbidden: tuple[str, ...] = ()) -> str:
    text = path.read_text(encoding="utf-8")
    missing = [value for value in required if value not in text]
    present = [value for value in forbidden if value in text]
    if missing or present:
        raise RuntimeError(
            f"Diagnostic signature differs for {path}: missing={missing}, forbidden={present}"
        )
    return text


def diagnostic_evidence(
    root: Path,
    *,
    expected_job_id: int,
    expected_node: str,
    expected_gpu_uuid: str,
    required: tuple[str, ...],
    forbidden: tuple[str, ...] = (),
) -> dict[str, object]:
    fields = environment_fields(root / "environment.txt")
    if fields.get("job_id") != str(expected_job_id):
        raise RuntimeError(f"Diagnostic job identity differs: {root}")
    environment_text = (root / "environment.txt").read_text(encoding="utf-8")
    if expected_gpu_uuid not in environment_text:
        raise RuntimeError(f"Diagnostic physical GPU identity differs: {root}")
    require_text(root / "diagnostic.log", required, forbidden)
    return {
        "artifacts": verify_manifest(root),
        "job_id": expected_job_id,
        "log_sha256": sha256_file(root / "diagnostic.log"),
        "node": expected_node,
        "physical_gpu_uuid_anchor": expected_gpu_uuid,
    }


def canonical_bytes(value: object) -> bytes:
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


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite incident receipt: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--failed-job-id", type=int, required=True)
    parser.add_argument("--failed-watchdog-job-id", type=int, required=True)
    parser.add_argument("--bad-node", required=True)
    parser.add_argument("--bad-gpu-uuid", required=True)
    parser.add_argument("--good-node", required=True)
    parser.add_argument("--good-gpu-uuid", required=True)
    parser.add_argument("--diagnostic-v2-job-id", type=int, required=True)
    parser.add_argument("--diagnostic-v3-job-id", type=int, required=True)
    parser.add_argument("--diagnostic-v4-job-id", type=int, required=True)
    parser.add_argument("--diagnostic-v5-job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    campaign = arguments.campaign_root.resolve()
    failed = campaign / "segment-000004"
    failed_environment = environment_fields(failed / "environment.txt")
    if (
        failed_environment.get("job_id") != str(arguments.failed_job_id)
        or failed_environment.get("node") != arguments.bad_node
        or failed_environment.get("segment_index") != "4"
    ):
        raise RuntimeError("Failed segment identity differs")
    failed_environment_text = (failed / "environment.txt").read_text(encoding="utf-8")
    if arguments.bad_gpu_uuid not in failed_environment_text:
        raise RuntimeError("Failed physical GPU identity differs")
    require_text(
        failed / "training.log",
        ("Gate trace contains a non-finite value", "rank      : 1"),
    )
    if any((failed / "run").iterdir()):
        raise RuntimeError("Failed segment unexpectedly published run state")

    incident_root = campaign / "incidents"
    diagnostics = {
        "all_rank_post_sync": diagnostic_evidence(
            incident_root / "segment-000004-nonfinite-diagnostic-v2",
            expected_job_id=arguments.diagnostic_v2_job_id,
            expected_node=arguments.bad_node,
            expected_gpu_uuid=arguments.bad_gpu_uuid,
            required=("NONFINITE_GATE_DIAGNOSTIC=", '"rank":0', '"rank":3'),
        ),
        "pre_sync_micro": diagnostic_evidence(
            incident_root / "segment-000004-nonfinite-diagnostic-v3",
            expected_job_id=arguments.diagnostic_v3_job_id,
            expected_node=arguments.bad_node,
            expected_gpu_uuid=arguments.bad_gpu_uuid,
            required=(
                "NONFINITE_MICRO_DIAGNOSTIC=",
                '"rank":1',
                "2aad616df61eb95a2fecb14dbd59c7f863855c595c675f9b4b9bc472f3505d2d",
            ),
        ),
        "rank_one_subsets": diagnostic_evidence(
            incident_root / "segment-000004-nonfinite-diagnostic-v4",
            expected_job_id=arguments.diagnostic_v4_job_id,
            expected_node=arguments.bad_node,
            expected_gpu_uuid=arguments.bad_gpu_uuid,
            required=(
                "NONFINITE_SUBSET_DIAGNOSTIC=",
                '"bad_subsets":[],"rank":0,"tested_subsets":31',
                '"rank":1,"tested_subsets":31',
                '"bad_subsets":[],"rank":3,"tested_subsets":31',
            ),
        ),
        "alternate_node_two_update_control": diagnostic_evidence(
            incident_root / "segment-000004-nonfinite-diagnostic-v5",
            expected_job_id=arguments.diagnostic_v5_job_id,
            expected_node=arguments.good_node,
            expected_gpu_uuid=arguments.good_gpu_uuid,
            required=("Diagnostic update limit reached without reproduction",),
            forbidden=("NONFINITE_GATE_DIAGNOSTIC=", "NONFINITE_MICRO_DIAGNOSTIC="),
        ),
    }

    payload: dict[str, object] = {
        "authorization": {
            "failed_attempt_archive": True,
            "production_segment_retry": True,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Operational closure of a node-specific non-finite-gradient incident. "
            "The retry must preserve the frozen checkpoint, data order, optimizer, "
            "sampler, seed, and scientific hyperparameters while excluding the bad node."
        ),
        "diagnosis": {
            "checkpoint_corruption_supported": False,
            "data_sample_root_cause_supported": False,
            "node_or_physical_gpu_path_specific": True,
        },
        "diagnostics": diagnostics,
        "failed_attempt": {
            "artifacts": verify_manifest(failed),
            "job_id": arguments.failed_job_id,
            "node": arguments.bad_node,
            "physical_rank_one_gpu_uuid": arguments.bad_gpu_uuid,
            "published_checkpoint": False,
            "training_log_sha256": sha256_file(failed / "training.log"),
            "watchdog_job_id": arguments.failed_watchdog_job_id,
        },
        "format_version": FORMAT_VERSION,
        "required_exclusion": {"nodes": [arguments.bad_node]},
        "status": "NODE_SPECIFIC_INCIDENT_CLOSED_RETRY_AUTHORIZED",
        "test_access": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    output = arguments.output.resolve()
    publish_create_only(output, canonical_bytes(payload))
    print(canonical_bytes(payload).decode("ascii"), end="")


if __name__ == "__main__":
    main()
