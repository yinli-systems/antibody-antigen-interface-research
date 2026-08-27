#!/usr/bin/env python3
"""Audit full ESM-C embedding shards and freeze development-only manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch


BASE_MODEL = {
    "name": "biohub/ESMC-600M",
    "revision": "a7e82012c83126b9eedb055fea9fa84b6c02f094",
    "weight_sha256": "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8",
}
CDR_MODEL = {
    "name": "NOC-Lab/AbCDR-ESMC",
    "revision": "d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8",
    "weight_sha256": "0490932ca515c6fe1738d070ede8b3ade28745d990a5e1325f286f3624c1d4e6",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def expected_units(plan: dict[str, object], role: str) -> dict[str, dict[str, object]]:
    units = {
        value["sequence_sha256"]: {**value, "kind": "antibody"}
        for value in plan["antibodies"]
    }
    if role == "base":
        for value in plan["antigens"]:
            key = value["sequence_sha256"]
            if key in units:
                raise RuntimeError("antibody/antigen sequence SHA collision")
            units[key] = {**value, "kind": "antigen"}
    return units


def audit_artifact(
    path: Path,
    record: dict[str, object],
    expected: dict[str, object],
    model: dict[str, str],
) -> None:
    if sha256_file(path) != record["artifact_sha256"]:
        raise RuntimeError(f"artifact SHA differs: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    chunks = artifact.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 1:
        raise RuntimeError("embedding chunk contract differs")
    chunk = chunks[0]
    if (
        artifact.get("format_version") != "sppider_seq_embedding_v1"
        or artifact.get("sequence_sha256") != expected["sequence_sha256"]
        or artifact.get("sequence_length") != expected["sequence_length"]
        or artifact.get("embedding_dimension") != 1152
        or artifact.get("input_kind") != expected["kind"]
        or artifact.get("model_name") != model["name"]
        or artifact.get("model_revision") != model["revision"]
        or artifact.get("model_weight_sha256") != model["weight_sha256"]
        or artifact.get("starts") != [0]
        or artifact.get("ends") != [expected["sequence_length"]]
        or tuple(chunk.shape) != (expected["sequence_length"], 1152)
        or not bool(torch.isfinite(chunk).all())
    ):
        raise RuntimeError(f"embedding artifact contract differs: {path}")


def audit_shards(
    result_root: Path,
    artifact_root: Path,
    plan: dict[str, object],
    plan_sha: str,
    role: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = BASE_MODEL if role == "base" else CDR_MODEL
    expected = expected_units(plan, role)
    observed: dict[str, dict[str, object]] = {}
    shard_evidence = []
    for shard in range(4):
        path = result_root / role / f"shard-{shard}.json"
        value = json.loads(path.read_text())
        if (
            value.get("status") != "PASS_ESMC_EMBEDDING_SHARD"
            or value.get("role") != role
            or value.get("canary") is not False
            or value.get("plan_sha256") != plan_sha
            or value.get("checkpoint_sha256") != model["weight_sha256"]
            or value.get("shard_index") != shard
            or value.get("shard_count") != 4
            or value.get("record_count") != len(value.get("records", []))
            or value.get("audit", {}).get("strict_model_load") is not True
            or value.get("audit", {}).get("finite") is not True
            or value.get("audit", {}).get("task_labels_read") is not False
            or value.get("audit", {}).get("test_access") is not False
            or value.get("audit", {}).get("training_started") is not False
        ):
            raise RuntimeError(f"{role} shard result contract differs: {path}")
        for record in value["records"]:
            key = record["sequence_sha256"]
            if key not in expected or key in observed:
                raise RuntimeError(f"unexpected or duplicate sequence: {key}")
            if int(key[:16], 16) % 4 != shard:
                raise RuntimeError(f"sequence assigned to wrong shard: {key}")
            if (
                record.get("kind") != expected[key]["kind"]
                or record.get("sequence_length") != expected[key]["sequence_length"]
                or record.get("artifact_path") != f"{key[:2]}/{key}.pt"
            ):
                raise RuntimeError(f"result record contract differs: {key}")
            artifact_path = artifact_root / record["artifact_path"]
            audit_artifact(artifact_path, record, expected[key], model)
            observed[key] = record
        shard_evidence.append(
            {
                "shard_index": shard,
                "result_path": str(path),
                "result_sha256": sha256_file(path),
                "record_count": value["record_count"],
                "elapsed_seconds": value["runtime"]["elapsed_seconds"],
                "peak_allocated_bytes": value["runtime"]["peak_allocated_bytes"],
            }
        )
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise RuntimeError(f"{role} full sequence coverage differs; missing={missing[:3]}")
    records = []
    for key in sorted(observed):
        record = observed[key]
        records.append(
            {
                "artifact_path": record["artifact_path"],
                "artifact_sha256": record["artifact_sha256"],
                "sequence_length": record["sequence_length"],
                "sequence_sha256": key,
                "input_kind": record["kind"],
                "embedding_source": role,
                "state": "verified_materialized",
            }
        )
    return records, shard_evidence


def link_base_antigens(
    base_root: Path,
    cdr_root: Path,
    plan: dict[str, object],
    base_records: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    shared = []
    for unit in sorted(plan["antigens"], key=lambda value: value["sequence_sha256"]):
        key = unit["sequence_sha256"]
        source = base_root / key[:2] / f"{key}.pt"
        target = cdr_root / key[:2] / f"{key}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if sha256_file(target) != base_records[key]["artifact_sha256"]:
                raise RuntimeError(f"existing shared antigen artifact differs: {target}")
        else:
            os.link(source, target)
        if sha256_file(target) != base_records[key]["artifact_sha256"]:
            raise RuntimeError(f"shared antigen artifact SHA differs: {target}")
        shared.append(
            {
                **base_records[key],
                "embedding_source": "base_shared_antigen",
                "state": "verified_hardlink_from_base",
            }
        )
    return shared


def manifest(
    plan: dict[str, object],
    plan_sha: str,
    role: str,
    records: list[dict[str, object]],
    shard_evidence: list[dict[str, object]],
) -> dict[str, object]:
    if role == "base":
        model_name = BASE_MODEL["name"]
        model_revision = BASE_MODEL["revision"]
        model_weight_sha = BASE_MODEL["weight_sha256"]
        component_models = {"antibody": BASE_MODEL, "antigen": BASE_MODEL}
    else:
        model_name = CDR_MODEL["name"]
        model_revision = CDR_MODEL["revision"]
        model_weight_sha = CDR_MODEL["weight_sha256"]
        component_models = {"antibody": CDR_MODEL, "antigen": BASE_MODEL}
    return {
        "schema_version": "sabdab2_esm_embedding_manifest_v1",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_revision": model_revision,
        "model_weight_sha256": model_weight_sha,
        "component_models": component_models,
        "embedding_dimension": 1152,
        "selected_sequence_count": len(records),
        "complete_cohort_sequence_count": len(records),
        "records": sorted(records, key=lambda value: value["sequence_sha256"]),
        "runtime": {"shards": shard_evidence},
        "authority": {
            "development_sequence_plan_sha256": plan_sha,
            "train_panel_sha256": plan["authority"]["train_panel_sha256"],
            "validation_panel_sha256": plan["authority"]["validation_panel_sha256"],
            "train_instances": plan["counts"]["train_instances"],
            "validation_instances": plan["counts"]["validation_instances"],
            "test_access": False,
            "task_labels_read": False,
        },
        "claim_boundary": (
            "Frozen train/validation SAbDab2 development embeddings only; no test access, "
            "task-label use, downstream performance, or local CDR-pretraining claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--cdr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise RuntimeError("development sequence plan SHA differs")
    plan = json.loads(args.plan.read_text())
    if (
        plan.get("status") != "FROZEN_TRAIN_VALIDATION_ONLY_BEFORE_EMBEDDING"
        or plan.get("authorization", {}).get("full_embedding_materialization_authorized") is not False
        or plan.get("authorization", {}).get("test_access") is not False
    ):
        raise RuntimeError("development sequence plan contract differs")
    canary_receipt = json.loads(args.canary_receipt.read_text())
    if (
        canary_receipt.get("status") != "PASS_BASE_AND_CDR_EMBEDDING_CANARY"
        or canary_receipt.get("plan_sha256") != args.plan_sha256
        or canary_receipt.get("full_embedding_materialization_authorized") is not True
        or canary_receipt.get("training_authorized") is not False
        or canary_receipt.get("test_access") is not False
    ):
        raise RuntimeError("embedding canary authorization differs")

    base_records, base_evidence = audit_shards(
        args.result_root, args.base_root, plan, args.plan_sha256, "base"
    )
    cdr_antibody_records, cdr_evidence = audit_shards(
        args.result_root, args.cdr_root, plan, args.plan_sha256, "published_cdr"
    )
    base_by_sha = {record["sequence_sha256"]: record for record in base_records}
    cdr_records = cdr_antibody_records + link_base_antigens(
        args.base_root, args.cdr_root, plan, base_by_sha
    )
    expected_total = len(plan["antibodies"]) + len(plan["antigens"])
    if len(base_records) != expected_total or len(cdr_records) != expected_total:
        raise RuntimeError("final embedding manifest coverage differs")

    base_manifest_path = args.base_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    cdr_manifest_path = args.cdr_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    atomic_write(
        base_manifest_path,
        manifest(plan, args.plan_sha256, "base", base_records, base_evidence),
    )
    atomic_write(
        cdr_manifest_path,
        manifest(plan, args.plan_sha256, "published_cdr", cdr_records, cdr_evidence),
    )
    result = {
        "format_version": "sabdab2_esmc_embedding_full_analysis_v1",
        "status": "PASS_BASE_AND_PUBLISHED_CDR_DEVELOPMENT_EMBEDDINGS_VERIFIED",
        "plan_sha256": args.plan_sha256,
        "canary_receipt_sha256": sha256_file(args.canary_receipt),
        "counts": {
            "development_instances": plan["counts"]["development_instances"],
            "base_unique_sequences": len(base_records),
            "cdr_antibody_unique_sequences": len(cdr_antibody_records),
            "shared_base_antigen_unique_sequences": len(plan["antigens"]),
            "cdr_combined_unique_sequences": len(cdr_records),
        },
        "manifests": {
            "base": {"path": str(base_manifest_path), "sha256": sha256_file(base_manifest_path)},
            "published_cdr": {"path": str(cdr_manifest_path), "sha256": sha256_file(cdr_manifest_path)},
        },
        "audit": {
            "all_shards_verified": True,
            "all_artifact_sha256_verified": True,
            "all_shapes_and_finite_verified": True,
            "exact_development_sequence_coverage": True,
            "antigen_artifacts_shared_without_reembedding": True,
            "task_labels_read": False,
            "test_access": False,
        },
        "authorization": {
            "same_protocol_downstream_development_screen_authorized": True,
            "local_esmc_adaptation_training_authorized": False,
            "test_access": False,
        },
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "sabdab2_esmc_embedding_full_analysis_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "base_manifest_sha256": result["manifests"]["base"]["sha256"],
        "published_cdr_manifest_sha256": result["manifests"]["published_cdr"]["sha256"],
        "same_protocol_downstream_development_screen_authorized": True,
        "local_esmc_adaptation_training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
