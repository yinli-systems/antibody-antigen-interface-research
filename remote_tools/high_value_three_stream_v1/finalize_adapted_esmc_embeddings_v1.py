#!/usr/bin/env python3
"""Audit adapted ESM-C shards and freeze complete development-only manifests."""

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
ADAPTED_MODELS = {
    "whole_chain_adapted": {
        "name": "local/ESMC-600M-whole-chain-adapted-epoch4",
        "revision": "fixed-epoch-4",
        "weight_sha256": "50b27675df33f843e7a9a71eca4423c449603246980e3296dfbb21fc27ffa1fb",
    },
    "cdr_focused_adapted": {
        "name": "local/ESMC-600M-cdr-focused-adapted-epoch4",
        "revision": "fixed-epoch-4",
        "weight_sha256": "5fd67e390f4602f88e9a59d03a11bff332da487ebe4425bdb562bcce8a3a9d9a",
    },
}
BASE_MANIFEST_SHA256 = "cf2b6734a4614f4c260b6f1f3f861bd732549010324d7053da237b5890f182ab"


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


def audit_adapted_artifact(
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
        raise RuntimeError("adapted embedding chunk contract differs")
    chunk = chunks[0]
    if (
        artifact.get("format_version") != "sppider_seq_embedding_v1"
        or artifact.get("sequence_sha256") != expected["sequence_sha256"]
        or artifact.get("sequence_length") != expected["sequence_length"]
        or artifact.get("embedding_dimension") != 1152
        or artifact.get("input_kind") != "antibody"
        or artifact.get("paired_input") is not bool(expected.get("paired", False))
        or artifact.get("model_name") != model["name"]
        or artifact.get("model_revision") != model["revision"]
        or artifact.get("model_weight_sha256") != model["weight_sha256"]
        or artifact.get("starts") != [0]
        or artifact.get("ends") != [expected["sequence_length"]]
        or tuple(chunk.shape) != (expected["sequence_length"], 1152)
        or not bool(torch.isfinite(chunk).all())
    ):
        raise RuntimeError(f"adapted embedding artifact contract differs: {path}")


def audit_shards(
    result_root: Path,
    artifact_root: Path,
    plan: dict[str, object],
    plan_sha: str,
    protocol_sha: str,
    role: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = ADAPTED_MODELS[role]
    expected = {value["sequence_sha256"]: value for value in plan["antibodies"]}
    observed: dict[str, dict[str, object]] = {}
    evidence = []
    for shard in range(4):
        path = result_root / role / f"shard-{shard}.json"
        value = json.loads(path.read_text())
        if (
            value.get("status") != "PASS_ADAPTED_ESMC_EMBEDDING_SHARD"
            or value.get("protocol_sha256") != protocol_sha
            or value.get("plan_sha256") != plan_sha
            or value.get("role") != role
            or value.get("checkpoint_sha256") != model["weight_sha256"]
            or value.get("canary") is not False
            or value.get("shard_index") != shard
            or value.get("shard_count") != 4
            or value.get("record_count") != len(value.get("records", []))
            or value.get("audit", {}).get("strict_model_load") is not True
            or value.get("audit", {}).get("finite") is not True
            or value.get("audit", {}).get("task_labels_read") is not False
            or value.get("audit", {}).get("test_access") is not False
            or value.get("audit", {}).get("training_started") is not False
            or value.get("audit", {}).get("antigen_embedding_materialized") is not False
        ):
            raise RuntimeError(f"{role} shard result contract differs: {path}")
        for record in value["records"]:
            key = record["sequence_sha256"]
            if key not in expected or key in observed:
                raise RuntimeError(f"unexpected or duplicate antibody sequence: {key}")
            if int(key[:16], 16) % 4 != shard:
                raise RuntimeError(f"antibody assigned to wrong shard: {key}")
            if (
                record.get("sequence_length") != expected[key]["sequence_length"]
                or record.get("artifact_path") != f"{key[:2]}/{key}.pt"
            ):
                raise RuntimeError(f"adapted result record contract differs: {key}")
            audit_adapted_artifact(
                artifact_root / record["artifact_path"], record, expected[key], model
            )
            observed[key] = record
        evidence.append(
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
        raise RuntimeError("adapted antibody full sequence coverage differs")
    records = [
        {
            "artifact_path": observed[key]["artifact_path"],
            "artifact_sha256": observed[key]["artifact_sha256"],
            "sequence_length": observed[key]["sequence_length"],
            "sequence_sha256": key,
            "input_kind": "antibody",
            "embedding_source": role,
            "state": "verified_materialized",
        }
        for key in sorted(observed)
    ]
    return records, evidence


def link_base_antigens(
    base_root: Path,
    target_root: Path,
    plan: dict[str, object],
    base_manifest: dict[str, object],
) -> list[dict[str, object]]:
    base_by_sha = {
        record["sequence_sha256"]: record
        for record in base_manifest["records"]
        if record.get("input_kind") == "antigen"
    }
    records = []
    for unit in sorted(plan["antigens"], key=lambda value: value["sequence_sha256"]):
        key = unit["sequence_sha256"]
        record = base_by_sha.get(key)
        if record is None or record.get("sequence_length") != unit["sequence_length"]:
            raise RuntimeError(f"base antigen manifest coverage differs: {key}")
        source = base_root / record["artifact_path"]
        if sha256_file(source) != record["artifact_sha256"]:
            raise RuntimeError(f"base antigen artifact SHA differs: {source}")
        target = target_root / record["artifact_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if sha256_file(target) != record["artifact_sha256"]:
                raise RuntimeError(f"existing shared antigen artifact differs: {target}")
        else:
            os.link(source, target)
        records.append(
            {
                "artifact_path": record["artifact_path"],
                "artifact_sha256": record["artifact_sha256"],
                "sequence_length": record["sequence_length"],
                "sequence_sha256": key,
                "input_kind": "antigen",
                "embedding_source": "base_shared_antigen",
                "state": "verified_hardlink_from_base",
            }
        )
    return records


def make_manifest(
    plan: dict[str, object],
    plan_sha: str,
    execution_sha: str,
    role: str,
    records: list[dict[str, object]],
    evidence: list[dict[str, object]],
) -> dict[str, object]:
    model = ADAPTED_MODELS[role]
    return {
        "schema_version": "sabdab2_esm_embedding_manifest_v1",
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model["name"],
        "model_revision": model["revision"],
        "model_weight_sha256": model["weight_sha256"],
        "component_models": {"antibody": model, "antigen": BASE_MODEL},
        "embedding_dimension": 1152,
        "selected_sequence_count": len(records),
        "complete_cohort_sequence_count": len(records),
        "records": sorted(records, key=lambda value: value["sequence_sha256"]),
        "runtime": {"shards": evidence},
        "authority": {
            "development_sequence_plan_sha256": plan_sha,
            "adapted_embedding_full_execution_sha256": execution_sha,
            "train_panel_sha256": plan["authority"]["train_panel_sha256"],
            "validation_panel_sha256": plan["authority"]["validation_panel_sha256"],
            "train_instances": plan["counts"]["train_instances"],
            "validation_instances": plan["counts"]["validation_instances"],
            "test_access": False,
            "task_labels_read": False,
        },
        "claim_boundary": (
            "Frozen train/validation SAbDab2 development embeddings only; no test access "
            "or downstream performance claim. Antigens reuse the same verified base ESM-C cache."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--whole-root", type=Path, required=True)
    parser.add_argument("--cdr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.execution) != args.execution_sha256:
        raise RuntimeError("full adapted embedding execution SHA differs")
    if sha256_file(args.base_manifest) != BASE_MANIFEST_SHA256:
        raise RuntimeError("base development embedding manifest SHA differs")
    execution = json.loads(args.execution.read_text())
    plan = json.loads(args.plan.read_text())
    base_manifest = json.loads(args.base_manifest.read_text())
    plan_sha = sha256_file(args.plan)
    if (
        execution.get("status") != "FROZEN_AFTER_CANARY_BEFORE_FULL_EMBEDDING"
        or execution.get("development_sequence_plan") != str(args.plan)
        or execution.get("development_sequence_plan_sha256") != plan_sha
        or execution.get("outputs", {}).get("result_root") != str(args.result_root)
        or execution.get("outputs", {}).get("whole_chain_adapted_root") != str(args.whole_root)
        or execution.get("outputs", {}).get("cdr_focused_adapted_root") != str(args.cdr_root)
        or execution.get("source_binding", {}).get("full_finalizer_sha256")
        != sha256_file(Path(__file__))
        or execution.get("authorization", {}).get(
            "full_development_embedding_materialization_authorized"
        )
        is not True
        or execution.get("test_access") is not False
        or plan.get("authorization", {}).get("test_access") is not False
        or base_manifest.get("complete") is not True
        or base_manifest.get("authority", {}).get("test_access") is not False
        or base_manifest.get("authority", {}).get("task_labels_read") is not False
    ):
        raise RuntimeError("full adapted embedding authority differs")

    manifests = {}
    counts = {}
    for role, root in (
        ("whole_chain_adapted", args.whole_root),
        ("cdr_focused_adapted", args.cdr_root),
    ):
        antibody_records, evidence = audit_shards(
            args.result_root,
            root,
            plan,
            plan_sha,
            execution["parent_protocol_sha256"],
            role,
        )
        antigen_records = link_base_antigens(args.base_root, root, plan, base_manifest)
        records = antibody_records + antigen_records
        expected_total = len(plan["antibodies"]) + len(plan["antigens"])
        if len(records) != expected_total:
            raise RuntimeError(f"{role} final embedding coverage differs")
        manifest_path = root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
        atomic_write(
            manifest_path,
            make_manifest(plan, plan_sha, args.execution_sha256, role, records, evidence),
        )
        manifests[role] = {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        counts[role] = {
            "adapted_antibody_unique_sequences": len(antibody_records),
            "shared_base_antigen_unique_sequences": len(antigen_records),
            "combined_unique_sequences": len(records),
        }

    result = {
        "format_version": "sabdab2_adapted_esmc_embedding_full_analysis_v1",
        "status": "PASS_ADAPTED_ESMC_DEVELOPMENT_EMBEDDINGS_VERIFIED",
        "execution_sha256": args.execution_sha256,
        "parent_protocol_sha256": execution["parent_protocol_sha256"],
        "development_sequence_plan_sha256": plan_sha,
        "counts": counts,
        "manifests": manifests,
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
            "test_access": False,
        },
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "sabdab2_adapted_esmc_embedding_full_analysis_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "execution_sha256": args.execution_sha256,
        "whole_chain_manifest_sha256": manifests["whole_chain_adapted"]["sha256"],
        "cdr_focused_manifest_sha256": manifests["cdr_focused_adapted"]["sha256"],
        "same_protocol_downstream_development_screen_authorized": True,
        "test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
