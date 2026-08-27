#!/usr/bin/env python3
"""Freeze train/validation-only ESM-C inputs without materializing test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"SHA differs: {path}")


def panel_ids(path: Path, expected_sha: str, expected_role: str) -> list[str]:
    verify(path, expected_sha)
    panel = json.loads(path.read_text())
    ids = [entry["instance_id"] for entry in panel.get("entries", [])]
    if (
        panel.get("format_version") != "framecontact_reciprocal_partner_panel_v2"
        or panel.get("role") != expected_role
        or panel.get("authority", {}).get("task_label_columns_read") is not False
        or len(ids) != len(set(ids))
    ):
        raise RuntimeError(f"{expected_role} panel contract differs")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--train-panel-sha256", required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-panel-sha256", required=True)
    parser.add_argument("--sequence-authority", type=Path, required=True)
    parser.add_argument("--sequence-authority-sha256", required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--runtime-result", type=Path, required=True)
    parser.add_argument("--runtime-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    train_ids = panel_ids(args.train_panel, args.train_panel_sha256, "train")
    validation_ids = panel_ids(
        args.validation_panel, args.validation_panel_sha256, "validation"
    )
    if len(train_ids) != 2565 or len(validation_ids) != 549:
        raise RuntimeError("development role counts differ")
    development_ids = set(train_ids) | set(validation_ids)
    if len(development_ids) != 3114:
        raise RuntimeError("development panels overlap")
    verify(args.sequence_authority, args.sequence_authority_sha256)
    verify(args.structure_root / "STRUCTURE_CACHE_MANIFEST.json", args.structure_manifest_sha256)
    verify(args.runtime_result, args.runtime_result_sha256)
    runtime = json.loads(args.runtime_result.read_text())
    if (
        runtime.get("status") != "PASS_BASE_AND_PUBLISHED_CDR_RUNTIME_VERIFIED"
        or runtime.get("authorization", {}).get("embedding_materialization_authorized")
        is not True
        or runtime.get("authorization", {}).get("test_access") is not False
    ):
        raise RuntimeError("runtime gate differs")

    import pyarrow.parquet as pq

    table = pq.read_table(
        args.sequence_authority,
        columns=["instance_id", "ab_seq", "ag_seq"],
        filters=[("instance_id", "in", sorted(development_ids))],
    )
    rows = table.to_pylist()
    if len(rows) != 3114 or {row["instance_id"] for row in rows} != development_ids:
        raise RuntimeError("filtered development sequence authority differs")
    row_by_id = {row["instance_id"]: row for row in rows}
    structure = json.loads(
        (args.structure_root / "STRUCTURE_CACHE_MANIFEST.json").read_text()
    )
    record_by_id = {
        record["instance_id"]: record
        for record in structure.get("records", [])
        if record["instance_id"] in development_ids
    }
    if set(record_by_id) != development_ids:
        raise RuntimeError("development structure records differ")

    role_by_id = {identifier: "train" for identifier in train_ids}
    role_by_id.update({identifier: "validation" for identifier in validation_ids})
    antibodies: dict[str, dict[str, object]] = {}
    antigens: dict[str, dict[str, object]] = {}
    instances = []
    substitution_total = 0
    paired_instances = 0
    single_chain_instances = 0
    allowed = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    for identifier in sorted(development_ids):
        row = row_by_id[identifier]
        ab_sequence = str(row["ab_seq"]).strip().upper()
        ag_sequence = str(row["ag_seq"]).strip().upper()
        if not ab_sequence or not ag_sequence or set(ab_sequence) - allowed or set(ag_sequence) - allowed:
            raise RuntimeError(f"unsupported sequence alphabet: {identifier}")
        record = record_by_id[identifier]
        artifact_path = args.structure_root / record["artifact_path"]
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise RuntimeError(f"structure artifact SHA differs: {identifier}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        chain_ids = artifact["antibody_chain_ids"].tolist()
        groups = []
        for chain_id in chain_ids:
            if not groups or groups[-1][0] != int(chain_id):
                groups.append([int(chain_id), 1])
            else:
                groups[-1][1] += 1
        if len(groups) not in (1, 2) or sum(group[1] for group in groups) != len(ab_sequence):
            raise RuntimeError(f"antibody chain layout differs: {identifier}")
        ab_sha = sequence_sha256(ab_sequence)
        ag_sha = sequence_sha256(ag_sequence)
        if (
            ab_sha != record["antibody_sequence_sha256"]
            or ag_sha != record["antigen_sequence_sha256"]
            or len(ab_sequence) != int(record["antibody_length"])
            or len(ag_sequence) != int(record["antigen_length"])
        ):
            raise RuntimeError(f"sequence/structure authority differs: {identifier}")
        if len(groups) == 2:
            heavy_length = groups[0][1]
            model_ab = ab_sequence[:heavy_length] + "-" + ab_sequence[heavy_length:]
            paired_instances += 1
        else:
            heavy_length = len(ab_sequence)
            model_ab = ab_sequence
            single_chain_instances += 1
        normalized_ab = re.sub(r"[UZOB]", "X", model_ab)
        normalized_ag = re.sub(r"[UZOB]", "X", ag_sequence)
        substitution_total += sum(a != b for a, b in zip(model_ab, normalized_ab))
        substitution_total += sum(a != b for a, b in zip(ag_sequence, normalized_ag))
        ab_value = {
            "sequence_sha256": ab_sha,
            "output_sequence": ab_sequence,
            "model_input_sequence": normalized_ab,
            "sequence_length": len(ab_sequence),
            "heavy_length": heavy_length,
            "light_length": len(ab_sequence) - heavy_length,
            "paired": len(groups) == 2,
        }
        ag_value = {
            "sequence_sha256": ag_sha,
            "output_sequence": ag_sequence,
            "model_input_sequence": normalized_ag,
            "sequence_length": len(ag_sequence),
        }
        if ab_sha in antibodies and antibodies[ab_sha] != ab_value:
            raise RuntimeError("same antibody SHA has inconsistent chain layout")
        if ag_sha in antigens and antigens[ag_sha] != ag_value:
            raise RuntimeError("same antigen SHA has inconsistent input")
        antibodies[ab_sha] = ab_value
        antigens[ag_sha] = ag_value
        instances.append(
            {
                "instance_id": identifier,
                "role": role_by_id[identifier],
                "antibody_sequence_sha256": ab_sha,
                "antigen_sequence_sha256": ag_sha,
            }
        )
    overlap = set(antibodies) & set(antigens)
    if overlap:
        raise RuntimeError("antibody/antigen SHA collision prevents one-root caching")
    plan = {
        "format_version": "sabdab2_esmc_development_sequence_plan_v1",
        "status": "FROZEN_TRAIN_VALIDATION_ONLY_BEFORE_EMBEDDING",
        "authority": {
            "train_panel_sha256": args.train_panel_sha256,
            "validation_panel_sha256": args.validation_panel_sha256,
            "sequence_authority_sha256": args.sequence_authority_sha256,
            "structure_manifest_sha256": args.structure_manifest_sha256,
            "runtime_result_sha256": args.runtime_result_sha256,
            "columns_materialized": ["instance_id", "ab_seq", "ag_seq"],
            "parquet_filter": "instance_id in frozen train/validation panel identifiers",
            "test_rows_returned": False,
            "task_labels_read": False,
        },
        "counts": {
            "train_instances": len(train_ids),
            "validation_instances": len(validation_ids),
            "development_instances": len(instances),
            "paired_antibody_instances": paired_instances,
            "single_chain_antibody_instances": single_chain_instances,
            "unique_antibody_sequences": len(antibodies),
            "unique_antigen_sequences": len(antigens),
            "uzob_to_x_substitutions": substitution_total,
        },
        "input_semantics": {
            "two_chain_antibody": "VH + literal '-' token + VL",
            "single_chain_antibody": "single observed chain without synthetic separator",
            "antigen": "base ESM-C in both comparison arms",
            "separator_token_id": 30,
            "output_axis": "BOS/EOS and antibody hyphen removed; original residue order retained",
        },
        "antibodies": [antibodies[key] for key in sorted(antibodies)],
        "antigens": [antigens[key] for key in sorted(antigens)],
        "instances": instances,
        "authorization": {
            "embedding_canary_authorized": True,
            "full_embedding_materialization_authorized": False,
            "training_authorized": False,
            "test_access": False,
        },
    }
    atomic_write(args.output, canonical_bytes(plan))
    receipt = {
        "format_version": "sabdab2_esmc_development_sequence_plan_receipt_v1",
        "status": plan["status"],
        "plan_sha256": sha256_file(args.output),
        "counts": plan["counts"],
        "embedding_canary_authorized": True,
        "full_embedding_materialization_authorized": False,
        "training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
