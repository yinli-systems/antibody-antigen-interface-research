#!/usr/bin/env python3
"""Freeze an auditable late-stage ABF-650M token-scaling protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_SELECTED_SEGMENTS = (58, 63, 69)
EXPECTED_FINAL_MODEL_SHA256 = (
    "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file missing: {path}")


def checkpoint_complete(path: Path) -> bool:
    if path.is_symlink() or not path.is_dir():
        return False
    metadata = path / ".metadata"
    shards = sorted(path.glob("*.distcp"))
    return (
        metadata.is_file()
        and not metadata.is_symlink()
        and len(shards) == 4
        and all(item.is_file() and not item.is_symlink() for item in shards)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    parser.add_argument("--base-transfer-protocol", type=Path, required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--control-analysis", type=Path, required=True)
    parser.add_argument("--final-analysis", type=Path, required=True)
    parser.add_argument("--final-analysis-receipt", type=Path, required=True)
    parser.add_argument("--final-embedding-manifest", type=Path, required=True)
    parser.add_argument("--final-embedding-receipt", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.base_transfer_protocol,
        args.chain_map,
        args.control_analysis,
        args.final_analysis,
        args.final_analysis_receipt,
        args.final_embedding_manifest,
        args.final_embedding_receipt,
        *args.source,
    ):
        require_regular(path)

    transfer = json.loads(args.base_transfer_protocol.read_text())
    chain_map = json.loads(args.chain_map.read_text())
    control = json.loads(args.control_analysis.read_text())
    final_analysis = json.loads(args.final_analysis.read_text())
    final_analysis_receipt = json.loads(args.final_analysis_receipt.read_text())
    final_embeddings = json.loads(args.final_embedding_manifest.read_text())
    final_embedding_receipt = json.loads(args.final_embedding_receipt.read_text())
    transfer_sha = sha256_file(args.base_transfer_protocol)
    final_analysis_sha = sha256_file(args.final_analysis)
    final_embedding_sha = sha256_file(args.final_embedding_manifest)

    if (
        transfer.get("status")
        != "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"
        or chain_map.get("status") != "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING"
        or chain_map.get("protocol_sha256") != transfer_sha
        or chain_map.get("task_labels_read") is not False
        or control.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or final_analysis.get("status")
        != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT"
        or final_analysis.get("abf30b_checkpoint_model_sha256")
        != EXPECTED_FINAL_MODEL_SHA256
        or final_analysis_receipt.get("analysis_sha256") != final_analysis_sha
        or final_analysis_receipt.get("test_loaded") is not False
        or final_embeddings.get("antibody_model", {}).get("checkpoint_model_sha256")
        != EXPECTED_FINAL_MODEL_SHA256
        or final_embedding_receipt.get("manifest_sha256") != final_embedding_sha
        or final_embedding_receipt.get("test_loaded") is not False
    ):
        raise ValueError("base transfer, control, or final checkpoint authority differs")

    retained: list[dict[str, Any]] = []
    for audit_path in sorted((args.pretraining_root / "audits").glob("segment-*.json")):
        audit = json.loads(audit_path.read_text())
        segment = int(audit["segment_index"])
        terminal = audit.get("terminal", {})
        checkpoint = args.pretraining_root / f"segment-{segment:06d}" / "run/checkpoint"
        if not checkpoint_complete(checkpoint):
            continue
        if (
            audit.get("artifact_audit", {}).get("manifest_verified") is not True
            or any(audit.get("test_access", {}).values())
            or not terminal.get("checkpoint_model_sha256")
            or not terminal.get("consumed_residue_tokens")
        ):
            raise ValueError(f"retained checkpoint audit differs: segment {segment}")
        retained.append(
            {
                "segment_index": segment,
                "consumed_residue_tokens": int(terminal["consumed_residue_tokens"]),
                "next_update": int(terminal["next_update"]),
                "checkpoint_model_sha256": terminal["checkpoint_model_sha256"],
                "checkpoint_path": str(checkpoint.resolve()),
                "audit_path": str(audit_path.resolve()),
                "audit_sha256": sha256_file(audit_path),
                "validation_masked_cross_entropy": float(
                    audit["validation"]["masked_cross_entropy"]
                ),
                "validation_masked_accuracy": float(
                    audit["validation"]["masked_accuracy"]
                ),
            }
        )
    if len(retained) < 3:
        raise ValueError("fewer than three audited retained checkpoints")
    retained.sort(key=lambda row: row["consumed_residue_tokens"])
    midpoint = (
        retained[0]["consumed_residue_tokens"]
        + retained[-1]["consumed_residue_tokens"]
    ) / 2.0
    middle = min(
        retained[1:-1],
        key=lambda row: (abs(row["consumed_residue_tokens"] - midpoint), row["segment_index"]),
    )
    selected = (retained[0], middle, retained[-1])
    selected_segments = tuple(row["segment_index"] for row in selected)
    if selected_segments != EXPECTED_SELECTED_SEGMENTS:
        raise ValueError(
            f"retained-checkpoint selection changed: {selected_segments}"
        )

    checkpoints = {
        f"segment_{row['segment_index']:06d}": row for row in selected
    }
    if checkpoints["segment_000069"]["checkpoint_model_sha256"] != EXPECTED_FINAL_MODEL_SHA256:
        raise ValueError("terminal checkpoint SHA differs")

    protocol = {
        "format_version": "abf30b_late_stage_token_scaling_protocol_v1",
        "status": "FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION",
        "study_role": "development-only late-stage checkpoint sensitivity",
        "selection_rule": (
            "Among all retained, manifest-verified checkpoints, select the earliest, "
            "the checkpoint nearest the token midpoint, and the terminal checkpoint."
        ),
        "selected_without_downstream_metrics": True,
        "retained_checkpoint_count": len(retained),
        "retained_token_range": [
            retained[0]["consumed_residue_tokens"],
            retained[-1]["consumed_residue_tokens"],
        ],
        "checkpoints": checkpoints,
        "base_transfer_protocol_sha256": transfer_sha,
        "chain_map_sha256": sha256_file(args.chain_map),
        "control_analysis_sha256": sha256_file(args.control_analysis),
        "final_checkpoint_reuse": {
            "analysis_sha256": final_analysis_sha,
            "analysis_receipt_sha256": sha256_file(args.final_analysis_receipt),
            "embedding_manifest_sha256": final_embedding_sha,
            "embedding_receipt_sha256": sha256_file(args.final_embedding_receipt),
            "six_runs_rerun": False,
        },
        "execution": {
            "new_embedding_checkpoints": ["segment_000058", "segment_000063"],
            "arms": ["baseline", "edge_topk2_supervised"],
            "seeds": [17, 29, 43],
            "task_routing": {
                "contact": "baseline same seed",
                "epitope": "edge_topk2_supervised same seed",
                "paratope": "baseline same seed",
            },
            "same_framecontact_architecture_loss_optimization_split_and_labels": True,
            "distance_loss_weight": 0.0,
            "antigen_embedding_changed": False,
        },
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(args.source, key=lambda item: str(item.resolve()))
        ],
        "claim_boundary": {
            "full_pretraining_scaling_curve": False,
            "late_stage_only": True,
            "development_only": True,
            "full_validation_is_checkpoint_selection_data": True,
            "independent_test": False,
            "sable_or_external_measurement_repeated": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "performance_gain_claim_authorized_before_frozen_analysis": False,
        },
        "task_labels_read_during_freeze": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    protocol_sha = publish(args.output, protocol)
    receipt = {
        "format_version": "abf30b_late_stage_token_scaling_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "selected_segments": list(selected_segments),
        "frozen_before_new_dcp_restore": True,
        "frozen_before_new_embedding_materialization": True,
        "frozen_before_new_development_training": True,
        "task_labels_read": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "selected_segments": selected_segments,
                "protocol_sha256": protocol_sha,
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
