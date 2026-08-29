#!/usr/bin/env python3
"""Audit one immutable four-GPU OAS pretraining segment and its chain link."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_production_batch_efficiency as artifacts
import freeze_antibody_four_gpu_pretraining_campaign as freezer


FORMAT_VERSION = "antibody_four_gpu_pretraining_segment_audit_v1"
RUN_FORMAT = "antibody_four_gpu_pretraining_segment_run_v1"
EVALUATION_FORMAT = "antibody_production_checkpoint_evaluation_v1"
EVALUATION_AUTHORIZATION = {
    "checkpoint_evaluation": True,
    "configuration_promotion": False,
    "production_pretraining": False,
    "scientific_claim": False,
}
EVALUATION_TEST_ACCESS = {
    "downstream_labels_used": False,
    "external_test_accessed": False,
    "frozen_test_accessed": False,
    "official_test_accessed": False,
    "sabdab2_heldout_accessed": False,
}


class SegmentAuditError(RuntimeError):
    """Report an invalid segment, evaluation, or previous-chain link."""


def _canonical_json_bytes(value: object) -> bytes:
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


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SegmentAuditError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise SegmentAuditError(f"Non-finite JSON constant: {value}")


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise SegmentAuditError(f"Evidence path is invalid: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SegmentAuditError(f"Evidence is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise SegmentAuditError(f"Evidence is not canonical JSON: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _receipt(path: Path) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _load(path)
    payload = dict(value)
    receipt_sha256 = payload.pop("receipt_sha256", None)
    if receipt_sha256 != hashlib.sha256(_canonical_json_bytes(payload)).hexdigest():
        raise SegmentAuditError(f"Receipt payload hash differs: {path}")
    return value, file_sha256


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SegmentAuditError(f"{label} is not finite")
    return result


def _validate_trace(
    trace: object,
    *,
    previous_consumed: int,
    previous_update: int,
) -> tuple[int, int]:
    if not isinstance(trace, list) or not trace:
        raise SegmentAuditError("Segment trace is empty")
    consumed = previous_consumed
    update = previous_update
    for item in trace:
        update += 1
        if not isinstance(item, Mapping) or item.get("update") != update:
            raise SegmentAuditError("Segment updates are not contiguous")
        residue_tokens = item.get("residue_tokens")
        next_consumed = item.get("consumed_residue_tokens")
        if (
            isinstance(residue_tokens, bool)
            or not isinstance(residue_tokens, int)
            or residue_tokens <= 0
            or isinstance(next_consumed, bool)
            or not isinstance(next_consumed, int)
            or next_consumed != consumed + residue_tokens
            or int(item.get("micro_steps", 0)) <= 0
            or int(item.get("selected_targets", 0)) <= 0
            or int(item.get("sequence_count", 0)) <= 0
            or _finite(item.get("seconds"), "step seconds") <= 0.0
            or _finite(item.get("loss"), "loss") < 0.0
            or _finite(item.get("gradient_norm"), "gradient norm") < 0.0
            or _finite(item.get("learning_rate"), "learning rate") < 0.0
        ):
            raise SegmentAuditError("Segment trace contains an invalid update")
        consumed = next_consumed
    return consumed, update


def _validate_evaluation(
    value: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    gate_sha256: str,
    run_receipt_sha256: str,
    run_state_sha256: str,
    final_model_sha256: str,
) -> dict[str, object]:
    panel = protocol["validation_panel"]
    if (
        value.get("format_version") != EVALUATION_FORMAT
        or value.get("authorization") != EVALUATION_AUTHORIZATION
        or value.get("evaluated_run_authorization") != freezer.AUTHORIZATION
        or value.get("test_access") != EVALUATION_TEST_ACCESS
        or value.get("gate_protocol_sha256") != gate_sha256
        or value.get("checkpoint", {}).get("receipt_file_sha256")
        != run_receipt_sha256
        or value.get("checkpoint", {}).get("trainer_state_file_sha256")
        != run_state_sha256
        or value.get("checkpoint", {}).get("final_model_sha256")
        != final_model_sha256
        or value.get("panel") != panel
        or value.get("world_size") != 4
    ):
        raise SegmentAuditError("Validation-panel evaluation identity differs")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != panel["record_count"]:
        raise SegmentAuditError("Validation-panel record inventory differs")
    indices = [int(item["index"]) for item in records]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise SegmentAuditError("Validation-panel membership is invalid")
    targets = sum(int(item["selected_targets"]) for item in records)
    correct = sum(int(item["correct"]) for item in records)
    loss_sum = sum(_finite(item["loss_sum"], "validation loss") for item in records)
    aggregate = value.get("aggregate", {})
    if (
        targets <= 0
        or aggregate.get("selected_targets") != targets
        or not math.isclose(
            _finite(aggregate.get("masked_accuracy"), "validation accuracy"),
            correct / targets,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            _finite(aggregate.get("masked_cross_entropy"), "validation CE"),
            loss_sum / targets,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise SegmentAuditError("Validation-panel aggregate does not replay")
    return {
        "masked_accuracy": aggregate["masked_accuracy"],
        "masked_cross_entropy": aggregate["masked_cross_entropy"],
        "record_count": len(records),
        "selected_targets": targets,
    }


def _previous_terminal(
    *,
    segment_index: int,
    previous_segment_root: Path | None,
    previous_audit_path: Path | None,
    gate_sha256: str,
) -> tuple[int, int, dict[str, Any] | None, dict[str, Any] | None]:
    if segment_index == 0:
        if previous_segment_root is not None or previous_audit_path is not None:
            raise SegmentAuditError("First segment cannot declare a predecessor")
        return 0, 0, None, None
    if previous_segment_root is None or previous_audit_path is None:
        raise SegmentAuditError("Resumed segment requires predecessor evidence")
    previous_audit, _ = _receipt(previous_audit_path)
    previous_receipt, previous_receipt_sha256 = _load(
        previous_segment_root / "run" / "receipt.json"
    )
    previous_state, previous_state_sha256 = _load(
        previous_segment_root / "run" / "trainer_state.json"
    )
    if (
        previous_audit.get("format_version") != FORMAT_VERSION
        or previous_audit.get("segment_index") != segment_index - 1
        or previous_audit.get("gate_protocol_file_sha256") != gate_sha256
        or previous_audit.get("authorization", {}).get("next_segment_submission")
        is not True
        or previous_audit.get("test_access") != EVALUATION_TEST_ACCESS
        or previous_audit.get("inputs", {}).get("segment_root")
        != str(previous_segment_root.resolve())
        or previous_audit.get("inputs", {}).get("run_receipt_file_sha256")
        != previous_receipt_sha256
        or previous_audit.get("inputs", {}).get("trainer_state_file_sha256")
        != previous_state_sha256
        or previous_audit.get("terminal")
        != {
            "checkpoint_model_sha256": previous_receipt.get(
                "checkpoint_model_sha256"
            ),
            "checkpoint_optimizer_sha256": previous_receipt.get(
                "checkpoint_optimizer_sha256"
            ),
            "checkpoint_rng_manifest_sha256": previous_receipt.get(
                "checkpoint_rng_manifest_sha256"
            ),
            "consumed_residue_tokens": previous_state.get(
                "consumed_residue_tokens"
            ),
            "next_update": previous_state.get("next_update"),
            "sample_cursor": previous_state.get("sample_cursor"),
        }
    ):
        raise SegmentAuditError("Previous segment audit does not authorize continuation")
    return (
        int(previous_state["consumed_residue_tokens"]),
        int(previous_state["next_update"]),
        previous_receipt,
        previous_state,
    )


def audit_segment(
    *,
    gate_protocol_path: Path,
    segment_root: Path,
    segment_index: int,
    previous_segment_root: Path | None = None,
    previous_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed receipt for one immutable segment and chain link."""
    if segment_index < 0:
        raise SegmentAuditError("Segment index must be non-negative")
    protocol, gate_sha256 = _load(gate_protocol_path)
    source_root = Path(__file__).resolve().parents[1]
    if (
        protocol.get("format_version") != freezer.FORMAT_VERSION
        or protocol.get("authorization") != freezer.AUTHORIZATION
        or protocol.get("test_access") != freezer.TEST_ACCESS
        or protocol.get("hardware_identity") != freezer.HARDWARE
        or protocol.get("execution", {}).get("optimization") != freezer.OPTIMIZATION
        or protocol.get("execution", {}).get("length_bucket_boundaries")
        != freezer.STATIC_LENGTH_BUCKET_BOUNDARIES
        or protocol.get("auditor_sha256") != artifacts._sha256(Path(__file__))
        or protocol.get("campaign_source", {}).get(
            "scripts/audit_antibody_four_gpu_pretraining_segment.py"
        )
        != artifacts._sha256(Path(__file__))
        or protocol.get("campaign_source") != freezer._campaign_source(source_root)
    ):
        raise SegmentAuditError("Pretraining campaign protocol differs")
    previous_consumed, previous_update, previous_receipt, previous_state = (
        _previous_terminal(
            segment_index=segment_index,
            previous_segment_root=previous_segment_root,
            previous_audit_path=previous_audit_path,
            gate_sha256=gate_sha256,
        )
    )
    receipt, receipt_sha256 = _load(segment_root / "run" / "receipt.json")
    state, state_sha256 = _load(segment_root / "run" / "trainer_state.json")
    expected_role = "stage1" if segment_index == 0 else "resumed"
    expected_run = protocol["runs"][expected_role]
    campaign = receipt.get("campaign", {})
    if (
        receipt.get("format_version") != RUN_FORMAT
        or receipt.get("authorization") != freezer.AUTHORIZATION
        or receipt.get("test_access") != freezer.TEST_ACCESS
        or receipt.get("gate_protocol_sha256") != gate_sha256
        or receipt.get("run_role") != expected_role
        or receipt.get("world_size") != 4
        or receipt.get("device") != freezer.HARDWARE["gpu_model"]
        or receipt.get("implementation") != protocol.get("implementation")
        or receipt.get("optimization") != freezer.OPTIMIZATION
        or receipt.get("checkpoint_model_sha256")
        != receipt.get("final_model_sha256")
        or receipt.get("checkpoint_optimizer_sha256") is None
        or receipt.get("checkpoint_rng_manifest_sha256") is None
        or not isinstance(receipt.get("membership_sha256"), str)
        or len(receipt.get("membership_sha256")) != 64
        or campaign.get("configured_maximum_steps") != expected_run["steps"]
        or campaign.get("executed_steps") != len(receipt.get("trace", []))
        or campaign.get("token_limit")
        != protocol["campaign"]["campaign_token_limit"]
        or state.get("gate_protocol_sha256") != gate_sha256
        or state.get("length_bucket_boundaries") != [
            int(freezer.STATIC_LENGTH_BUCKET_BOUNDARIES)
        ]
        or state.get("checkpoint_rng_manifest_sha256")
        != receipt.get("checkpoint_rng_manifest_sha256")
    ):
        raise SegmentAuditError("Segment run identity differs")
    if segment_index == 0:
        if receipt.get("resume_used") is not False:
            raise SegmentAuditError("First segment unexpectedly resumed")
    elif (
        receipt.get("resume_used") is not True
        or receipt.get("resume_payload_model_sha256")
        != previous_receipt.get("checkpoint_model_sha256")
        or receipt.get("resume_payload_optimizer_sha256")
        != previous_receipt.get("checkpoint_optimizer_sha256")
        or receipt.get("resume_loaded_model_sha256")
        != previous_receipt.get("final_model_sha256")
        or receipt.get("resume_loaded_optimizer_sha256")
        != previous_receipt.get("final_optimizer_sha256")
        or receipt.get("resume_rng_manifest_sha256")
        != previous_receipt.get("checkpoint_rng_manifest_sha256")
        or previous_state.get("gate_protocol_sha256") != gate_sha256
    ):
        raise SegmentAuditError("Resumed checkpoint chain differs")
    consumed, next_update = _validate_trace(
        receipt.get("trace"),
        previous_consumed=previous_consumed,
        previous_update=previous_update,
    )
    token_limit = int(protocol["campaign"]["campaign_token_limit"])
    complete = consumed >= token_limit
    executed_steps = len(receipt["trace"])
    if (
        state.get("consumed_residue_tokens") != consumed
        or state.get("next_update") != next_update
        or campaign.get("complete") is not complete
        or (complete and consumed - token_limit > int(
            protocol["campaign"]["maximum_final_overrun_residue_tokens"]
        ))
        or (not complete and executed_steps != expected_run["steps"])
    ):
        raise SegmentAuditError("Segment terminal coordinates differ")
    evaluation, evaluation_sha256 = _load(segment_root / "evaluation.json")
    validation = _validate_evaluation(
        evaluation,
        protocol=protocol,
        gate_sha256=gate_sha256,
        run_receipt_sha256=receipt_sha256,
        run_state_sha256=state_sha256,
        final_model_sha256=receipt["final_model_sha256"],
    )
    artifact_audit = artifacts._verify_artifacts(segment_root)
    if not artifact_audit["manifest_verified"] or artifact_audit["writable_paths"]:
        raise SegmentAuditError("Segment artifact tree is not immutable")
    return {
        "artifact_audit": artifact_audit,
        "authorization": {
            "campaign_complete": complete,
            "downstream_transfer": False,
            "next_segment_submission": not complete,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Immutable four-GPU OAS MLM pretraining segment and validation-panel "
            "checkpoint audit only; no downstream, official/frozen/external-test, "
            "biological, comparative-superiority, model-promotion, or SOTA claim"
        ),
        "format_version": FORMAT_VERSION,
        "gate_protocol_file_sha256": gate_sha256,
        "inputs": {
            "auditor_sha256": artifacts._sha256(Path(__file__)),
            "evaluation_file_sha256": evaluation_sha256,
            "run_receipt_file_sha256": receipt_sha256,
            "segment_root": str(segment_root.resolve()),
            "trainer_state_file_sha256": state_sha256,
        },
        "segment_index": segment_index,
        "test_access": EVALUATION_TEST_ACCESS,
        "terminal": {
            "checkpoint_model_sha256": receipt["checkpoint_model_sha256"],
            "checkpoint_optimizer_sha256": receipt["checkpoint_optimizer_sha256"],
            "checkpoint_rng_manifest_sha256": receipt[
                "checkpoint_rng_manifest_sha256"
            ],
            "consumed_residue_tokens": consumed,
            "next_update": next_update,
            "sample_cursor": state.get("sample_cursor"),
        },
        "validation": validation,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise SegmentAuditError("Segment audit output must be create-only")
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--segment-index", type=int, required=True)
    parser.add_argument("--previous-segment-root", type=Path)
    parser.add_argument("--previous-segment-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_segment(
        gate_protocol_path=args.gate_protocol,
        segment_root=args.segment_root,
        segment_index=args.segment_index,
        previous_segment_root=args.previous_segment_root,
        previous_audit_path=args.previous_segment_audit,
    )
    _publish(args.output, result)


if __name__ == "__main__":
    main()
