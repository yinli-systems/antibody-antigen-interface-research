#!/usr/bin/env python3
"""Independently audit exact one-GPU recovery on the production OAS corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


RUN_FORMAT = "antibody_production_loader_recovery_run_v1"
GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
AUDIT_FORMAT = "antibody_production_loader_recovery_audit_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
TRACE_KEYS = (
    "consumed_residue_tokens",
    "gradient_norm",
    "learning_rate",
    "loss",
    "micro_steps",
    "residue_tokens",
    "selected_targets",
    "sequence_count",
    "update",
)


class RecoveryAuditError(RuntimeError):
    """Report an evidence, provenance, or exact-replay disagreement."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RecoveryAuditError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise RecoveryAuditError(f"Non-finite JSON constant: {value}")


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise RecoveryAuditError(f"Evidence path is invalid: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryAuditError(f"Evidence is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise RecoveryAuditError(f"Evidence is not a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _trace_equal(
    left: list[Mapping[str, object]],
    right: list[Mapping[str, object]],
) -> bool:
    return len(left) == len(right) and all(
        all(left_item.get(key) == right_item.get(key) for key in TRACE_KEYS)
        for left_item, right_item in zip(left, right)
    )


def _load_run(root: Path, role: str) -> dict[str, Any]:
    receipt, receipt_file_sha256 = _load(root / "receipt.json")
    state, state_file_sha256 = _load(root / "trainer_state.json")
    if (
        receipt.get("format_version") != RUN_FORMAT
        or receipt.get("run_role") != role
        or receipt.get("test_access") != TEST_ACCESS
        or receipt.get("authorization")
        != {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        }
        or receipt.get("world_size") != 1
        or state.get("world_size") != 1
    ):
        raise RecoveryAuditError(f"{role} run identity differs")
    return {
        "receipt": receipt,
        "receipt_file_sha256": receipt_file_sha256,
        "root": str(root),
        "state": state,
        "state_file_sha256": state_file_sha256,
    }


def audit_recovery(
    *,
    gate_protocol_path: Path,
    continuous_root: Path,
    stage1_root: Path,
    resumed_root: Path,
) -> dict[str, Any]:
    """Validate provenance and exact uninterrupted-versus-resumed equality."""
    protocol, protocol_file_sha256 = _load(gate_protocol_path)
    auditor_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (
        protocol.get("format_version") != GATE_FORMAT
        or protocol.get("auditor_sha256") != auditor_sha256
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("authorization")
        != {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        }
    ):
        raise RecoveryAuditError("Gate protocol identity differs")
    runs = {
        "continuous": _load_run(continuous_root, "continuous"),
        "stage1": _load_run(stage1_root, "stage1"),
        "resumed": _load_run(resumed_root, "resumed"),
    }
    receipts = {name: item["receipt"] for name, item in runs.items()}
    states = {name: item["state"] for name, item in runs.items()}
    common_receipt_fields = (
        "data_loader",
        "data_source",
        "efficiency_protocol_sha256",
        "gate_protocol_sha256",
        "implementation",
        "model_family_sha256",
        "optimization",
        "preset",
        "runtime",
        "training_protocol_sha256",
        "training_universe_count",
        "world_size",
    )
    reference = receipts["continuous"]
    if any(
        receipt.get(field) != reference.get(field)
        for receipt in receipts.values()
        for field in common_receipt_fields
    ):
        raise RecoveryAuditError("Run provenance differs across recovery roles")
    if reference.get("gate_protocol_sha256") != protocol_file_sha256:
        raise RecoveryAuditError("Run does not bind the supplied gate protocol")
    if reference.get("implementation") != protocol.get("implementation"):
        raise RecoveryAuditError("Run implementation differs from the protocol")
    if reference.get("data_loader") != protocol.get("execution", {}).get(
        "data_loader"
    ):
        raise RecoveryAuditError("Run data-loader contract differs from the protocol")
    expected_data = {
        key: value
        for key, value in protocol.get("inputs", {}).items()
        if key
        not in {
            "efficiency_protocol_file_sha256",
            "model_family_file_sha256",
            "training_protocol_file_sha256",
        }
    }
    if reference.get("data_source") != expected_data:
        raise RecoveryAuditError("Run data source differs from the protocol")
    continuous_trace = receipts["continuous"].get("trace")
    stage1_trace = receipts["stage1"].get("trace")
    resumed_trace = receipts["resumed"].get("trace")
    if not all(isinstance(trace, list) for trace in (
        continuous_trace,
        stage1_trace,
        resumed_trace,
    )):
        raise RecoveryAuditError("Recovery traces are absent")
    stage1_steps = protocol.get("runs", {}).get("stage1", {}).get("steps")
    if (
        isinstance(stage1_steps, bool)
        or not isinstance(stage1_steps, int)
        or stage1_steps <= 0
        or len(continuous_trace) != stage1_steps + len(resumed_trace)
        or not _trace_equal(continuous_trace[:stage1_steps], stage1_trace)
        or not _trace_equal(continuous_trace[stage1_steps:], resumed_trace)
    ):
        raise RecoveryAuditError("Uninterrupted and resumed scientific traces differ")
    stage1_receipt = receipts["stage1"]
    resumed_receipt = receipts["resumed"]
    if (
        receipts["continuous"].get("resume_used") is not False
        or stage1_receipt.get("resume_used") is not False
        or resumed_receipt.get("resume_used") is not True
        or not _is_sha256(stage1_receipt.get("checkpoint_model_sha256"))
        or not _is_sha256(stage1_receipt.get("checkpoint_optimizer_sha256"))
        or not _is_sha256(stage1_receipt.get("checkpoint_rng_manifest_sha256"))
        or stage1_receipt.get("checkpoint_model_sha256")
        != stage1_receipt.get("final_model_sha256")
        or not _is_sha256(stage1_receipt.get("final_optimizer_sha256"))
        or resumed_receipt.get("resume_payload_model_sha256")
        != stage1_receipt.get("checkpoint_model_sha256")
        or resumed_receipt.get("resume_payload_optimizer_sha256")
        != stage1_receipt.get("checkpoint_optimizer_sha256")
        or resumed_receipt.get("resume_loaded_model_sha256")
        != stage1_receipt.get("final_model_sha256")
        or resumed_receipt.get("resume_loaded_optimizer_sha256")
        != stage1_receipt.get("final_optimizer_sha256")
        or resumed_receipt.get("resume_rng_manifest_sha256")
        != stage1_receipt.get("checkpoint_rng_manifest_sha256")
    ):
        raise RecoveryAuditError("Checkpoint payload or applied state differs")
    if (
        receipts["continuous"].get("final_model_sha256")
        != resumed_receipt.get("final_model_sha256")
        or receipts["continuous"].get("final_optimizer_sha256")
        != resumed_receipt.get("final_optimizer_sha256")
    ):
        raise RecoveryAuditError("Final uninterrupted and resumed states differ")
    terminal_fields = (
        "consumed_residue_tokens",
        "next_update",
        "sample_cursor",
    )
    if any(
        states["continuous"].get(field) != states["resumed"].get(field)
        for field in terminal_fields
    ):
        raise RecoveryAuditError("Terminal trainer coordinates differ")
    if (
        receipts["continuous"].get("membership_sha256")
        == resumed_receipt.get("membership_sha256")
    ):
        raise RecoveryAuditError("Disjoint run segments share a membership digest")
    evidence = {
        name: {
            "receipt_file_sha256": item["receipt_file_sha256"],
            "root": item["root"],
            "state_file_sha256": item["state_file_sha256"],
        }
        for name, item in runs.items()
    }
    return {
        "auditor_sha256": auditor_sha256,
        "authorization": {
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Exact one-GPU OAS loader and checkpoint recovery only; no "
            "production-pretraining, scale, biological, or SOTA claim"
        ),
        "evidence": evidence,
        "exact_recovery": {
            "final_model_equal": True,
            "final_optimizer_equal": True,
            "scientific_trace_equal": True,
            "terminal_coordinates_equal": True,
        },
        "format_version": AUDIT_FORMAT,
        "gate_protocol_file_sha256": protocol_file_sha256,
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RecoveryAuditError("Audit output must be create-only")
    value = dict(payload)
    value["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_recovery(
        gate_protocol_path=args.gate_protocol,
        continuous_root=args.continuous,
        stage1_root=args.stage1,
        resumed_root=args.resumed,
    )
    _publish(args.output, result)


if __name__ == "__main__":
    main()
