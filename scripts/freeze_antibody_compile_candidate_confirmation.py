#!/usr/bin/env python3
"""Freeze a strict confirmation protocol from an audited compile diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


DIAGNOSTIC_FORMAT = "antibody_compile_numerics_diagnostic_result_v2"
DIAGNOSTIC_PROTOCOL_FORMAT = "antibody_compile_numerics_diagnostic_protocol_v1"
AMENDMENT_FORMAT = "antibody_compile_numerics_audit_amendment_v2"
OUTPUT_FORMAT = "antibody_production_batch_efficiency_protocol_v3"
ALLOWED_CANDIDATES = {
    "compile_default",
    "compile_gc_off",
    "compile_accuracy",
    "compile_accuracy_gc_off",
}
AUTHORIZATION = {
    "configuration_promotion": False,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
AMENDMENT_CORRECTION = {
    "automatic_gc_semantics": (
        "gc.disable prevents automatic cyclic collection but does not forbid "
        "explicit gc.collect calls"
    ),
    "collection_counters": (
        "reported as telemetry and not interpreted as proof of automatic "
        "collection"
    ),
}


class ConfirmationFreezeError(RuntimeError):
    """Report invalid diagnostic evidence or a mutable source closure."""


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ConfirmationFreezeError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmationFreezeError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict):
        raise ConfirmationFreezeError(f"Input is not a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _self_hash(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ConfirmationFreezeError("Diagnostic audit self-hash differs")
    return observed


def freeze_confirmation(
    *,
    source_root: Path,
    diagnostic_protocol_path: Path,
    diagnostic_protocol_file_sha256: str,
    diagnostic_audit_path: Path,
    diagnostic_audit_file_sha256: str,
    audit_amendment_path: Path,
    audit_amendment_file_sha256: str,
) -> dict[str, Any]:
    """Build a fail-closed crossover protocol from one audited candidate."""
    diagnostic_protocol, observed_protocol_sha256 = _load(
        diagnostic_protocol_path
    )
    diagnostic_audit, observed_audit_sha256 = _load(diagnostic_audit_path)
    audit_amendment, observed_amendment_sha256 = _load(audit_amendment_path)
    if observed_protocol_sha256 != diagnostic_protocol_file_sha256:
        raise ConfirmationFreezeError("Diagnostic protocol external anchor differs")
    if observed_audit_sha256 != diagnostic_audit_file_sha256:
        raise ConfirmationFreezeError("Diagnostic audit external anchor differs")
    if observed_amendment_sha256 != audit_amendment_file_sha256:
        raise ConfirmationFreezeError("Audit amendment external anchor differs")
    diagnostic_receipt_sha256 = _self_hash(diagnostic_audit)
    candidate = diagnostic_audit.get("selected_candidate_for_strict_confirmation")
    comparison = diagnostic_audit.get("comparisons", {}).get(candidate, {})
    if (
        diagnostic_protocol.get("format_version") != DIAGNOSTIC_PROTOCOL_FORMAT
        or diagnostic_audit.get("format_version") != DIAGNOSTIC_FORMAT
        or audit_amendment.get("format_version") != AMENDMENT_FORMAT
        or audit_amendment.get("correction") != AMENDMENT_CORRECTION
        or audit_amendment.get("failed_parent_interpretation")
        != {
            "field": "python_gc.stats_delta[*].collections",
            "preserved": True,
            "reason": (
                "Explicit gc.collect calls remain legal while automatic "
                "cyclic garbage collection is disabled"
            ),
        }
        or audit_amendment.get("parent_protocol_file_sha256")
        != diagnostic_protocol_file_sha256
        or audit_amendment.get("authorization") != AUTHORIZATION
        or audit_amendment.get("test_access") != TEST_ACCESS
        or audit_amendment.get("amended_auditor_sha256")
        != diagnostic_audit.get("inputs", {}).get("auditor_sha256")
        or audit_amendment.get("parent_auditor_sha256")
        != diagnostic_audit.get("inputs", {}).get("parent_auditor_sha256")
        or diagnostic_audit.get("audit_amendment")
        != {
            "amendment_file_sha256": audit_amendment_file_sha256,
            "correction": AMENDMENT_CORRECTION,
            "failed_parent_interpretation_preserved": True,
        }
        or diagnostic_audit.get("inputs", {}).get("protocol_file_sha256")
        != diagnostic_protocol_file_sha256
        or diagnostic_audit.get("authorization") != AUTHORIZATION
        or diagnostic_audit.get("test_access") != TEST_ACCESS
        or diagnostic_protocol.get("authorization") != AUTHORIZATION
        or diagnostic_protocol.get("test_access") != TEST_ACCESS
        or diagnostic_audit.get("diagnostic_valid") is not True
        or diagnostic_audit.get("candidate_found") is not True
        or candidate not in ALLOWED_CANDIDATES
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {}))
        != {
            "evaluation_noninferiority",
            "memory",
            "short_horizon_numerical",
            "step_stability",
            "throughput",
        }
        or not all(comparison.get("gates", {}).values())
    ):
        raise ConfirmationFreezeError(
            "Diagnostic evidence does not authorize strict confirmation"
        )
    mode_flags = diagnostic_protocol.get("modes", {}).get(candidate)
    if mode_flags not in (
        {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": True,
            "disable_python_gc": False,
        },
        {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": True,
            "disable_python_gc": True,
        },
        {
            "accuracy_first_bf16_reduction": True,
            "compile_blocks": True,
            "disable_python_gc": False,
        },
        {
            "accuracy_first_bf16_reduction": True,
            "compile_blocks": True,
            "disable_python_gc": True,
        },
    ):
        raise ConfirmationFreezeError("Selected diagnostic mode flags differ")
    inputs = diagnostic_protocol["inputs"]
    source_bindings = {
        "confirmation_auditor_sha256": (
            "scripts/audit_antibody_compile_candidate_confirmation.py"
        ),
        "confirmation_wrapper_sha256": (
            "scripts/run_antibody_compile_candidate_confirmation.py"
        ),
        "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": (
            "scripts/run_antibody_pretraining_parity_extension.py"
        ),
    }
    bound_sources: dict[str, str] = {}
    for key, relative in source_bindings.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ConfirmationFreezeError(f"Source input is invalid: {relative}")
        bound_sources[key] = _sha256(path)
    expected_source_inputs = {
        "runner_sha256": inputs["base_runner_sha256"],
        "runner_wrapper_sha256": inputs["parity_wrapper_sha256"],
    }
    if any(bound_sources[key] != value for key, value in expected_source_inputs.items()):
        raise ConfirmationFreezeError("Diagnostic runner source closure has drifted")
    acceptance = diagnostic_protocol["acceptance"]
    return {
        "acceptance": {
            "bootstrap_confidence": acceptance["bootstrap_confidence"],
            "bootstrap_iterations": acceptance["bootstrap_iterations"],
            "bootstrap_seed": int(acceptance["bootstrap_seed"]) + 100,
            "compiled_relative_to_eager_each_round": acceptance[
                "minimum_candidate_relative_throughput"
            ],
            "compiled_relative_to_eager_pooled": acceptance[
                "minimum_candidate_relative_throughput"
            ],
            "maximum_masked_accuracy_absolute_decrease": acceptance[
                "maximum_masked_accuracy_absolute_decrease"
            ],
            "maximum_masked_cross_entropy_relative_increase": acceptance[
                "maximum_masked_cross_entropy_relative_increase"
            ],
            "maximum_memory_fraction": acceptance["maximum_memory_fraction"],
            "maximum_p95_over_p50_step_time": acceptance[
                "maximum_p95_over_p50_step_time"
            ],
            "maximum_replicate_throughput_relative_range": acceptance[
                "maximum_reference_throughput_relative_range"
            ],
            "maximum_short_horizon_gradient_norm_relative_difference": acceptance[
                "maximum_short_horizon_gradient_norm_relative_difference"
            ],
            "maximum_short_horizon_loss_absolute_difference": acceptance[
                "maximum_short_horizon_loss_absolute_difference"
            ],
            "minimum_approximate_6n_mfu": 0.35,
            "short_horizon_updates": acceptance["short_horizon_updates"],
            "warmup_updates_excluded": acceptance["warmup_updates_excluded"],
        },
        "authorization": AUTHORIZATION,
        "batch": {
            "batch_quantum_sequences": 512,
            "global_residue_token_target_per_update": 262_144,
            "micro_batch_per_gpu": 16,
            "updates_per_run": 600,
        },
        "claim_boundary": (
            "Training-excluded repeated eight-GPU confirmation of one "
            "independently audited compile candidate; no production-corpus "
            "training, scale, biological, downstream, or state-of-the-art claim"
        ),
        "confirmation_selection": {
            "audit_amendment_file_sha256": audit_amendment_file_sha256,
            "diagnostic_audit_file_sha256": diagnostic_audit_file_sha256,
            "diagnostic_audit_receipt_sha256": diagnostic_receipt_sha256,
            "diagnostic_protocol_file_sha256": diagnostic_protocol_file_sha256,
            "selected_candidate": candidate,
            "selected_mode_flags": mode_flags,
        },
        "format_version": OUTPUT_FORMAT,
        "hardware": {
            "gpu_count": 8,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "inputs": {
            "communication_protocol_sha256": inputs[
                "communication_protocol_sha256"
            ],
            "confirmation_auditor_sha256": bound_sources[
                "confirmation_auditor_sha256"
            ],
            "confirmation_wrapper_sha256": bound_sources[
                "confirmation_wrapper_sha256"
            ],
            "dense_manifest_payload_sha256": inputs[
                "dense_manifest_payload_sha256"
            ],
            "audit_amendment_file_sha256": audit_amendment_file_sha256,
            "diagnostic_audit_file_sha256": diagnostic_audit_file_sha256,
            "diagnostic_protocol_file_sha256": diagnostic_protocol_file_sha256,
            "efficiency_protocol_sha256": inputs["efficiency_protocol_sha256"],
            "mfu_protocol_sha256": inputs["mfu_protocol_sha256"],
            "model_family_sha256": inputs["model_family_sha256"],
            "parity_evaluation_indices_sha256": inputs[
                "parity_evaluation_indices_sha256"
            ],
            "parity_extension_protocol_sha256": inputs[
                "extension_protocol_sha256"
            ],
            "parity_protocol_sha256": inputs["parity_protocol_sha256"],
            "parity_split_receipt_file_sha256": inputs[
                "parity_split_receipt_file_sha256"
            ],
            "parity_training_indices_sha256": inputs[
                "parity_training_indices_sha256"
            ],
            "runner_sha256": bound_sources["runner_sha256"],
            "runner_wrapper_sha256": bound_sources["runner_wrapper_sha256"],
            "training_protocol_sha256": inputs["training_protocol_sha256"],
        },
        "measurement": {
            "approximate_mfu_semantics": "protocols/antibody_foundation_scale_v2.json",
            "crossover_run_order": [
                "safe_eager_r1",
                "fp32_compiled_r1",
                "fp32_compiled_r2",
                "safe_eager_r2",
            ],
            "membership_equal_within_and_across_rounds_required": True,
            "run_seed": 17,
            "same_allocation_required": True,
            "training_evaluation_panel": (
                "training_excluded_digest_panel_from_scientific_parity_v2"
            ),
        },
        "optimization": {
            "candidate": {
                "bucketed_padding": False,
                "compile_blocks": True,
                "compile_prewarm_buckets": True,
                "fused_adamw": False,
                "pinned_staging": False,
            },
            "compute_dtype": "bfloat16",
            "ddp_gradient_reduction_dtype": "float32",
            "execution_controls": {
                "accuracy_first_bf16_reduction": mode_flags[
                    "accuracy_first_bf16_reduction"
                ],
                "disable_python_gc": mode_flags["disable_python_gc"],
            },
            "experimental_bf16_ddp_hook_allowed": False,
            "reference": "safe_eager",
            "regional_compile_only": True,
        },
        "parent_protocol_file_sha256": diagnostic_protocol_file_sha256,
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ConfirmationFreezeError("Confirmation protocol must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol-file-sha256", required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--diagnostic-audit-file-sha256", required=True)
    parser.add_argument("--audit-amendment", type=Path, required=True)
    parser.add_argument("--audit-amendment-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = freeze_confirmation(
        source_root=arguments.source_root,
        diagnostic_protocol_path=arguments.diagnostic_protocol,
        diagnostic_protocol_file_sha256=(
            arguments.diagnostic_protocol_file_sha256
        ),
        diagnostic_audit_path=arguments.diagnostic_audit,
        diagnostic_audit_file_sha256=arguments.diagnostic_audit_file_sha256,
        audit_amendment_path=arguments.audit_amendment,
        audit_amendment_file_sha256=(
            arguments.audit_amendment_file_sha256
        ),
    )
    _publish(arguments.output, value)


if __name__ == "__main__":
    main()
