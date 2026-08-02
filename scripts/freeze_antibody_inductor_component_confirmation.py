#!/usr/bin/env python3
"""Freeze repeated confirmation of one audited Inductor component scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import freeze_antibody_compile_candidate_confirmation_v2 as shared
import run_antibody_inductor_component_localization as localization


LOCALIZATION_PROTOCOL_FORMAT = "antibody_inductor_component_localization_protocol_v1"
LOCALIZATION_AUDIT_FORMAT = "antibody_inductor_component_localization_result_v1"
OUTPUT_FORMAT = "antibody_production_batch_efficiency_protocol_v3"
PROTOCOL_VARIANT = "antibody_inductor_component_confirmation_protocol_v1"
AUTHORIZATION = localization.AUTHORIZATION
TEST_ACCESS = localization.TEST_ACCESS
CANDIDATES = {
    "inductor_attention_only": "inductor_attention_only_gc_off",
    "inductor_ffn_only": "inductor_ffn_only_gc_off",
}
REQUIRED_GATES = {
    "evaluation_noninferiority",
    "memory",
    "short_horizon_numerical",
    "step_stability",
    "throughput",
}
SOURCE_BINDINGS = {
    "communication_protocol_sha256": (
        "protocols/antibody_ddp_communication_optimization_v1.json"
    ),
    "confirmation_audit_launcher_sha256": (
        "slurm/antibody_inductor_component_confirmation_audit.sbatch"
    ),
    "confirmation_auditor_sha256": (
        "scripts/audit_antibody_inductor_component_confirmation.py"
    ),
    "confirmation_freezer_sha256": (
        "scripts/freeze_antibody_inductor_component_confirmation.py"
    ),
    "confirmation_launcher_sha256": (
        "slurm/antibody_inductor_component_confirmation.sbatch"
    ),
    "confirmation_wrapper_sha256": (
        "scripts/run_antibody_inductor_component_confirmation.py"
    ),
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "mfu_protocol_sha256": "protocols/antibody_foundation_scale_v2.json",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "parity_evaluation_indices_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy"
    ),
    "parity_extension_protocol_sha256": (
        "protocols/antibody_ddp_scientific_parity_extension_600_seed17_v1.json"
    ),
    "parity_protocol_sha256": (
        "protocols/antibody_ddp_scientific_parity_gate_v2.json"
    ),
    "parity_split_receipt_file_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
    ),
    "parity_training_indices_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy"
    ),
    "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
    "runner_wrapper_sha256": (
        "scripts/run_antibody_pretraining_parity_extension.py"
    ),
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
}


class ComponentConfirmationFreezeError(RuntimeError):
    """Report invalid localization evidence or source-closure drift."""


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


def _self_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ComponentConfirmationFreezeError("Localization audit self-hash differs")
    return observed


def _load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value, digest = shared._load(path)
    except shared.ConfirmationV2FreezeError as error:
        raise ComponentConfirmationFreezeError(str(error)) from error
    return value, digest


def _gate(comparison: Mapping[str, Any], name: str) -> bool:
    return comparison.get("gates", {}).get(name) is True


def _validate_selection(
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    protocol_sha256: str,
) -> tuple[str, str, dict[str, Any]]:
    candidate = audit.get("selected_component_candidate")
    mode = CANDIDATES.get(candidate)
    comparison = audit.get("comparisons", {}).get(candidate, {})
    if (
        protocol.get("format_version") != LOCALIZATION_PROTOCOL_FORMAT
        or audit.get("format_version") != LOCALIZATION_AUDIT_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or audit.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or audit.get("test_access") != TEST_ACCESS
        or audit.get("inputs", {}).get("protocol_file_sha256")
        != protocol_sha256
        or audit.get("inputs", {}).get("auditor_sha256")
        != protocol.get("inputs", {}).get("auditor_sha256")
        or audit.get("diagnostic_valid") is not True
        or audit.get("candidate_found") is not True
        or audit.get("selected_candidate_for_strict_confirmation") != candidate
        or candidate not in CANDIDATES
        or protocol.get("measurement", {}).get("promotable_runs")
        != list(CANDIDATES)
        or mode is None
        or protocol.get("modes", {}).get(mode) != localization.MODES[mode]
        or comparison.get("compile_scope")
        != localization.MODES[mode]["compile_scope"]
        or comparison.get("promotable_backend") is not True
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {})) != REQUIRED_GATES
        or not all(_gate(comparison, gate) for gate in REQUIRED_GATES)
    ):
        raise ComponentConfirmationFreezeError(
            "Localization evidence does not authorize strict confirmation"
        )
    return candidate, mode, dict(localization.MODES[mode])


def _bind_sources(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, relative in SOURCE_BINDINGS.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ComponentConfirmationFreezeError(
                f"Confirmation source input is invalid: {relative}"
            )
        result[key] = shared._sha256(path)
    return result


def freeze_confirmation(
    *,
    source_root: Path,
    localization_protocol_path: Path,
    localization_protocol_file_sha256: str,
    localization_audit_path: Path,
    localization_audit_file_sha256: str,
) -> dict[str, Any]:
    """Build a fail-closed repeated crossover from one component candidate."""
    protocol, observed_protocol_sha256 = _load(localization_protocol_path)
    audit, observed_audit_sha256 = _load(localization_audit_path)
    if observed_protocol_sha256 != localization_protocol_file_sha256:
        raise ComponentConfirmationFreezeError("Localization protocol anchor differs")
    if observed_audit_sha256 != localization_audit_file_sha256:
        raise ComponentConfirmationFreezeError("Localization audit anchor differs")
    audit_receipt_sha256 = _self_hash(audit)
    candidate, mode, flags = _validate_selection(
        protocol,
        audit,
        protocol_sha256=localization_protocol_file_sha256,
    )
    bound_sources = _bind_sources(source_root)
    shared_inputs = protocol["inputs"]
    for key in (
        "communication_protocol_sha256",
        "efficiency_protocol_sha256",
        "model_family_sha256",
        "parity_evaluation_indices_sha256",
        "parity_protocol_sha256",
        "parity_split_receipt_file_sha256",
        "parity_training_indices_sha256",
        "training_protocol_sha256",
    ):
        if bound_sources[key] != shared_inputs[key]:
            raise ComponentConfirmationFreezeError(
                f"Localization source closure has drifted: {key}"
            )
    acceptance = protocol["acceptance"]
    return {
        "acceptance": {
            "bootstrap_confidence": acceptance["bootstrap_confidence"],
            "bootstrap_iterations": acceptance["bootstrap_iterations"],
            "bootstrap_seed": int(acceptance["bootstrap_seed"]) + 2000,
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
            "warmup_updates_excluded": acceptance[
                "warmup_updates_excluded"
            ],
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
            "independently selected Inductor component scope; no production, "
            "scale, biological, downstream, or state-of-the-art claim"
        ),
        "component_confirmation_selection": {
            "localization_audit_file_sha256": localization_audit_file_sha256,
            "localization_audit_receipt_sha256": audit_receipt_sha256,
            "localization_protocol_file_sha256": (
                localization_protocol_file_sha256
            ),
            "selected_candidate": candidate,
            "selected_mode": mode,
            "selected_mode_flags": flags,
        },
        "format_version": OUTPUT_FORMAT,
        "hardware": {
            "gpu_count": 8,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "inputs": {
            **bound_sources,
            "dense_manifest_payload_sha256": shared_inputs[
                "dense_manifest_payload_sha256"
            ],
            "localization_audit_file_sha256": localization_audit_file_sha256,
            "localization_protocol_file_sha256": (
                localization_protocol_file_sha256
            ),
        },
        "measurement": {
            "approximate_mfu_semantics": (
                "protocols/antibody_foundation_scale_v2.json"
            ),
            "crossover_run_order": [
                "safe_eager_r1",
                "component_compiled_r1",
                "component_compiled_r2",
                "safe_eager_r2",
            ],
            "membership_equal_within_and_across_rounds_required": True,
            "run_seed": int(protocol["measurement"]["run_seed"]),
            "same_allocation_required": True,
            "training_evaluation_panel": protocol["measurement"][
                "training_evaluation_panel"
            ],
        },
        "optimization": {
            "candidate": flags,
            "execution_controls": {
                "compile_backend": "inductor",
                "compile_scope": flags["compile_scope"],
                "disable_compiled_cudagraphs": True,
                "disable_python_gc": True,
            },
            "safe_eager": localization.MODES["safe_eager"],
        },
        "protocol_variant": PROTOCOL_VARIANT,
        "selection": {
            "required_successor": (
                "exact real-OAS recovery only after independent audit passes"
            )
        },
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ComponentConfirmationFreezeError("Protocol must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--localization-protocol", type=Path, required=True)
    parser.add_argument("--localization-protocol-file-sha256", required=True)
    parser.add_argument("--localization-audit", type=Path, required=True)
    parser.add_argument("--localization-audit-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = freeze_confirmation(
        source_root=arguments.source_root,
        localization_protocol_path=arguments.localization_protocol,
        localization_protocol_file_sha256=(
            arguments.localization_protocol_file_sha256
        ),
        localization_audit_path=arguments.localization_audit,
        localization_audit_file_sha256=(
            arguments.localization_audit_file_sha256
        ),
    )
    _publish(arguments.output, value)


if __name__ == "__main__":
    main()
