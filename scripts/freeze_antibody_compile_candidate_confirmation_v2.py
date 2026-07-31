#!/usr/bin/env python3
"""Freeze strict confirmation from an audited compiler-backend attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


ATTRIBUTION_PROTOCOL_FORMAT = "antibody_compile_backend_attribution_protocol_v1"
ATTRIBUTION_AUDIT_FORMAT = "antibody_compile_backend_attribution_result_v1"
OUTPUT_FORMAT = "antibody_production_batch_efficiency_protocol_v3"
CANDIDATE = "inductor_no_cudagraphs"
MODE = "inductor_no_cudagraphs_gc_off"
MODE_FLAGS = {
    "compile_backend": "inductor",
    "compile_blocks": True,
    "disable_compiled_cudagraphs": True,
    "disable_python_gc": True,
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
REQUIRED_GATES = {
    "evaluation_noninferiority",
    "memory",
    "short_horizon_numerical",
    "step_stability",
    "throughput",
}


class ConfirmationV2FreezeError(RuntimeError):
    """Report invalid attribution evidence or a drifting source closure."""


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
        raise ConfirmationV2FreezeError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmationV2FreezeError(
            f"Input is not strict ASCII JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ConfirmationV2FreezeError(f"Input is not a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _self_hash(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ConfirmationV2FreezeError("Attribution audit self-hash differs")
    return observed


def _bind_sources(source_root: Path) -> dict[str, str]:
    bindings = {
        "confirmation_audit_launcher_sha256": (
            "slurm/antibody_compile_candidate_confirmation_v2_audit.sbatch"
        ),
        "confirmation_auditor_sha256": (
            "scripts/audit_antibody_compile_candidate_confirmation_v2.py"
        ),
        "confirmation_launcher_sha256": (
            "slurm/antibody_compile_candidate_confirmation_v2.sbatch"
        ),
        "confirmation_wrapper_sha256": (
            "scripts/run_antibody_compile_candidate_confirmation_v2.py"
        ),
        "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": (
            "scripts/run_antibody_pretraining_parity_extension.py"
        ),
    }
    result: dict[str, str] = {}
    for key, relative in bindings.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ConfirmationV2FreezeError(f"Source input is invalid: {relative}")
        result[key] = _sha256(path)
    return result


def freeze_confirmation_v2(
    *,
    source_root: Path,
    attribution_protocol_path: Path,
    attribution_protocol_file_sha256: str,
    attribution_audit_path: Path,
    attribution_audit_file_sha256: str,
) -> dict[str, Any]:
    """Build a fail-closed repeated crossover from one promotable backend."""
    protocol, observed_protocol_sha256 = _load(attribution_protocol_path)
    audit, observed_audit_sha256 = _load(attribution_audit_path)
    if observed_protocol_sha256 != attribution_protocol_file_sha256:
        raise ConfirmationV2FreezeError("Attribution protocol anchor differs")
    if observed_audit_sha256 != attribution_audit_file_sha256:
        raise ConfirmationV2FreezeError("Attribution audit anchor differs")
    audit_receipt_sha256 = _self_hash(audit)
    comparison = audit.get("comparisons", {}).get(CANDIDATE, {})
    if (
        protocol.get("format_version") != ATTRIBUTION_PROTOCOL_FORMAT
        or audit.get("format_version") != ATTRIBUTION_AUDIT_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or audit.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or audit.get("test_access") != TEST_ACCESS
        or audit.get("inputs", {}).get("protocol_file_sha256")
        != attribution_protocol_file_sha256
        or audit.get("diagnostic_valid") is not True
        or audit.get("candidate_found") is not True
        or audit.get("selected_candidate_for_strict_confirmation") != CANDIDATE
        or protocol.get("measurement", {}).get("promotable_runs") != [CANDIDATE]
        or protocol.get("modes", {}).get(MODE) != MODE_FLAGS
        or comparison.get("promotable_backend") is not True
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {})) != REQUIRED_GATES
        or not all(comparison.get("gates", {}).values())
    ):
        raise ConfirmationV2FreezeError(
            "Attribution evidence does not authorize strict confirmation"
        )
    bound_sources = _bind_sources(source_root)
    source_inputs = protocol["inputs"]
    if bound_sources["runner_sha256"] != source_inputs["base_runner_sha256"]:
        raise ConfirmationV2FreezeError("Base runner has drifted")
    fixed_inputs = {
        "communication_protocol_sha256": (
            "protocols/antibody_ddp_communication_optimization_v1.json"
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
        "training_protocol_sha256": "protocols/antibody_training_v1.json",
    }
    anchored_inputs = {
        key: _sha256(source_root / relative)
        for key, relative in fixed_inputs.items()
    }
    shared_keys = (
        "communication_protocol_sha256",
        "efficiency_protocol_sha256",
        "model_family_sha256",
        "parity_evaluation_indices_sha256",
        "parity_protocol_sha256",
        "parity_split_receipt_file_sha256",
        "parity_training_indices_sha256",
        "training_protocol_sha256",
    )
    if any(anchored_inputs[key] != source_inputs[key] for key in shared_keys):
        raise ConfirmationV2FreezeError("Attribution input closure has drifted")
    acceptance = protocol["acceptance"]
    inputs = {
        **anchored_inputs,
        **bound_sources,
        "attribution_audit_file_sha256": attribution_audit_file_sha256,
        "attribution_protocol_file_sha256": attribution_protocol_file_sha256,
        "dense_manifest_payload_sha256": source_inputs[
            "dense_manifest_payload_sha256"
        ],
    }
    return {
        "acceptance": {
            "bootstrap_confidence": acceptance["bootstrap_confidence"],
            "bootstrap_iterations": acceptance["bootstrap_iterations"],
            "bootstrap_seed": int(acceptance["bootstrap_seed"]) + 200,
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
            "independently audited compiler backend; no production-corpus "
            "training, scale, biological, downstream, or SOTA claim"
        ),
        "confirmation_selection": {
            "attribution_audit_file_sha256": attribution_audit_file_sha256,
            "attribution_audit_receipt_sha256": audit_receipt_sha256,
            "attribution_protocol_file_sha256": attribution_protocol_file_sha256,
            "selected_candidate": CANDIDATE,
            "selected_mode_flags": MODE_FLAGS,
        },
        "format_version": OUTPUT_FORMAT,
        "hardware": {
            "gpu_count": 8,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
        "inputs": inputs,
        "measurement": {
            "approximate_mfu_semantics": (
                "protocols/antibody_foundation_scale_v2.json"
            ),
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
                "compile_backend": "inductor",
                "compile_blocks": True,
                "compile_prewarm_buckets": True,
                "disable_compiled_cudagraphs": True,
                "fused_adamw": False,
                "pinned_staging": False,
            },
            "compute_dtype": "bfloat16",
            "ddp_gradient_reduction_dtype": "float32",
            "execution_controls": {
                "compile_backend": "inductor",
                "disable_compiled_cudagraphs": True,
                "disable_python_gc": True,
            },
            "experimental_bf16_ddp_hook_allowed": False,
            "reference": "safe_eager",
            "regional_compile_only": True,
        },
        "parent_protocol_file_sha256": attribution_protocol_file_sha256,
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--attribution-protocol", type=Path, required=True)
    parser.add_argument("--attribution-protocol-file-sha256", required=True)
    parser.add_argument("--attribution-audit", type=Path, required=True)
    parser.add_argument("--attribution-audit-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = freeze_confirmation_v2(
        source_root=arguments.source_root,
        attribution_protocol_path=arguments.attribution_protocol,
        attribution_protocol_file_sha256=(
            arguments.attribution_protocol_file_sha256
        ),
        attribution_audit_path=arguments.attribution_audit,
        attribution_audit_file_sha256=arguments.attribution_audit_file_sha256,
    )
    _publish(arguments.output, value)


if __name__ == "__main__":
    main()
