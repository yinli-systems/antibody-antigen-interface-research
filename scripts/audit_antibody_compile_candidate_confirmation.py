#!/usr/bin/env python3
"""Audit a repeated eight-GPU confirmation of one compile candidate."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_candidate_confirmation_result_v1"
SIDECAR_FORMAT = "antibody_compile_candidate_confirmation_receipt_v1"
DIAGNOSTIC_FORMAT = "antibody_compile_numerics_diagnostic_result_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("fp32_compiled_r1", "candidate"),
    ("fp32_compiled_r2", "candidate"),
    ("safe_eager_r2", "safe_eager"),
)
AUTHORIZATION = {
    "configuration_promotion": False,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}


def _expected_optimization(mode: str) -> dict[str, Any]:
    compiled = mode == "fp32_compiled"
    if mode not in {"safe_eager", "fp32_compiled"}:
        raise parent.ProductionBatchAuditError(f"Unexpected mode: {mode}")
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": False,
        "compile_blocks": compiled,
        "compile_prewarm_buckets": compiled,
        "ddp_batched_grad_copy": False,
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": "float32",
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise parent.ProductionBatchAuditError(f"Self-hash differs: {path}")
    return value


def _validate_diagnostic(
    *,
    protocol: Mapping[str, Any],
    diagnostic_protocol_path: Path,
    diagnostic_audit_path: Path,
) -> tuple[str, dict[str, bool], dict[str, str]]:
    diagnostic_protocol, diagnostic_protocol_raw = parent._load(
        diagnostic_protocol_path,
        canonical=False,
    )
    diagnostic_audit = _self_hashed(diagnostic_audit_path)
    protocol_sha256 = hashlib.sha256(diagnostic_protocol_raw).hexdigest()
    audit_file_sha256 = parent._sha256(diagnostic_audit_path)
    selected = diagnostic_audit.get("selected_candidate_for_strict_confirmation")
    selection = protocol.get("confirmation_selection", {})
    comparison = diagnostic_audit.get("comparisons", {}).get(selected, {})
    expected_gates = {
        "evaluation_noninferiority",
        "memory",
        "short_horizon_numerical",
        "step_stability",
        "throughput",
    }
    if (
        diagnostic_audit.get("format_version") != DIAGNOSTIC_FORMAT
        or diagnostic_audit.get("diagnostic_valid") is not True
        or diagnostic_audit.get("candidate_found") is not True
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {})) != expected_gates
        or not all(comparison.get("gates", {}).values())
        or selection.get("diagnostic_protocol_file_sha256") != protocol_sha256
        or selection.get("diagnostic_audit_file_sha256") != audit_file_sha256
        or selection.get("diagnostic_audit_receipt_sha256")
        != diagnostic_audit.get("receipt_sha256")
        or selection.get("selected_candidate") != selected
        or selection.get("selected_mode_flags")
        != diagnostic_protocol.get("modes", {}).get(selected)
        or protocol.get("inputs", {}).get("diagnostic_protocol_file_sha256")
        != protocol_sha256
        or protocol.get("inputs", {}).get("diagnostic_audit_file_sha256")
        != audit_file_sha256
        or protocol.get("optimization", {}).get("execution_controls")
        != {
            "accuracy_first_bf16_reduction": selection.get(
                "selected_mode_flags", {}
            ).get("accuracy_first_bf16_reduction"),
            "disable_python_gc": selection.get(
                "selected_mode_flags", {}
            ).get("disable_python_gc"),
        }
    ):
        raise parent.ProductionBatchAuditError(
            "Diagnostic candidate selection evidence differs"
        )
    return selected, selection["selected_mode_flags"], {
        "diagnostic_audit_file_sha256": audit_file_sha256,
        "diagnostic_audit_receipt_sha256": diagnostic_audit["receipt_sha256"],
        "diagnostic_protocol_file_sha256": protocol_sha256,
    }


def _sidecar(
    mode_root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    mode: str,
    flags: dict[str, bool],
) -> dict[str, Any]:
    path = mode_root / "run/compile_candidate_confirmation_receipt.json"
    value = _self_hashed(path)
    reduction = value.get("bf16_reduced_precision_reduction", {})
    python_gc = value.get("python_gc", {})
    if (
        value.get("format_version") != SIDECAR_FORMAT
        or value.get("mode") != mode
        or value.get("mode_flags") != flags
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("wrapper_sha256")
        != protocol["inputs"]["confirmation_wrapper_sha256"]
        or value.get("authorization") != AUTHORIZATION
        or value.get("test_access") != protocol["test_access"]
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
        or value.get("parity_extension_receipt_file_sha256")
        != parent._sha256(mode_root / "run/parity_extension_receipt.json")
        or reduction
        != {
            "after_restore": True,
            "before": True,
            "during_run": not flags["accuracy_first_bf16_reduction"],
        }
        or python_gc.get("enabled_after_restore") is not True
        or python_gc.get("enabled_during_run")
        is not (not flags["disable_python_gc"])
    ):
        raise parent.ProductionBatchAuditError(
            f"Confirmation sidecar differs: {mode_root.name}"
        )
    before = python_gc.get("stats_before")
    after = python_gc.get("stats_after")
    delta = python_gc.get("stats_delta")
    if (
        not isinstance(before, list)
        or not isinstance(after, list)
        or not isinstance(delta, list)
        or len(before) != len(after)
        or len(before) != len(delta)
        or any(
            right[key] - left[key] != difference[key]
            for left, right, difference in zip(before, after, delta)
            for key in ("collected", "collections", "uncollectable")
        )
    ):
        raise parent.ProductionBatchAuditError(
            f"Confirmation GC statistics differ: {mode_root.name}"
        )
    return {
        "automatic_gc_disabled_during_run": flags["disable_python_gc"],
        "collection_counters_include_explicit_calls": True,
        "file_sha256": parent._sha256(path),
        "observed_collection_counter_delta": [
            item["collections"] for item in delta
        ],
        "receipt_sha256": value["receipt_sha256"],
    }


def audit_confirmation(
    *,
    run_root: Path,
    protocol_path: Path,
    diagnostic_protocol_path: Path,
    diagnostic_audit_path: Path,
) -> dict[str, Any]:
    """Audit the full crossover and its immutable diagnostic selection."""
    protocol, protocol_raw = parent._load(protocol_path, canonical=False)
    observed_auditor = parent._sha256(Path(__file__))
    if (
        protocol.get("inputs", {}).get("confirmation_auditor_sha256")
        != observed_auditor
        or protocol.get("authorization") != AUTHORIZATION
    ):
        raise parent.ProductionBatchAuditError(
            "Confirmation auditor or authorization differs"
        )
    selected, candidate_flags, diagnostic_inputs = _validate_diagnostic(
        protocol=protocol,
        diagnostic_protocol_path=diagnostic_protocol_path,
        diagnostic_audit_path=diagnostic_audit_path,
    )
    original = parent._expected_optimization
    parent._expected_optimization = _expected_optimization
    try:
        result = parent.audit_production_batch(
            run_root=run_root,
            protocol_path=protocol_path,
        )
    finally:
        parent._expected_optimization = original
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    sidecars = {}
    for name, kind in RUNS:
        if kind == "safe_eager":
            mode = "safe_eager"
            flags = {
                "accuracy_first_bf16_reduction": False,
                "compile_blocks": False,
                "disable_python_gc": False,
            }
        else:
            mode = selected
            flags = candidate_flags
        sidecars[name] = _sidecar(
            run_root / name,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            mode=mode,
            flags=flags,
        )
    parent_auditor = result["inputs"].pop("auditor_sha256")
    result["candidate_confirmation_sidecars"] = sidecars
    result["confirmed_candidate"] = selected
    result["diagnostic_selection_inputs"] = diagnostic_inputs
    result["format_version"] = FORMAT_VERSION
    result["inputs"]["auditor_sha256"] = observed_auditor
    result["inputs"]["parent_auditor_sha256"] = parent_auditor
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_confirmation(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        diagnostic_protocol_path=arguments.diagnostic_protocol,
        diagnostic_audit_path=arguments.diagnostic_audit,
    )
    parent._publish(arguments.output, result)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
