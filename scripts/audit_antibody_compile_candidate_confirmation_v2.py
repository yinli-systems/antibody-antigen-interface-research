#!/usr/bin/env python3
"""Audit a repeated confirmation of an attributed compiler backend."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_candidate_confirmation_result_v2"
SIDECAR_FORMAT = "antibody_compile_candidate_confirmation_receipt_v2"
ATTRIBUTION_PROTOCOL_FORMAT = "antibody_compile_backend_attribution_protocol_v1"
ATTRIBUTION_AUDIT_FORMAT = "antibody_compile_backend_attribution_result_v1"
CANDIDATE = "inductor_no_cudagraphs"
MODE = "inductor_no_cudagraphs_gc_off"
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
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
SAFE_EAGER_FLAGS = {
    "compile_backend": None,
    "compile_blocks": False,
    "disable_compiled_cudagraphs": False,
    "disable_python_gc": False,
}
PROMOTABLE_FLAGS = {
    "compile_backend": "inductor",
    "compile_blocks": True,
    "disable_compiled_cudagraphs": True,
    "disable_python_gc": True,
}
REQUIRED_GATES = {
    "evaluation_noninferiority",
    "memory",
    "short_horizon_numerical",
    "step_stability",
    "throughput",
}


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise parent.ProductionBatchAuditError(f"Self-hash differs: {path}")
    return value


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
        "disable_compiled_cudagraphs": compiled,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }


def _validate_attribution(
    *,
    protocol: Mapping[str, Any],
    attribution_protocol_path: Path,
    attribution_audit_path: Path,
) -> dict[str, str]:
    attribution_protocol, attribution_protocol_raw = parent._load(
        attribution_protocol_path,
        canonical=False,
    )
    attribution_audit = _self_hashed(attribution_audit_path)
    protocol_sha256 = hashlib.sha256(attribution_protocol_raw).hexdigest()
    audit_file_sha256 = parent._sha256(attribution_audit_path)
    selection = protocol.get("confirmation_selection", {})
    comparison = attribution_audit.get("comparisons", {}).get(CANDIDATE, {})
    if (
        attribution_protocol.get("format_version") != ATTRIBUTION_PROTOCOL_FORMAT
        or attribution_audit.get("format_version") != ATTRIBUTION_AUDIT_FORMAT
        or attribution_protocol.get("authorization") != AUTHORIZATION
        or attribution_audit.get("authorization") != AUTHORIZATION
        or attribution_protocol.get("test_access") != TEST_ACCESS
        or attribution_audit.get("test_access") != TEST_ACCESS
        or attribution_audit.get("inputs", {}).get("protocol_file_sha256")
        != protocol_sha256
        or attribution_audit.get("diagnostic_valid") is not True
        or attribution_audit.get("candidate_found") is not True
        or attribution_audit.get("selected_candidate_for_strict_confirmation")
        != CANDIDATE
        or attribution_protocol.get("modes", {}).get(MODE) != PROMOTABLE_FLAGS
        or comparison.get("promotable_backend") is not True
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {})) != REQUIRED_GATES
        or not all(comparison.get("gates", {}).values())
        or selection
        != {
            "attribution_audit_file_sha256": audit_file_sha256,
            "attribution_audit_receipt_sha256": attribution_audit[
                "receipt_sha256"
            ],
            "attribution_protocol_file_sha256": protocol_sha256,
            "selected_candidate": CANDIDATE,
            "selected_mode_flags": PROMOTABLE_FLAGS,
        }
        or protocol.get("inputs", {}).get("attribution_audit_file_sha256")
        != audit_file_sha256
        or protocol.get("inputs", {}).get("attribution_protocol_file_sha256")
        != protocol_sha256
    ):
        raise parent.ProductionBatchAuditError(
            "Backend-attribution selection evidence differs"
        )
    return {
        "attribution_audit_file_sha256": audit_file_sha256,
        "attribution_audit_receipt_sha256": attribution_audit[
            "receipt_sha256"
        ],
        "attribution_protocol_file_sha256": protocol_sha256,
    }


def _sidecar(
    mode_root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    mode: str,
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    path = mode_root / "run/compile_candidate_confirmation_receipt_v2.json"
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
        or value.get("test_access") != TEST_ACCESS
        or value.get("base_receipt_file_sha256")
        != parent._sha256(mode_root / "run/receipt.json")
        or value.get("base_trainer_state_file_sha256")
        != parent._sha256(mode_root / "run/trainer_state.json")
        or value.get("parity_extension_receipt_file_sha256")
        != parent._sha256(mode_root / "run/parity_extension_receipt.json")
        or reduction
        != {"after_restore": True, "before": True, "during_run": True}
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
        "compile_backend": flags["compile_backend"],
        "compiled_cudagraphs_disabled": flags[
            "disable_compiled_cudagraphs"
        ],
        "file_sha256": parent._sha256(path),
        "receipt_sha256": value["receipt_sha256"],
    }


def audit_confirmation_v2(
    *,
    run_root: Path,
    protocol_path: Path,
    attribution_protocol_path: Path,
    attribution_audit_path: Path,
) -> dict[str, Any]:
    """Audit the crossover, selection evidence, and runtime sidecars."""
    protocol, protocol_raw = parent._load(protocol_path, canonical=False)
    observed_auditor = parent._sha256(Path(__file__))
    if (
        protocol.get("inputs", {}).get("confirmation_auditor_sha256")
        != observed_auditor
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise parent.ProductionBatchAuditError(
            "Confirmation auditor or authority differs"
        )
    selection_inputs = _validate_attribution(
        protocol=protocol,
        attribution_protocol_path=attribution_protocol_path,
        attribution_audit_path=attribution_audit_path,
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
    sidecars: dict[str, Any] = {}
    for name, kind in RUNS:
        if kind == "safe_eager":
            mode = "safe_eager"
            flags = SAFE_EAGER_FLAGS
        else:
            mode = CANDIDATE
            flags = PROMOTABLE_FLAGS
        sidecars[name] = _sidecar(
            run_root / name,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            mode=mode,
            flags=flags,
        )
    parent_auditor = result["inputs"].pop("auditor_sha256")
    result["attribution_selection_inputs"] = selection_inputs
    result["candidate_confirmation_sidecars"] = sidecars
    result["confirmed_candidate"] = CANDIDATE
    result["format_version"] = FORMAT_VERSION
    result["inputs"]["auditor_sha256"] = observed_auditor
    result["inputs"]["parent_auditor_sha256"] = parent_auditor
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attribution-protocol", type=Path, required=True)
    parser.add_argument("--attribution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_confirmation_v2(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        attribution_protocol_path=arguments.attribution_protocol,
        attribution_audit_path=arguments.attribution_audit,
    )
    parent._publish(arguments.output, result)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
