#!/usr/bin/env python3
"""Audit the repeated eight-GPU compile-only confirmation experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_only_confirmation_result_v1"
CANDIDATE = {
    "bucketed_padding": False,
    "compile_blocks": True,
    "compile_prewarm_buckets": True,
    "fused_adamw": False,
    "pinned_staging": False,
}


def _expected_optimization(mode: str) -> dict[str, Any]:
    """Return the frozen eager or regional-compile-only execution contract."""
    compiled = mode == "fp32_compiled"
    if mode not in {"safe_eager", "fp32_compiled"}:
        raise parent.ProductionBatchAuditError(f"Unexpected mode: {mode}")
    return {
        "audit_gradient_rank_parity": True,
        "bucketed_padding": compiled and CANDIDATE["bucketed_padding"],
        "compile_blocks": compiled and CANDIDATE["compile_blocks"],
        "compile_prewarm_buckets": (
            compiled and CANDIDATE["compile_prewarm_buckets"]
        ),
        "ddp_batched_grad_copy": False,
        "ddp_bucket_cap_mb": 25,
        "ddp_communication_dtype": "float32",
        "ddp_static_graph": False,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "disable_ddp_gradient_as_bucket_view": False,
        "fsdp_output_dtype": "bfloat16",
        "fused_adamw": compiled and CANDIDATE["fused_adamw"],
        "length_bucketing": False,
        "pinned_staging": compiled and CANDIDATE["pinned_staging"],
    }


def audit_compile_only(
    *,
    run_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Delegate shared replay logic while binding compile-only semantics."""
    protocol, _ = parent._load(protocol_path, canonical=False)
    if protocol.get("optimization", {}).get("candidate") != CANDIDATE:
        raise parent.ProductionBatchAuditError(
            "Compile-only candidate differs from the auditor"
        )
    expected_auditor = protocol.get("inputs", {}).get(
        "confirmation_auditor_sha256"
    )
    observed_auditor = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if expected_auditor != observed_auditor:
        raise parent.ProductionBatchAuditError(
            "Compile-only auditor differs from the protocol"
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
    parent_auditor = result["inputs"].pop("auditor_sha256")
    result["inputs"]["parent_auditor_sha256"] = parent_auditor
    result["inputs"]["auditor_sha256"] = observed_auditor
    result["format_version"] = FORMAT_VERSION
    result["confirmed_candidate"] = "compile_only"
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    result = audit_compile_only(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
    )
    parent._publish(arguments.output, result)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
