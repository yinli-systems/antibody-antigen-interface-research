#!/usr/bin/env python3
"""Independently audit an Inductor component-localization crossover."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_compile_backend_attribution as parent
import audit_antibody_production_batch_efficiency as artifacts


FORMAT_VERSION = "antibody_inductor_component_localization_result_v1"
PROTOCOL_FORMAT = "antibody_inductor_component_localization_protocol_v1"
SIDECAR_FORMAT = "antibody_inductor_component_localization_receipt_v1"
PARENT_PROTOCOL_FORMAT = "antibody_compile_backend_attribution_protocol_v1"
PARENT_AUDIT_FORMAT = "antibody_compile_backend_attribution_result_v1"
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("inductor_attention_only", "inductor_attention_only_gc_off"),
    ("inductor_ffn_only", "inductor_ffn_only_gc_off"),
    ("inductor_full_block", "inductor_full_block_gc_off"),
    ("safe_eager_r2", "safe_eager"),
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
ATTRIBUTION_RUNS = (
    "inductor_attention_only",
    "inductor_ffn_only",
    "inductor_full_block",
)
PROMOTABLE_RUNS = ("inductor_attention_only", "inductor_ffn_only")


class ComponentLocalizationAuditError(RuntimeError):
    """Report invalid predecessor, run, or immutable artifact evidence."""


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = artifacts._load(path, canonical=False)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(artifacts._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ComponentLocalizationAuditError(f"Receipt self-hash differs: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _gate(comparison: Mapping[str, Any], name: str) -> bool:
    return comparison.get("gates", {}).get(name) is True


def _validate_parent_attribution(
    *,
    protocol: Mapping[str, Any],
    attribution_protocol_path: Path,
    attribution_audit_path: Path,
) -> dict[str, str]:
    parent_protocol, parent_protocol_raw = artifacts._load(
        attribution_protocol_path,
        canonical=False,
    )
    parent_protocol_sha256 = hashlib.sha256(parent_protocol_raw).hexdigest()
    parent_audit, parent_audit_file_sha256 = _self_hashed(
        attribution_audit_path
    )
    parent_record = protocol.get("parent_attribution", {})
    comparisons = parent_audit.get("comparisons", {})
    dynamo = comparisons.get("dynamo_eager", {})
    aot = comparisons.get("aot_eager", {})
    inductor = comparisons.get("inductor_no_cudagraphs", {})
    stable_gates = (
        "evaluation_noninferiority",
        "memory",
        "short_horizon_numerical",
        "step_stability",
    )
    if (
        parent_protocol.get("format_version") != PARENT_PROTOCOL_FORMAT
        or parent_audit.get("format_version") != PARENT_AUDIT_FORMAT
        or parent_protocol.get("authorization") != protocol.get("authorization")
        or parent_audit.get("authorization") != protocol.get("authorization")
        or parent_protocol.get("test_access") != protocol.get("test_access")
        or parent_audit.get("test_access") != protocol.get("test_access")
        or parent_audit.get("inputs", {}).get("protocol_file_sha256")
        != parent_protocol_sha256
        or parent_audit.get("inputs", {}).get("auditor_sha256")
        != parent_protocol.get("inputs", {}).get("auditor_sha256")
        or parent_protocol.get("inputs", {}).get("auditor_sha256")
        != artifacts._sha256(Path(parent.__file__))
        or protocol.get("inputs", {}).get("attribution_protocol_file_sha256")
        != parent_protocol_sha256
        or protocol.get("inputs", {}).get("attribution_audit_file_sha256")
        != parent_audit_file_sha256
        or parent_record.get("protocol_file_sha256") != parent_protocol_sha256
        or parent_record.get("audit_file_sha256") != parent_audit_file_sha256
        or parent_record.get("audit_receipt_sha256")
        != parent_audit.get("receipt_sha256")
        or parent_record.get("earliest_failing_layer") != "inductor"
        or parent_audit.get("diagnostic_valid") is not True
        or parent_audit.get("candidate_found") is not False
        or parent_audit.get("selected_candidate_for_strict_confirmation") is not None
        or not all(_gate(dynamo, gate) for gate in stable_gates)
        or not all(_gate(aot, gate) for gate in stable_gates)
        or _gate(inductor, "evaluation_noninferiority")
        or not all(
            _gate(inductor, gate)
            for gate in (
                "memory",
                "short_horizon_numerical",
                "step_stability",
                "throughput",
            )
        )
        or dynamo.get("backend_layer") != "eager"
        or aot.get("backend_layer") != "aot_eager"
        or inductor.get("backend_layer") != "inductor"
    ):
        raise ComponentLocalizationAuditError(
            "Predecessor does not prove an Inductor-only quality failure"
        )
    return {
        "attribution_audit_file_sha256": parent_audit_file_sha256,
        "attribution_audit_receipt_sha256": parent_audit["receipt_sha256"],
        "attribution_protocol_file_sha256": parent_protocol_sha256,
    }


def audit_localization(
    *,
    run_root: Path,
    protocol_path: Path,
    attribution_protocol_path: Path,
    attribution_audit_path: Path,
) -> dict[str, Any]:
    """Replay predecessor authority and audit all component arms."""
    protocol, protocol_raw = artifacts._load(protocol_path, canonical=False)
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("measurement", {}).get("run_order")
        != [name for name, _ in RUNS]
        or protocol.get("measurement", {}).get("promotable_runs")
        != list(PROMOTABLE_RUNS)
        or artifacts._sha256(run_root / "protocol.json") != protocol_sha256
        or protocol.get("inputs", {}).get("auditor_sha256")
        != artifacts._sha256(Path(__file__))
    ):
        raise ComponentLocalizationAuditError("Localization protocol differs")
    predecessor = _validate_parent_attribution(
        protocol=protocol,
        attribution_protocol_path=attribution_protocol_path,
        attribution_audit_path=attribution_audit_path,
    )
    saved = {
        "ATTRIBUTION_RUNS": parent.ATTRIBUTION_RUNS,
        "FORMAT_VERSION": parent.FORMAT_VERSION,
        "PROMOTABLE_RUNS": parent.PROMOTABLE_RUNS,
        "PROTOCOL_FORMAT": parent.PROTOCOL_FORMAT,
        "REFERENCE_RUNS": parent.REFERENCE_RUNS,
        "RUNS": parent.RUNS,
        "SIDECAR_FORMAT": parent.SIDECAR_FORMAT,
        "__file__": parent.__file__,
    }
    parent.ATTRIBUTION_RUNS = ATTRIBUTION_RUNS
    parent.FORMAT_VERSION = FORMAT_VERSION
    parent.PROMOTABLE_RUNS = PROMOTABLE_RUNS
    parent.PROTOCOL_FORMAT = PROTOCOL_FORMAT
    parent.REFERENCE_RUNS = REFERENCE_RUNS
    parent.RUNS = RUNS
    parent.SIDECAR_FORMAT = SIDECAR_FORMAT
    parent.__file__ = str(Path(__file__))
    try:
        result = parent.audit_backend_attribution(
            run_root=run_root,
            protocol_path=protocol_path,
        )
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)
    for run_name in ATTRIBUTION_RUNS:
        mode = dict(RUNS)[run_name]
        scope = protocol["modes"][mode]["compile_scope"]
        result["comparisons"][run_name]["compile_scope"] = scope
        result["mode_sidecars"][run_name]["compile_scope"] = scope
    result["claim_boundary"] = protocol["claim_boundary"]
    result["parent_attribution_evidence"] = predecessor
    result["selected_component_candidate"] = result[
        "selected_candidate_for_strict_confirmation"
    ]
    return result


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        artifacts._canonical_json_bytes(payload)
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(artifacts._canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--attribution-protocol", type=Path, required=True)
    parser.add_argument("--attribution-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_localization(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        attribution_protocol_path=arguments.attribution_protocol,
        attribution_audit_path=arguments.attribution_audit,
    )
    _publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
