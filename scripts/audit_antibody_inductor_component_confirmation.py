#!/usr/bin/env python3
"""Independently audit repeated confirmation of one Inductor component."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_compile_candidate_confirmation_v2 as confirmation
import audit_antibody_inductor_component_localization as localization_auditor
import audit_antibody_production_batch_efficiency as parent
import run_antibody_inductor_component_localization as localization


FORMAT_VERSION = "antibody_inductor_component_confirmation_result_v1"
SIDECAR_FORMAT = "antibody_inductor_component_confirmation_receipt_v1"
PROTOCOL_VARIANT = "antibody_inductor_component_confirmation_protocol_v1"
LOCALIZATION_PROTOCOL_FORMAT = "antibody_inductor_component_localization_protocol_v1"
LOCALIZATION_AUDIT_FORMAT = "antibody_inductor_component_localization_result_v1"
AUTHORIZATION = localization.AUTHORIZATION
TEST_ACCESS = localization.TEST_ACCESS
CANDIDATES = {
    "inductor_attention_only": "inductor_attention_only_gc_off",
    "inductor_ffn_only": "inductor_ffn_only_gc_off",
}
CONFIRMATION_RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("component_compiled_r1", "candidate"),
    ("component_compiled_r2", "candidate"),
    ("safe_eager_r2", "safe_eager"),
)
PARENT_RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("component_compiled_r1", "fp32_compiled"),
    ("component_compiled_r2", "fp32_compiled"),
    ("safe_eager_r2", "safe_eager"),
)
REQUIRED_GATES = {
    "evaluation_noninferiority",
    "memory",
    "short_horizon_numerical",
    "step_stability",
    "throughput",
}


class ComponentConfirmationAuditError(RuntimeError):
    """Report invalid selection, run, or immutable-artifact evidence."""


def _self_hashed(path: Path) -> tuple[dict[str, Any], str]:
    value, raw = parent._load(path, canonical=False)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ComponentConfirmationAuditError(f"Receipt self-hash differs: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_localization(
    *,
    protocol: Mapping[str, Any],
    localization_protocol_path: Path,
    localization_audit_path: Path,
) -> tuple[dict[str, str], str, dict[str, Any]]:
    localization_protocol, localization_protocol_raw = parent._load(
        localization_protocol_path,
        canonical=False,
    )
    localization_audit, localization_audit_file_sha256 = _self_hashed(
        localization_audit_path
    )
    localization_protocol_sha256 = hashlib.sha256(
        localization_protocol_raw
    ).hexdigest()
    selection = protocol.get("component_confirmation_selection", {})
    candidate = selection.get("selected_candidate")
    mode = CANDIDATES.get(candidate)
    flags = localization.MODES.get(mode)
    comparison = localization_audit.get("comparisons", {}).get(candidate, {})
    expected_selection = {
        "localization_audit_file_sha256": localization_audit_file_sha256,
        "localization_audit_receipt_sha256": localization_audit[
            "receipt_sha256"
        ],
        "localization_protocol_file_sha256": localization_protocol_sha256,
        "selected_candidate": candidate,
        "selected_mode": mode,
        "selected_mode_flags": flags,
    }
    if (
        localization_protocol.get("format_version")
        != LOCALIZATION_PROTOCOL_FORMAT
        or localization_audit.get("format_version")
        != LOCALIZATION_AUDIT_FORMAT
        or localization_protocol.get("authorization") != AUTHORIZATION
        or localization_audit.get("authorization") != AUTHORIZATION
        or localization_protocol.get("test_access") != TEST_ACCESS
        or localization_audit.get("test_access") != TEST_ACCESS
        or localization_audit.get("inputs", {}).get("protocol_file_sha256")
        != localization_protocol_sha256
        or localization_audit.get("inputs", {}).get("auditor_sha256")
        != localization_protocol.get("inputs", {}).get("auditor_sha256")
        or localization_protocol.get("inputs", {}).get("auditor_sha256")
        != parent._sha256(Path(localization_auditor.__file__))
        or localization_audit.get("diagnostic_valid") is not True
        or localization_audit.get("candidate_found") is not True
        or candidate not in CANDIDATES
        or mode is None
        or flags is None
        or localization_audit.get("selected_component_candidate") != candidate
        or localization_audit.get("selected_candidate_for_strict_confirmation")
        != candidate
        or localization_protocol.get("modes", {}).get(mode) != flags
        or comparison.get("compile_scope") != flags["compile_scope"]
        or comparison.get("promotable_backend") is not True
        or comparison.get("eligible_for_strict_confirmation") is not True
        or set(comparison.get("gates", {})) != REQUIRED_GATES
        or not all(comparison.get("gates", {}).values())
        or selection != expected_selection
        or protocol.get("inputs", {}).get("localization_protocol_file_sha256")
        != localization_protocol_sha256
        or protocol.get("inputs", {}).get("localization_audit_file_sha256")
        != localization_audit_file_sha256
    ):
        raise ComponentConfirmationAuditError(
            "Component-localization selection evidence differs"
        )
    evidence = {
        "localization_audit_file_sha256": localization_audit_file_sha256,
        "localization_audit_receipt_sha256": localization_audit[
            "receipt_sha256"
        ],
        "localization_protocol_file_sha256": localization_protocol_sha256,
    }
    return evidence, candidate, dict(flags)


def audit_confirmation(
    *,
    run_root: Path,
    protocol_path: Path,
    localization_protocol_path: Path,
    localization_audit_path: Path,
) -> dict[str, Any]:
    """Replay component selection and audit the repeated crossover."""
    protocol, _ = parent._load(protocol_path, canonical=False)
    observed_auditor = parent._sha256(Path(__file__))
    if (
        protocol.get("format_version")
        != "antibody_production_batch_efficiency_protocol_v3"
        or protocol.get("protocol_variant") != PROTOCOL_VARIANT
        or protocol.get("inputs", {}).get("confirmation_auditor_sha256")
        != observed_auditor
        or protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise ComponentConfirmationAuditError(
            "Component confirmation protocol or authority differs"
        )
    selection_evidence, candidate, flags = _validate_localization(
        protocol=protocol,
        localization_protocol_path=localization_protocol_path,
        localization_audit_path=localization_audit_path,
    )

    def validate_selection(**_: Any) -> dict[str, str]:
        return selection_evidence

    saved_confirmation = {
        "CANDIDATE": confirmation.CANDIDATE,
        "FORMAT_VERSION": confirmation.FORMAT_VERSION,
        "PROMOTABLE_FLAGS": confirmation.PROMOTABLE_FLAGS,
        "RUNS": confirmation.RUNS,
        "SIDECAR_FORMAT": confirmation.SIDECAR_FORMAT,
        "_validate_attribution": confirmation._validate_attribution,
        "__file__": confirmation.__file__,
    }
    saved_parent_runs = parent.RUNS
    confirmation.CANDIDATE = candidate
    confirmation.FORMAT_VERSION = FORMAT_VERSION
    confirmation.PROMOTABLE_FLAGS = flags
    confirmation.RUNS = CONFIRMATION_RUNS
    confirmation.SIDECAR_FORMAT = SIDECAR_FORMAT
    confirmation._validate_attribution = validate_selection
    confirmation.__file__ = str(Path(__file__))
    parent.RUNS = PARENT_RUNS
    try:
        result = confirmation.audit_confirmation_v2(
            run_root=run_root,
            protocol_path=protocol_path,
            attribution_protocol_path=localization_protocol_path,
            attribution_audit_path=localization_audit_path,
        )
    finally:
        parent.RUNS = saved_parent_runs
        for name, value in saved_confirmation.items():
            setattr(confirmation, name, value)
    result["component_confirmation_selection_inputs"] = result.pop(
        "attribution_selection_inputs"
    )
    result["confirmed_component_candidate"] = result.pop(
        "confirmed_candidate"
    )
    for name in ("component_compiled_r1", "component_compiled_r2"):
        result["candidate_confirmation_sidecars"][name]["compile_scope"] = flags[
            "compile_scope"
        ]
    result["claim_boundary"] = protocol["claim_boundary"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--localization-protocol", type=Path, required=True)
    parser.add_argument("--localization-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_confirmation(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        localization_protocol_path=arguments.localization_protocol,
        localization_audit_path=arguments.localization_audit,
    )
    parent._publish(arguments.output, result)
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
