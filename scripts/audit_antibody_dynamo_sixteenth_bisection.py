#!/usr/bin/env python3
"""Independently audit a nineteen-arm Dynamo two-block bisection."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping

import audit_antibody_dynamo_region_bisection as parent


FORMAT_VERSION = "antibody_dynamo_sixteenth_bisection_result_v1"
PROTOCOL_FORMAT = "antibody_dynamo_sixteenth_bisection_protocol_v1"
SIDECAR_FORMAT = "antibody_dynamo_sixteenth_bisection_receipt_v1"
PREDECESSOR_PROTOCOL_FORMAT = "antibody_dynamo_eighth_bisection_protocol_v1"
PREDECESSOR_AUDIT_FORMAT = "antibody_dynamo_eighth_bisection_result_v1"
RUNS = (
    (("safe_eager_r1", "safe_eager"),)
    + tuple((f"dynamo_s{i}", f"dynamo_s{i}_gc_off") for i in range(1, 17))
    + (
        ("dynamo_full_control", "dynamo_full_control_gc_off"),
        ("safe_eager_r2", "safe_eager"),
    )
)
REFERENCE_RUNS = ("safe_eager_r1", "safe_eager_r2")
SIXTEENTH_RUNS = tuple(f"dynamo_s{i}" for i in range(1, 17))
ATTRIBUTION_RUNS = SIXTEENTH_RUNS + ("dynamo_full_control",)
PROMOTABLE_RUNS: tuple[str, ...] = ()
EXPECTED_SIXTEENTHS = [list(range(i * 2, i * 2 + 2)) for i in range(16)]


def _validate_predecessor(
    protocol: Mapping[str, Any],
    predecessor_protocol_path: Path,
    predecessor_audit_path: Path,
) -> dict[str, str]:
    predecessor_protocol, protocol_raw = parent.artifacts._load(
        predecessor_protocol_path,
        canonical=False,
    )
    predecessor_protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    predecessor_audit, predecessor_audit_file_sha256 = parent._self_hashed(
        predecessor_audit_path
    )
    registered = protocol.get("predecessor_bisection", {})
    if (
        predecessor_protocol.get("format_version")
        != PREDECESSOR_PROTOCOL_FORMAT
        or predecessor_audit.get("format_version") != PREDECESSOR_AUDIT_FORMAT
        or predecessor_audit.get("diagnostic_valid") is not True
        or predecessor_audit.get("candidate_found") is not False
        or predecessor_audit.get("selected_candidate_for_strict_confirmation")
        is not None
        or predecessor_audit.get("localization", {}).get("next_disposition")
        != "bisect_failed_eighths_into_sixteenths"
        or predecessor_audit.get("localization", {}).get(
            "next_registered_regions"
        )
        != EXPECTED_SIXTEENTHS
        or predecessor_audit.get("localization", {}).get(
            "full_control_reproduced"
        )
        is not True
        or predecessor_audit.get("exact_predecessor_replay", {}).get("passed")
        is not True
        or predecessor_audit.get("authorization") != protocol.get("authorization")
        or predecessor_audit.get("test_access") != protocol.get("test_access")
        or predecessor_audit.get("inputs", {}).get("protocol_file_sha256")
        != predecessor_protocol_sha256
        or registered.get("protocol_file_sha256")
        != predecessor_protocol_sha256
        or registered.get("audit_file_sha256")
        != predecessor_audit_file_sha256
        or registered.get("audit_receipt_sha256")
        != predecessor_audit.get("receipt_sha256")
        or protocol.get("inputs", {}).get("predecessor_protocol_file_sha256")
        != predecessor_protocol_sha256
        or protocol.get("inputs", {}).get("predecessor_audit_file_sha256")
        != predecessor_audit_file_sha256
    ):
        raise parent.DynamoRegionBisectionAuditError(
            "The eighth-bisection predecessor differs"
        )
    return {
        "predecessor_audit_file_sha256": predecessor_audit_file_sha256,
        "predecessor_audit_receipt_sha256": predecessor_audit["receipt_sha256"],
        "predecessor_protocol_file_sha256": predecessor_protocol_sha256,
    }


def _localization(result: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = result["comparisons"]
    failed = [
        name
        for name in SIXTEENTH_RUNS
        if not comparisons[name]["gates"]["evaluation_noninferiority"]
    ]
    full_failed = not comparisons["dynamo_full_control"]["gates"][
        "evaluation_noninferiority"
    ]
    next_regions: list[list[int]] = []
    if not full_failed:
        disposition = "full_control_did_not_reproduce_stop"
    elif not failed:
        disposition = "cross_sixteenth_interaction_required"
    else:
        disposition = "bisect_failed_sixteenths_into_blocks"
        for name in failed:
            index = SIXTEENTH_RUNS.index(name)
            start = index * 2
            next_regions.extend([[start], [start + 1]])
    return {
        "failed_sixteenth_runs": failed,
        "full_control_reproduced": full_failed,
        "next_disposition": disposition,
        "next_registered_regions": next_regions,
    }


@contextmanager
def _configured_parent() -> Iterator[None]:
    saved = {
        "ATTRIBUTION_RUNS": parent.ATTRIBUTION_RUNS,
        "FORMAT_VERSION": parent.FORMAT_VERSION,
        "PROMOTABLE_RUNS": parent.PROMOTABLE_RUNS,
        "PROTOCOL_FORMAT": parent.PROTOCOL_FORMAT,
        "REFERENCE_RUNS": parent.REFERENCE_RUNS,
        "RUNS": parent.RUNS,
        "SIDECAR_FORMAT": parent.SIDECAR_FORMAT,
        "_localization": parent._localization,
        "_validate_minifier_predecessor": parent._validate_minifier_predecessor,
        "__file__": parent.__file__,
    }
    parent.ATTRIBUTION_RUNS = ATTRIBUTION_RUNS
    parent.FORMAT_VERSION = FORMAT_VERSION
    parent.PROMOTABLE_RUNS = PROMOTABLE_RUNS
    parent.PROTOCOL_FORMAT = PROTOCOL_FORMAT
    parent.REFERENCE_RUNS = REFERENCE_RUNS
    parent.RUNS = RUNS
    parent.SIDECAR_FORMAT = SIDECAR_FORMAT
    parent._localization = _localization
    parent._validate_minifier_predecessor = _validate_predecessor
    parent.__file__ = str(Path(__file__))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def audit_sixteenth_bisection(
    *,
    run_root: Path,
    protocol_path: Path,
    predecessor_protocol_path: Path,
    predecessor_audit_path: Path,
) -> dict[str, Any]:
    """Audit exact replay, fixed two-block membership, and unchanged gates."""
    protocol, _ = parent.artifacts._load(protocol_path, canonical=False)
    bindings = {
        "parent_auditor_sha256": (
            Path(__file__).resolve().parent
            / "audit_antibody_dynamo_region_bisection.py"
        ),
        "parent_wrapper_sha256": (
            Path(__file__).resolve().parent
            / "run_antibody_dynamo_region_bisection.py"
        ),
        "predecessor_auditor_sha256": (
            Path(__file__).resolve().parent
            / "audit_antibody_dynamo_eighth_bisection.py"
        ),
        "predecessor_wrapper_sha256": (
            Path(__file__).resolve().parent
            / "run_antibody_dynamo_eighth_bisection.py"
        ),
    }
    if any(
        parent.artifacts._sha256(path) != protocol["inputs"][key]
        for key, path in bindings.items()
    ):
        raise parent.DynamoRegionBisectionAuditError(
            "Inherited bisection implementation differs"
        )
    with _configured_parent():
        result = parent.audit_bisection(
            run_root=run_root,
            protocol_path=protocol_path,
            minifier_protocol_path=predecessor_protocol_path,
            minifier_audit_path=predecessor_audit_path,
        )
    result["predecessor_bisection_evidence"] = result.pop(
        "failed_minifier_evidence"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predecessor-protocol", type=Path, required=True)
    parser.add_argument("--predecessor-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_sixteenth_bisection(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        predecessor_protocol_path=arguments.predecessor_protocol,
        predecessor_audit_path=arguments.predecessor_audit,
    )
    parent._publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
