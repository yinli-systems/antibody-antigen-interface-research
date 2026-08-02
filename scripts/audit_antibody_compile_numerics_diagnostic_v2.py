#!/usr/bin/env python3
"""Re-audit compile numerics with corrected Python-GC semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import audit_antibody_compile_numerics_diagnostic as v1
import audit_antibody_production_batch_efficiency as parent


FORMAT_VERSION = "antibody_compile_numerics_diagnostic_result_v2"
AMENDMENT_FORMAT = "antibody_compile_numerics_audit_amendment_v2"


class CompileNumericsAuditV2Error(RuntimeError):
    """Report invalid amendment evidence or corrected sidecar semantics."""


def _self_hashed(path: Path) -> dict[str, Any]:
    value, _ = parent._load(path)
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(parent._canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise CompileNumericsAuditV2Error(f"Receipt self-hash differs: {path}")
    return value


def _corrected_sidecar(
    mode_root: Path,
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    mode: str,
) -> dict[str, Any]:
    path = mode_root / "run/compile_numerics_diagnostic_receipt.json"
    value = _self_hashed(path)
    flags = protocol["modes"][mode]
    reduction = value.get("bf16_reduced_precision_reduction", {})
    python_gc = value.get("python_gc", {})
    if (
        value.get("format_version") != v1.SIDECAR_FORMAT
        or value.get("mode") != mode
        or value.get("mode_flags") != flags
        or value.get("protocol_sha256") != protocol_sha256
        or value.get("wrapper_sha256")
        != protocol["inputs"]["diagnostic_wrapper_sha256"]
        or value.get("authorization") != protocol["authorization"]
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
        or python_gc.get("enabled_before") is not True
        or python_gc.get("enabled_after_restore") is not True
        or python_gc.get("enabled_during_run")
        is not (not flags["disable_python_gc"])
    ):
        raise CompileNumericsAuditV2Error(f"Diagnostic sidecar differs: {mode}")
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
        raise CompileNumericsAuditV2Error(f"Python-GC statistics differ: {mode}")
    return {
        "automatic_gc_disabled_during_run": flags["disable_python_gc"],
        "collection_counters_include_explicit_calls": True,
        "file_sha256": parent._sha256(path),
        "observed_collection_counter_delta": [
            item["collections"] for item in delta
        ],
        "python_gc": python_gc,
        "receipt_sha256": value["receipt_sha256"],
    }


def _load_amendment(
    path: Path,
    *,
    parent_protocol_path: Path,
) -> tuple[dict[str, Any], str]:
    value, raw = parent._load(path, canonical=False)
    amendment_sha256 = hashlib.sha256(raw).hexdigest()
    expected = {
        "automatic_gc_semantics": (
            "gc.disable prevents automatic cyclic collection but does not "
            "forbid explicit gc.collect calls"
        ),
        "collection_counters": (
            "reported as telemetry and not interpreted as proof of automatic "
            "collection"
        ),
    }
    if (
        value.get("format_version") != AMENDMENT_FORMAT
        or value.get("correction") != expected
        or value.get("parent_protocol_file_sha256")
        != parent._sha256(parent_protocol_path)
        or value.get("parent_auditor_sha256") != parent._sha256(Path(v1.__file__))
        or value.get("amended_auditor_sha256") != parent._sha256(Path(__file__))
        or value.get("authorization")
        != {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
        }
        or value.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise CompileNumericsAuditV2Error("Audit amendment protocol differs")
    return value, amendment_sha256


def audit_compile_numerics_v2(
    *,
    run_root: Path,
    protocol_path: Path,
    amendment_path: Path,
) -> dict[str, Any]:
    """Replay the original audit while correcting one GC interpretation."""
    amendment, amendment_sha256 = _load_amendment(
        amendment_path,
        parent_protocol_path=protocol_path,
    )
    original = v1._diagnostic_sidecar
    v1._diagnostic_sidecar = _corrected_sidecar
    try:
        result = v1.audit_compile_numerics(
            run_root=run_root,
            protocol_path=protocol_path,
        )
    finally:
        v1._diagnostic_sidecar = original
    parent_auditor = result["inputs"].pop("auditor_sha256")
    result["audit_amendment"] = {
        "amendment_file_sha256": amendment_sha256,
        "correction": amendment["correction"],
        "failed_parent_interpretation_preserved": True,
    }
    result["format_version"] = FORMAT_VERSION
    result["inputs"]["auditor_sha256"] = parent._sha256(Path(__file__))
    result["inputs"]["parent_auditor_sha256"] = parent_auditor
    return result


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        parent._canonical_json_bytes(payload)
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(parent._canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit_compile_numerics_v2(
        run_root=arguments.run_root,
        protocol_path=arguments.protocol,
        amendment_path=arguments.amendment,
    )
    _publish(arguments.output, result)
    return 0 if result["diagnostic_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
