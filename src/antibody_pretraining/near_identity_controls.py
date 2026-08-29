"""Registered synthetic sensitivity controls for the MMseqs2 GPU audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .near_identity_overlap import NearIdentityOverlapError, normalize_fraction


CONTROL_FORMAT = "oas_near_identity_mmseqs_sensitivity_controls_v1"
CONTROL_NAMES = (
    "exact_match",
    "single_substitution_above_threshold",
    "internal_indel_above_threshold",
    "terminal_indel_above_threshold",
)


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


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _rotate(value: str, positions: int) -> str:
    return value[positions:] + value[:positions]


def build_sensitivity_controls(output_root: Path) -> dict[str, Any]:
    """Create deterministic exact, substitution, and indel query-target pairs."""
    output_root.mkdir(parents=True, exist_ok=False)
    base = "ACDEFGHIKLMNPQRSTVWY" * 5
    targets = {
        "ctrl_target_exact": base,
        "ctrl_target_substitution": _rotate(base, 3),
        "ctrl_target_internal_indel": _rotate(base, 7),
        "ctrl_target_terminal_indel": _rotate(base, 11),
    }
    substitution_target = targets["ctrl_target_substitution"]
    replacement = "A" if substitution_target[50] != "A" else "C"
    queries = {
        "ctrl_query_exact": targets["ctrl_target_exact"],
        "ctrl_query_substitution": (
            substitution_target[:50] + replacement + substitution_target[51:]
        ),
        "ctrl_query_internal_indel": (
            targets["ctrl_target_internal_indel"][:50]
            + "A"
            + targets["ctrl_target_internal_indel"][50:]
        ),
        "ctrl_query_terminal_indel": targets["ctrl_target_terminal_indel"][1:],
    }
    expected = [
        {
            "control": "exact_match",
            "query": "ctrl_query_exact",
            "target": "ctrl_target_exact",
        },
        {
            "control": "single_substitution_above_threshold",
            "query": "ctrl_query_substitution",
            "target": "ctrl_target_substitution",
        },
        {
            "control": "internal_indel_above_threshold",
            "query": "ctrl_query_internal_indel",
            "target": "ctrl_target_internal_indel",
        },
        {
            "control": "terminal_indel_above_threshold",
            "query": "ctrl_query_terminal_indel",
            "target": "ctrl_target_terminal_indel",
        },
    ]
    query_path = output_root / "queries.fasta"
    target_path = output_root / "targets.fasta"
    expected_path = output_root / "expected.json"
    _exclusive_write(
        query_path,
        "".join(f">{key}\n{value}\n" for key, value in queries.items()).encode("ascii"),
    )
    _exclusive_write(
        target_path,
        "".join(f">{key}\n{value}\n" for key, value in targets.items()).encode("ascii"),
    )
    manifest = {
        "format_version": CONTROL_FORMAT,
        "identity_threshold": 0.9,
        "query_coverage_threshold": 0.9,
        "target_coverage_threshold": 0.9,
        "expected_pairs": expected,
        "query_fasta_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "target_fasta_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "test_access": {
            "development_data_accessed": False,
            "official_frozen_external_test_accessed": False,
        },
    }
    manifest["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    _exclusive_write(expected_path, _canonical_json_bytes(manifest))
    return manifest


def audit_sensitivity_hits(
    lines: Iterable[str], *, expected_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Require every registered pair to pass the inclusive 90/90/90 gate."""
    unsigned = dict(expected_manifest)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        expected_manifest.get("format_version") != CONTROL_FORMAT
        or claimed != hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise NearIdentityOverlapError("Sensitivity-control manifest identity differs")
    rows: dict[tuple[str, str], tuple[float, float, float]] = {}
    parsed_rows = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 5:
            raise NearIdentityOverlapError(
                f"Sensitivity hit row {line_number} does not have five columns"
            )
        query, target, identity, query_coverage, target_coverage = fields
        key = (query, target)
        if key in rows:
            raise NearIdentityOverlapError("Sensitivity hits repeat a query-target pair")
        rows[key] = (
            normalize_fraction(identity),
            normalize_fraction(query_coverage),
            normalize_fraction(target_coverage),
        )
        parsed_rows += 1
    controls: dict[str, bool] = {}
    evidence: list[dict[str, Any]] = []
    tolerance = 1e-12
    for expected in expected_manifest.get("expected_pairs", []):
        control = str(expected.get("control", ""))
        if control not in CONTROL_NAMES or control in controls:
            raise NearIdentityOverlapError("Sensitivity-control inventory differs")
        key = (str(expected.get("query", "")), str(expected.get("target", "")))
        metrics = rows.get(key)
        passed = metrics is not None and all(
            value + tolerance >= 0.9 for value in metrics
        )
        controls[control] = passed
        evidence.append(
            {
                "control": control,
                "identity": None if metrics is None else metrics[0],
                "passed": passed,
                "query": key[0],
                "query_coverage": None if metrics is None else metrics[1],
                "target": key[1],
                "target_coverage": None if metrics is None else metrics[2],
            }
        )
    if set(controls) != set(CONTROL_NAMES):
        raise NearIdentityOverlapError("Sensitivity-control inventory is incomplete")
    return {
        "format_version": "oas_near_identity_mmseqs_sensitivity_result_v1",
        "controls": controls,
        "evidence": evidence,
        "gate_passed": all(controls.values()),
        "parsed_hit_rows": parsed_rows,
        "test_access": expected_manifest["test_access"],
    }


__all__ = [
    "CONTROL_FORMAT",
    "CONTROL_NAMES",
    "audit_sensitivity_hits",
    "build_sensitivity_controls",
]
