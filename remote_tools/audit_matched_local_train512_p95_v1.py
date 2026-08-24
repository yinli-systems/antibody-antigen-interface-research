#!/usr/bin/env python3
"""Audit frozen matched-local train supervision without reading validation or test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CANARY_IDS = {
    "pdb_00008h91_D_+",
    "pdb_00009ck4_A_B",
    "pdb_00009m1o_S1_S2",
    "pdb_00009qub_F_E",
}
EXPECTED_CONFIGURATION = {
    "antigen_patch_radius_angstrom": 12.0,
    "cdr_edge_radius_angstrom": 10.0,
    "cdr_patch_radius_angstrom": 8.0,
    "composition_dirichlet_pseudocount_per_amino_acid": 1.0,
    "distance_bin_boundaries_angstrom": [8.0, 16.0, 24.0, 32.0],
    "matching_thresholds": {
        "center_rsa_absolute_difference": 0.25,
        "center_sasa_relative_difference": 0.40,
        "composition_js_distance": 0.25,
        "count_relative_difference": 0.25,
        "curvature_absolute_difference": 0.10,
        "surface_area_relative_difference": 0.25,
    },
    "max_decoys_per_complex": 32,
    "max_decoys_per_positive": 4,
    "rsa_threshold": 0.2,
    "surface_edge_radius_angstrom": 10.0,
}
EXPECTED_SPLIT_SHA256 = (
    "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sum(int(record["eligible_cdr_surface_contact_pair_count"]) for record in records)
    covered = sum(int(record["covered_cdr_surface_contact_pair_count"]) for record in records)
    return {
        "instance_count": len(records),
        "eligible_pairs": eligible,
        "covered_pairs": covered,
        "strict_pair_coverage": None if eligible == 0 else covered / eligible,
        "instances_with_decoys": sum(int(record["decoy_count"]) > 0 for record in records),
        "decoys": sum(int(record["decoy_count"]) for record in records),
    }


def _audit_artifact(root: Path, record: dict[str, Any]) -> None:
    path = root / str(record["artifact_path"])
    if not path.is_file() or path.stat().st_size != int(record["artifact_size_bytes"]):
        raise ValueError(f"Artifact size/path mismatch: {record['instance_id']}")
    if _sha256(path) != record["artifact_sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch: {record['instance_id']}")
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["instance_id"].item()) != record["instance_id"]:
            raise ValueError(f"Artifact identity mismatch: {record['instance_id']}")
        decoy_count = int(payload["decoy_count"].item())
        if decoy_count != int(record["decoy_count"]) or decoy_count > 32:
            raise ValueError(f"Artifact decoy count mismatch: {record['instance_id']}")
        for name in (
            "positive_center_flat_indices",
            "negative_center_flat_indices",
            "distance_bins",
            "distance_to_epitope_angstrom",
            "matching_scores",
            "matching_diagnostics",
        ):
            if payload[name].shape[0] != decoy_count:
                raise ValueError(f"Artifact leading axis mismatch: {record['instance_id']}:{name}")
        for name in payload.files:
            values = payload[name]
            if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
                raise ValueError(f"Non-finite artifact values: {record['instance_id']}:{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-heldout-coverage", type=float, default=0.95)
    arguments = parser.parse_args()

    manifest_path = arguments.artifact_root / "MATCHED_LOCAL_DECOY_MANIFEST.json"
    receipt_path = arguments.artifact_root / "MATCHED_LOCAL_DECOY_RECEIPT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(manifest_path)
    if receipt.get("manifest_sha256") != manifest_sha256 or receipt.get("status") != "PASS":
        raise ValueError("Materialization receipt is not an authenticated PASS")
    authority = manifest.get("authority", {})
    if (
        authority.get("split_sha256") != EXPECTED_SPLIT_SHA256
        or authority.get("task_labels_read_for_roles") != ["train"]
        or authority.get("validation_labels_read") is not False
        or authority.get("test_labels_read") is not False
        or authority.get("model_predictions_read") is not False
    ):
        raise ValueError("Matched-local authority differs from the frozen train-only contract")
    if manifest.get("configuration") != EXPECTED_CONFIGURATION:
        raise ValueError("Matched-local configuration differs from the frozen p95 caliper")

    records = list(manifest.get("records", []))
    identifiers = [str(record["instance_id"]) for record in records]
    if len(records) != 512 or len(set(identifiers)) != 512:
        raise ValueError("Expected exactly 512 unique training instances")
    if not CANARY_IDS.issubset(identifiers):
        raise ValueError("Frozen four-complex canary is absent from the full training run")
    for record in records:
        _audit_artifact(arguments.artifact_root, record)

    canary = [record for record in records if record["instance_id"] in CANARY_IDS]
    heldout = [record for record in records if record["instance_id"] not in CANARY_IDS]
    full_metrics = _coverage(records)
    canary_metrics = _coverage(canary)
    heldout_metrics = _coverage(heldout)
    counts = manifest.get("counts", {})
    for observed, expected, name in (
        (full_metrics["eligible_pairs"], counts.get("eligible_cdr_surface_contact_pairs"), "eligible"),
        (full_metrics["covered_pairs"], counts.get("covered_cdr_surface_contact_pairs"), "covered"),
        (full_metrics["decoys"], counts.get("decoys"), "decoys"),
    ):
        if observed != expected:
            raise ValueError(f"Manifest aggregate mismatch: {name}")
    if not math.isclose(
        float(full_metrics["strict_pair_coverage"]),
        float(counts.get("strict_pair_coverage")),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Manifest aggregate mismatch: strict coverage")

    heldout_pass = (
        heldout_metrics["strict_pair_coverage"] is not None
        and heldout_metrics["strict_pair_coverage"] >= arguments.minimum_heldout_coverage
    )
    result = {
        "format_version": "sabdab2_matched_local_train512_p95_audit_v1",
        "status": "PASS" if heldout_pass else "FAIL_INSUFFICIENT_HELDOUT_COVERAGE",
        "manifest_sha256": manifest_sha256,
        "receipt_sha256": _sha256(receipt_path),
        "artifact_sha256_verified_count": len(records),
        "artifact_nonfinite_count": 0,
        "frozen_caliper_canary": canary_metrics,
        "heldout_after_caliper_freeze": heldout_metrics,
        "full_training_set": full_metrics,
        "minimum_heldout_strict_pair_coverage": arguments.minimum_heldout_coverage,
        "validation_labels_read": False,
        "test_labels_read": False,
        "model_predictions_read": False,
        "test_loaded": False,
    }
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_json(result))
    print(json.dumps({**result, "result_sha256": _sha256(arguments.output)}, indent=2, sort_keys=True))
    if not heldout_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
