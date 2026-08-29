#!/usr/bin/env python3
"""Audit and gate the paired matched-local seed-17 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_SHA256 = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
EXPECTED_MATCHED_MANIFEST_SHA256 = "58860008dad531d65938b2e24ddfdfaa1360b36caf4c7ce7fe16ca5a88505dc0"
EXPECTED_MATCHED_AUDIT_SHA256 = "69c2db5b71f763d4e6abf6b6e4fb2643c6d7c45a46441635c03319ce0d90f525"
EXPECTED_PROTOCOLS = {
    "control": (
        "cd7fc6eb4d68054e9d3a4fee5bc10a81b4bf3d645d7760bffca181ed678d7cfe",
        "2c9b012bda1ff3acb6055551bae9dc7f2fcfc23fc7c8d29cb97647f4311a9e17",
        0.0,
    ),
    "matched_local": (
        "1fbde99c69acadad32ecf54e554dd2e4001e1575c0ead5177931f5f7c321e639",
        "8d91f2782915b752e01bea75fce0f2a3411f31bd332b1f71ac9570761728a027",
        0.25,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite screen value")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(arguments.output)

    arms: dict[str, Any] = {}
    for arm, (protocol_sha, receipt_sha, weight) in EXPECTED_PROTOCOLS.items():
        root = arguments.run_root / arm / "seed-17"
        result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        receipt_path = root / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        result_sha = _sha256(result_path)
        authority = result["authority"]
        selected = result["selected_checkpoint_validation"]
        expected_matched = EXPECTED_MATCHED_MANIFEST_SHA256 if weight > 0 else None
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or result.get("data", {}).get("feature_mode") != "external_embedding_cache"
            or result.get("data", {}).get("train_complexes") != 512
            or result.get("data", {}).get("validation_complexes") != 128
            or result.get("configuration", {}).get("seed") != 17
            or not math.isclose(
                float(result.get("configuration", {}).get("matched_local_loss_weight")),
                weight,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or result.get("selection", {}).get("primary_metric")
            != "contact_mean_per_complex_auprc"
            or authority.get("protocol_file_sha256") != protocol_sha
            or authority.get("protocol_receipt_sha256") != receipt_sha
            or authority.get("source_manifest_sha256") != EXPECTED_SOURCE_SHA256
            or authority.get("matched_local_manifest_sha256") != expected_matched
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or receipt.get("result_sha256") != result_sha
            or receipt.get("matched_local_manifest_sha256") != expected_matched
            or receipt.get("test_loaded") is not False
            or receipt.get("external_test_loaded") is not False
        ):
            raise ValueError(f"{arm} authority or receipt differs")
        contact = _finite(selected["contact"]["mean_per_complex_auprc"])
        primary = _finite(result["selection"]["best_validation_primary_metric"])
        if not math.isclose(contact, primary, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{arm} selected checkpoint does not match contact primary")
        history = result.get("history", [])
        if not history:
            raise ValueError(f"{arm} training history is empty")
        matched_losses = [_finite(row["train"]["matched_local_loss"]) for row in history]
        if (weight > 0 and min(matched_losses) <= 0.0) or (
            weight == 0.0 and any(value != 0.0 for value in matched_losses)
        ):
            raise ValueError(f"{arm} matched-local loss instrumentation differs")
        arms[arm] = {
            "result_sha256": result_sha,
            "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
            "selected_epoch": int(result["selection"]["best_epoch"]),
            "epochs_completed": int(result["selection"]["epochs_completed"]),
            "contact_mean_per_complex_auprc": contact,
            "epitope_mean_per_complex_auprc": _finite(
                selected["epitope"]["mean_per_complex_auprc"]
            ),
            "paratope_mean_per_complex_auprc": _finite(
                selected["paratope"]["mean_per_complex_auprc"]
            ),
            "minimum_direction_cluster_weighted_margin": _finite(
                selected["selection_metric"]
            ),
            "matched_local_loss_by_epoch": matched_losses,
        }

    control = arms["control"]
    candidate = arms["matched_local"]
    deltas = {
        key: candidate[key] - control[key]
        for key in (
            "contact_mean_per_complex_auprc",
            "epitope_mean_per_complex_auprc",
            "paratope_mean_per_complex_auprc",
            "minimum_direction_cluster_weighted_margin",
        )
    }
    checks = {
        "contact_strictly_improves": deltas["contact_mean_per_complex_auprc"] > 0.0,
        "epitope_regression_at_most_0_01": deltas["epitope_mean_per_complex_auprc"]
        >= -0.01,
        "paratope_regression_at_most_0_01": deltas["paratope_mean_per_complex_auprc"]
        >= -0.01,
        "partner_margin_regression_at_most_0_01": deltas[
            "minimum_direction_cluster_weighted_margin"
        ]
        >= -0.01,
    }
    payload = {
        "format_version": "sabdab2_matched_local_seed17_screen_result_v2",
        "status": "PASS_AUDIT",
        "promotion_status": "PASS" if all(checks.values()) else "FAIL",
        "authority": {
            "source_manifest_sha256": EXPECTED_SOURCE_SHA256,
            "matched_local_manifest_sha256": EXPECTED_MATCHED_MANIFEST_SHA256,
            "matched_local_audit_sha256": EXPECTED_MATCHED_AUDIT_SHA256,
            "test_loaded": False,
        },
        "arms": arms,
        "candidate_minus_control": deltas,
        "promotion_checks": checks,
        "promotion_scope_if_pass": "three_seed_frozen_development_screen_only",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical(payload))
    print(json.dumps({"promotion_status": payload["promotion_status"], "deltas": deltas}, sort_keys=True))


if __name__ == "__main__":
    main()
