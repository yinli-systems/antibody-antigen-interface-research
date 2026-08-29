#!/usr/bin/env python3
"""Audit and gate the paired three-seed matched-local development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
EXPECTED_SOURCE = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
EXPECTED_MANIFEST = "58860008dad531d65938b2e24ddfdfaa1360b36caf4c7ce7fe16ca5a88505dc0"
EXPECTED_AUDIT = "69c2db5b71f763d4e6abf6b6e4fb2643c6d7c45a46441635c03319ce0d90f525"
EXPECTED_CAMPAIGN = "8b33ba578ab1573f50fddabfdeeb5cefefeb0e3d0d7dd20baf3565b337a167b6"
EXPECTED_PROTOCOLS = {
    "control": (
        "8e8947088d94d02fe8c72a5a2c7dd92238653c8e0b745451406cca94cefcbbe9",
        "2878fe20d40e3537ebd697577c54eda50aa4c2e59c331ad2c4e22c0f2e4512b4",
        0.0,
    ),
    "matched_local": (
        "57ceab3899e8167af6d0043321df922938e35f13bb18e74ab59e5694946bae0e",
        "bae1ceaccd646d9234535603d0506f3756dc1c41b7b3ba97ef78730b8eb317a7",
        0.25,
    ),
}
METRICS = (
    "contact_mean_per_complex_auprc",
    "epitope_mean_per_complex_auprc",
    "paratope_mean_per_complex_auprc",
    "minimum_direction_cluster_weighted_margin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    campaign_path = args.run_root / "authority" / "THREE_SEED_CAMPAIGN_AUTHORITY_V1.json"
    if sha256_file(campaign_path) != EXPECTED_CAMPAIGN:
        raise ValueError("campaign authority SHA-256 differs")
    campaign = json.loads(campaign_path.read_text())
    if campaign.get("seeds") != list(SEEDS) or campaign.get("test_loaded") is not False:
        raise ValueError("campaign authority differs")

    arms: dict[str, dict[str, Any]] = {}
    for arm, (protocol_sha, receipt_sha, weight) in EXPECTED_PROTOCOLS.items():
        per_seed = {}
        for seed in SEEDS:
            root = args.run_root / arm / f"seed-{seed}"
            result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            receipt_path = root / "RUN_RECEIPT.json"
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            result_sha = sha256_file(result_path)
            authority = result["authority"]
            selected = result["selected_checkpoint_validation"]
            expected_matched = EXPECTED_MANIFEST if weight else None
            if (
                result.get("experiment") != "epiformer_style_same_protocol"
                or result.get("data", {}).get("feature_mode") != "external_embedding_cache"
                or result.get("data", {}).get("train_complexes") != 512
                or result.get("data", {}).get("validation_complexes") != 128
                or result.get("configuration", {}).get("seed") != seed
                or not math.isclose(float(result.get("configuration", {}).get("matched_local_loss_weight")), weight, abs_tol=1e-15)
                or result.get("selection", {}).get("primary_metric") != "contact_mean_per_complex_auprc"
                or authority.get("protocol_file_sha256") != protocol_sha
                or authority.get("protocol_receipt_sha256") != receipt_sha
                or authority.get("source_manifest_sha256") != EXPECTED_SOURCE
                or authority.get("matched_local_manifest_sha256") != expected_matched
                or authority.get("test_loaded") is not False
                or authority.get("external_test_loaded") is not False
                or receipt.get("result_sha256") != result_sha
                or receipt.get("matched_local_manifest_sha256") != expected_matched
                or receipt.get("test_loaded") is not False
                or receipt.get("external_test_loaded") is not False
            ):
                raise ValueError(f"authority differs for {arm} seed {seed}")
            contact = finite(selected["contact"]["mean_per_complex_auprc"])
            if not math.isclose(contact, finite(result["selection"]["best_validation_primary_metric"]), abs_tol=1e-12):
                raise ValueError(f"selection mismatch for {arm} seed {seed}")
            history = result.get("history", [])
            losses = [finite(row["train"]["matched_local_loss"]) for row in history]
            if not losses or (weight and min(losses) <= 0) or (not weight and any(losses)):
                raise ValueError(f"matched-local instrumentation differs for {arm} seed {seed}")
            per_seed[str(seed)] = {
                "result_sha256": result_sha,
                "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                "selected_epoch": int(result["selection"]["best_epoch"]),
                "epochs_completed": int(result["selection"]["epochs_completed"]),
                "contact_mean_per_complex_auprc": contact,
                "epitope_mean_per_complex_auprc": finite(selected["epitope"]["mean_per_complex_auprc"]),
                "paratope_mean_per_complex_auprc": finite(selected["paratope"]["mean_per_complex_auprc"]),
                "minimum_direction_cluster_weighted_margin": finite(selected["selection_metric"]),
            }
        arms[arm] = {
            "per_seed": per_seed,
            "median": {metric: statistics.median(row[metric] for row in per_seed.values()) for metric in METRICS},
        }

    deltas = {
        str(seed): {
            metric: arms["matched_local"]["per_seed"][str(seed)][metric] - arms["control"]["per_seed"][str(seed)][metric]
            for metric in METRICS
        }
        for seed in SEEDS
    }
    median_deltas = {metric: statistics.median(deltas[str(seed)][metric] for seed in SEEDS) for metric in METRICS}
    contact_deltas = [deltas[str(seed)]["contact_mean_per_complex_auprc"] for seed in SEEDS]
    checks = {
        "paired_median_contact_delta_strictly_positive": median_deltas["contact_mean_per_complex_auprc"] > 0,
        "contact_improves_in_at_least_two_seeds": sum(value > 0 for value in contact_deltas) >= 2,
        "single_seed_contact_regression_at_most_0_01": min(contact_deltas) >= -0.01,
        "median_epitope_regression_at_most_0_01": median_deltas["epitope_mean_per_complex_auprc"] >= -0.01,
        "median_paratope_regression_at_most_0_01": median_deltas["paratope_mean_per_complex_auprc"] >= -0.01,
        "median_partner_margin_regression_at_most_0_01": median_deltas["minimum_direction_cluster_weighted_margin"] >= -0.01,
    }
    payload = {
        "format_version": "sabdab2_matched_local_three_seed_screen_result_v1",
        "status": "PASS_AUDIT",
        "promotion_status": "PASS" if all(checks.values()) else "FAIL",
        "authority": {
            "campaign_authority_sha256": EXPECTED_CAMPAIGN,
            "source_manifest_sha256": EXPECTED_SOURCE,
            "matched_local_manifest_sha256": EXPECTED_MANIFEST,
            "matched_local_audit_sha256": EXPECTED_AUDIT,
            "test_loaded": False,
            "temporal_holdout_authorized": False,
        },
        "arms": arms,
        "candidate_minus_control_by_seed": deltas,
        "paired_median_candidate_minus_control": median_deltas,
        "promotion_checks": checks,
        "promotion_scope_if_pass": "full_2565_549_frozen_development_only",
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(json.dumps({"promotion_status": payload["promotion_status"], "median_deltas": median_deltas, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
