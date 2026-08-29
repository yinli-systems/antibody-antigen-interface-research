#!/usr/bin/env python3
"""Audit full-development top-k2 experts and construct the fixed routed result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FRESH_ANALYSIS_SHA256 = "0e6c6c1a029f35c6bc8f42e6d33aa15f8bac58cb376a5e89f38bc8ce84747800"
CANDIDATE_SOURCE_MANIFEST_SHA256 = "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
SEEDS = (17, 29, 43)
METRICS = ("contact", "epitope", "paratope")


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite metric")
    return number


def selected_metrics(result: dict[str, Any]) -> dict[str, float]:
    validation = result["selected_checkpoint_validation"]
    return {
        metric: finite(validation[metric]["mean_per_complex_auprc"])
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--fresh-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    authority = args.campaign_root / "authority"
    protocol_path = authority / "PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1.json"
    campaign_path = authority / "CAMPAIGN_EDGE_TOPK2_ROUTED_FULL_V1.json"
    if sha256_file(protocol_path) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    if sha256_file(args.fresh_analysis) != FRESH_ANALYSIS_SHA256:
        raise ValueError("fresh156 analysis SHA-256 differs")
    protocol = json.loads(protocol_path.read_text())
    campaign = json.loads(campaign_path.read_text())
    fresh = json.loads(args.fresh_analysis.read_text())
    if (
        protocol.get("protocol_status")
        != "frozen_before_full_development_training"
        or protocol.get("execution_sources", {}).get("analyzer_sha256")
        != sha256_file(Path(__file__))
        or protocol.get("claim_boundary", {}).get("test_loaded") is not False
        or protocol.get("claim_boundary", {}).get("temporal_holdout_loaded") is not False
        or campaign.get("status") != "FROZEN_BEFORE_FULL_DEVELOPMENT_TRAINING"
        or campaign.get("protocol_sha256") != args.protocol_sha256
        or fresh.get("status") != "PASS_FREEZE_FULL_DEVELOPMENT_TASK_ROUTING"
    ):
        raise ValueError("full-development authority differs")

    baseline_values: dict[int, dict[str, float]] = {}
    candidate_values: dict[int, dict[str, float]] = {}
    run_bindings: dict[str, dict[str, dict[str, object]]] = {
        "baseline_current": {},
        "edge_topk2_supervised": {},
    }
    for seed in SEEDS:
        bound = campaign["baseline_binding"][str(seed)]
        baseline_root = Path(bound["run_root"])
        baseline_result_path = baseline_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        baseline_checkpoint = baseline_root / "FRAMECONTACT_SELECTED.pt"
        baseline_receipt = baseline_root / "RUN_RECEIPT.json"
        if (
            sha256_file(baseline_result_path) != bound["result_sha256"]
            or sha256_file(baseline_checkpoint) != bound["checkpoint_sha256"]
            or sha256_file(baseline_receipt) != bound["receipt_sha256"]
        ):
            raise ValueError(f"baseline seed-{seed} artifact differs")
        baseline_result = json.loads(baseline_result_path.read_text())
        baseline_authority = baseline_result.get("authority", {})
        if (
            baseline_result.get("experiment") != "epiformer_style_same_protocol"
            or baseline_result.get("configuration", {}).get("seed") != seed
            or baseline_authority.get("split_manifest_sha256") != FULL_SPLIT_SHA256
            or baseline_authority.get("test_loaded") is not False
            or baseline_authority.get("external_test_loaded") is not False
        ):
            raise ValueError(f"baseline seed-{seed} result authority differs")

        candidate_root = args.campaign_root / f"runs/edge_topk2_supervised/seed-{seed}"
        candidate_result_path = candidate_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        candidate_checkpoint = candidate_root / "FRAMECONTACT_SELECTED.pt"
        candidate_receipt_path = candidate_root / "RUN_RECEIPT.json"
        candidate_result = json.loads(candidate_result_path.read_text())
        candidate_receipt = json.loads(candidate_receipt_path.read_text())
        candidate_authority = candidate_result.get("authority", {})
        if (
            candidate_result.get("experiment") != "epiformer_style_same_protocol"
            or candidate_result.get("configuration", {}).get("seed") != seed
            or candidate_authority.get("split_manifest_sha256") != FULL_SPLIT_SHA256
            or candidate_authority.get("source_manifest_sha256")
            != CANDIDATE_SOURCE_MANIFEST_SHA256
            or candidate_authority.get("protocol_file_sha256")
            != args.protocol_sha256
            or candidate_authority.get("test_loaded") is not False
            or candidate_authority.get("external_test_loaded") is not False
            or candidate_receipt.get("test_loaded") is not False
        ):
            raise ValueError(f"candidate seed-{seed} result authority differs")

        baseline_values[seed] = selected_metrics(baseline_result)
        candidate_values[seed] = selected_metrics(candidate_result)
        run_bindings["baseline_current"][str(seed)] = dict(bound)
        run_bindings["edge_topk2_supervised"][str(seed)] = {
            "run_root": str(candidate_root),
            "result_sha256": sha256_file(candidate_result_path),
            "checkpoint_sha256": sha256_file(candidate_checkpoint),
            "receipt_sha256": sha256_file(candidate_receipt_path),
            "selected_epoch": int(candidate_result["selection"]["best_epoch"]),
        }

    routed = {
        seed: {
            "contact": baseline_values[seed]["contact"],
            "epitope": candidate_values[seed]["epitope"],
            "paratope": baseline_values[seed]["paratope"],
        }
        for seed in SEEDS
    }
    deltas = {
        str(seed): {
            metric: routed[seed][metric] - baseline_values[seed][metric]
            for metric in METRICS
        }
        for seed in SEEDS
    }
    medians = {
        metric: float(np.median([deltas[str(seed)][metric] for seed in SEEDS]))
        for metric in METRICS
    }
    baseline_medians = {
        metric: float(np.median([baseline_values[seed][metric] for seed in SEEDS]))
        for metric in METRICS
    }
    routed_medians = {
        metric: float(np.median([routed[seed][metric] for seed in SEEDS]))
        for metric in METRICS
    }
    gate = protocol["full_development_gate"]
    epitope_positive = sum(deltas[str(seed)]["epitope"] > 0 for seed in SEEDS)
    checks = {
        "median_epitope_delta": medians["epitope"]
        >= finite(gate["median_epitope_delta_minimum"]),
        "epitope_positive_seed_count": epitope_positive
        >= int(gate["epitope_positive_seed_count_minimum"]),
        "minimum_seed_epitope_delta": min(
            deltas[str(seed)]["epitope"] for seed in SEEDS
        )
        >= finite(gate["minimum_seed_epitope_delta"]),
        "maximum_absolute_contact_delta": max(
            abs(deltas[str(seed)]["contact"]) for seed in SEEDS
        )
        <= finite(gate["maximum_absolute_contact_delta"]),
        "maximum_absolute_paratope_delta": max(
            abs(deltas[str(seed)]["paratope"]) for seed in SEEDS
        )
        <= finite(gate["maximum_absolute_paratope_delta"]),
    }
    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_edge_topk2_task_routed_full_development_result_v1",
        "status": (
            "PASS_CURRENT_BEST_FULL_DEVELOPMENT_TASK_ROUTING"
            if passed
            else "FAIL_RETAIN_FULL_DEVELOPMENT_BASELINE"
        ),
        "protocol_sha256": args.protocol_sha256,
        "fresh156_analysis_sha256": FRESH_ANALYSIS_SHA256,
        "task_routing": protocol["task_routing"],
        "run_bindings": run_bindings,
        "baseline_metrics_by_seed": {
            str(seed): baseline_values[seed] for seed in SEEDS
        },
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in SEEDS},
        "baseline_median_metrics": baseline_medians,
        "routed_median_metrics": routed_medians,
        "candidate_minus_baseline_by_seed": deltas,
        "candidate_minus_baseline_median": medians,
        "epitope_positive_seed_count": epitope_positive,
        "promotion_gate": gate,
        "promotion_checks": checks,
        "independent_confirmation": {
            "panel": "fresh156_cluster_disjoint",
            "complex_count": 156,
            "antigen_cluster_count": 49,
            "epitope_delta_by_seed": fresh["candidate_minus_baseline_by_seed"],
            "epitope_paired_antigen_cluster_bootstrap": fresh[
                "epitope_paired_antigen_cluster_bootstrap"
            ],
        },
        "full_validation_is_checkpoint_selection_data": True,
        "full_validation_is_independent_confirmation": False,
        "old_test_loaded": False,
        "temporal_holdout_loaded": False,
        "temporal_holdout_authorized": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "baseline_median_metrics": baseline_medians,
                "routed_median_metrics": routed_medians,
                "median_deltas": medians,
                "promotion_checks": checks,
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
