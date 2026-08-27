#!/usr/bin/env python3
"""Audit the six sealed confirmation measurements and apply the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CAMPAIGN_SHA256 = "31ac56aca9ed32371abfb34154693faf11b36908fe035201524bac8c4cc6cf75"
CONTRACT_SHA256 = "a0f8a67d43bd3fe30c4769e3ad2a38bd7923f6d42071f7ce9fed8bfe0b609760"
PARTITION_SHA256 = "a1a21789c1d1890ff957b92f966720fd169b2b4004fe4acd17c693c37658d73a"
ARMS = ("baseline_current", "edge_topk2_supervised")
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
        raise ValueError("non-finite value")
    return number


def bootstrap_cluster_ci(
    deltas_by_instance: dict[str, float],
    cluster_by_instance: dict[str, str],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    by_cluster: dict[str, list[float]] = {}
    for identifier, delta in deltas_by_instance.items():
        by_cluster.setdefault(cluster_by_instance[identifier], []).append(delta)
    cluster_means = np.asarray(
        [np.mean(by_cluster[key]) for key in sorted(by_cluster)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(cluster_means), size=(replicates, len(cluster_means)), endpoint=False
    )
    bootstrap = cluster_means[sampled].mean(axis=1)
    return {
        "cluster_count": int(len(cluster_means)),
        "point_estimate": float(cluster_means.mean()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    authority = args.campaign_root / "authority"
    campaign_path = authority / "CAMPAIGN_EDGE_TOPK2_THREE_SEED_CONFIRMATION_V1.json"
    contract_path = authority / "PROTOCOL_CONFIRMATION_MEASUREMENT_CONTRACT_V1.json"
    binding_path = authority / "CHECKPOINT_BINDING_EDGE_TOPK2_CONFIRMATION_V1.json"
    if (
        sha256_file(campaign_path) != CAMPAIGN_SHA256
        or sha256_file(contract_path) != CONTRACT_SHA256
        or sha256_file(args.partition) != PARTITION_SHA256
        or sha256_file(binding_path) != args.binding_sha256
    ):
        raise ValueError("frozen authority differs")
    campaign = json.loads(campaign_path.read_text())
    contract = json.loads(contract_path.read_text())
    binding = json.loads(binding_path.read_text())
    partition = json.loads(args.partition.read_text())
    if (
        campaign.get("confirmation_labels_read") is not False
        or campaign.get("test_loaded") is not False
        or contract.get("status") != "FROZEN_BEFORE_CHECKPOINT_BINDING"
        or binding.get("status") != "FROZEN_BEFORE_CONFIRMATION_MEASUREMENT"
        or binding.get("selection_performed_on_confirmation") is not False
        or binding.get("execution_sources", {}).get("analyzer_sha256")
        != sha256_file(Path(__file__))
        or partition.get("task_labels_read") is not False
        or partition.get("test_features_or_labels_read") is not False
    ):
        raise ValueError("measurement authority differs")
    confirmation = partition["roles"]["confirmation"]
    cluster_by_instance = {
        str(row["instance_id"]): str(row["antigen_cluster_id"])
        for row in confirmation
    }
    if len(cluster_by_instance) != 65 or len(set(cluster_by_instance.values())) != 27:
        raise ValueError("confirmation partition differs")

    runs: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    values: dict[str, dict[int, dict[str, float]]] = {arm: {} for arm in ARMS}
    per_complex: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        arm: {} for arm in ARMS
    }
    for arm in ARMS:
        for seed in SEEDS:
            run_root = args.campaign_root / f"confirmation-results/{arm}/seed-{seed}"
            result_path = run_root / "CONFIRMATION_RESULT.json"
            receipt_path = run_root / "CONFIRMATION_RECEIPT.json"
            result_sha256 = sha256_file(result_path)
            receipt_sha256 = sha256_file(receipt_path)
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            audit = result.get("audit", {})
            if (
                result.get("status") != "PASS_MEASUREMENT_ONLY_CONFIRMATION"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or receipt.get("result_sha256") != result_sha256
                or receipt.get("checkpoint_binding_sha256") != args.binding_sha256
                or receipt.get("selection_performed_on_confirmation") is not False
                or receipt.get("calibration_refit_on_confirmation") is not False
                or receipt.get("model_parameters_updated") is not False
                or receipt.get("test_loaded") is not False
                or receipt.get("temporal_holdout_loaded") is not False
                or audit.get("checkpoint_binding_sha256") != args.binding_sha256
                or audit.get("selection_performed_on_confirmation") is not False
                or audit.get("calibration_refit_on_confirmation") is not False
                or audit.get("model_parameters_updated") is not False
                or audit.get("test_loaded") is not False
                or audit.get("temporal_holdout_loaded") is not False
            ):
                raise ValueError(f"{arm} seed-{seed} receipt differs")
            local = result.get("per_complex_auprc", {})
            if set(local) != set(cluster_by_instance):
                raise ValueError(f"{arm} seed-{seed} per-complex IDs differ")
            values[arm][seed] = {
                metric: finite(result["metrics"][metric]["mean_per_complex_auprc"])
                for metric in METRICS
            }
            per_complex[arm][seed] = {
                identifier: {metric: finite(row[metric]) for metric in METRICS}
                for identifier, row in local.items()
            }
            for metric in METRICS:
                observed_mean = float(
                    np.mean([row[metric] for row in per_complex[arm][seed].values()])
                )
                if abs(observed_mean - values[arm][seed][metric]) > 1e-10:
                    raise ValueError(f"{arm} seed-{seed} {metric} mean differs")
            runs[arm][str(seed)] = {
                "result_sha256": result_sha256,
                "receipt_sha256": receipt_sha256,
                "metrics": values[arm][seed],
            }

    deltas_by_seed = {
        str(seed): {
            metric: values["edge_topk2_supervised"][seed][metric]
            - values["baseline_current"][seed][metric]
            for metric in METRICS
        }
        for seed in SEEDS
    }
    medians = {
        metric: float(np.median([deltas_by_seed[str(seed)][metric] for seed in SEEDS]))
        for metric in METRICS
    }
    minima = {
        metric: float(min(deltas_by_seed[str(seed)][metric] for seed in SEEDS))
        for metric in METRICS
    }
    epitope_positive_seed_count = sum(
        deltas_by_seed[str(seed)]["epitope"] > 0 for seed in SEEDS
    )
    gate = contract["promotion_gate"]
    checks = {
        "median_epitope_delta": medians["epitope"]
        >= finite(gate["median_epitope_delta_minimum"]),
        "epitope_positive_seed_count": epitope_positive_seed_count
        >= int(gate["epitope_positive_seed_count_minimum"]),
        "minimum_seed_epitope_delta": minima["epitope"]
        >= finite(gate["minimum_seed_epitope_delta"]),
        "median_contact_delta": medians["contact"]
        >= finite(gate["median_contact_delta_minimum"]),
        "minimum_seed_contact_delta": minima["contact"]
        >= finite(gate["minimum_seed_contact_delta"]),
        "median_paratope_delta": medians["paratope"]
        >= finite(gate["median_paratope_delta_minimum"]),
        "minimum_seed_paratope_delta": minima["paratope"]
        >= finite(gate["minimum_seed_paratope_delta"]),
    }

    paired_cluster_bootstrap: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        averaged_delta = {
            identifier: float(
                np.mean(
                    [
                        per_complex["edge_topk2_supervised"][seed][identifier][metric]
                        - per_complex["baseline_current"][seed][identifier][metric]
                        for seed in SEEDS
                    ]
                )
            )
            for identifier in cluster_by_instance
        }
        paired_cluster_bootstrap[metric] = bootstrap_cluster_ci(
            averaged_delta,
            cluster_by_instance,
            seed=20260826 + metric_index,
        )

    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_edge_topk2_confirmation_analysis_v1",
        "status": (
            "PASS_FREEZE_FULL_DEVELOPMENT_PROTOCOL"
            if passed
            else "FAIL_RETAIN_BASELINE"
        ),
        "campaign_sha256": CAMPAIGN_SHA256,
        "measurement_contract_sha256": CONTRACT_SHA256,
        "checkpoint_binding_sha256": args.binding_sha256,
        "runs": runs,
        "candidate_minus_baseline_by_seed": deltas_by_seed,
        "candidate_minus_baseline_median": medians,
        "candidate_minus_baseline_minimum_seed": minima,
        "epitope_positive_seed_count": epitope_positive_seed_count,
        "paired_antigen_cluster_bootstrap": paired_cluster_bootstrap,
        "promotion_gate": gate,
        "promotion_checks": checks,
        "full_development_protocol_authorized": passed,
        "old_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "median_deltas": medians,
                "promotion_checks": checks,
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
