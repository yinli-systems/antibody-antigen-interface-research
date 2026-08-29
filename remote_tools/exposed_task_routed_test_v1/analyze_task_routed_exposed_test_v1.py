#!/usr/bin/env python3
"""Compose the two precommitted components into one task-routed test result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("baseline_current", "edge_topk2_supervised")
TASK_SOURCE = {
    "contact": "baseline_current",
    "epitope": "edge_topk2_supervised",
    "paratope": "baseline_current",
}
TABLE_FIELDS = (
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "mcc",
    "brier",
    "ece_15",
    "mean_per_complex_auprc",
    "threshold_selected_on_validation",
)


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


def cluster_bootstrap(
    values: dict[str, float],
    groups: dict[str, str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if set(values) - set(groups):
        raise ValueError("per-complex values have unknown cluster identifiers")
    grouped: dict[str, list[float]] = {}
    for identifier, value in values.items():
        grouped.setdefault(groups[identifier], []).append(float(value))
    group_means = {
        group: float(np.mean(group_values))
        for group, group_values in sorted(grouped.items())
    }
    array = np.asarray(list(group_means.values()), dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(
            np.mean(generator.choice(array, size=array.size, replace=True))
        )
    return {
        "estimate": float(array.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "replicates": replicates,
        "seed": seed,
        "eligible_complex_count": len(values),
        "group_count": len(group_means),
        "group_means": group_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    args = parser.parse_args()

    authority = args.campaign_root / "authority"
    protocol_path = authority / "PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1.json"
    if sha256_file(protocol_path) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_HISTORICAL_BATCH"
        or protocol.get("execution", {}).get("analyzer_sha256")
        != sha256_file(Path(__file__))
        or protocol.get("task_routing", {}).get("routing_is_fixed_per_task") is not True
        or protocol.get("claim_boundary", {}).get("fresh_blind_test") is not False
    ):
        raise ValueError("analysis authority differs")
    if sha256_file(args.test_panel) != protocol["data"]["test_partner_panel_sha256"]:
        raise ValueError("test panel SHA-256 differs")

    components: dict[str, dict[str, object]] = {}
    component_receipt_hashes: dict[str, str] = {}
    for arm in ARMS:
        root = args.campaign_root / "component-results" / arm
        result_path = root / "COMPONENT_RESULT.json"
        receipt_path = root / "COMPONENT_RECEIPT.json"
        result = json.loads(result_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("result_sha256") != sha256_file(result_path)
            or receipt.get("protocol_sha256") != args.protocol_sha256
            or receipt.get("arm") != arm
            or receipt.get("selection_performed_on_this_measurement") is not False
            or receipt.get("fresh_blind_test") is not False
            or result.get("status") != "PASS_EXPOSED_HISTORICAL_COMPONENT"
            or result.get("arm") != arm
            or result.get("audit", {}).get("model_parameters_updated") is not False
            or result.get("audit", {}).get("calibration_refit_on_test") is not False
        ):
            raise ValueError(f"{arm} component authority differs")
        components[arm] = result
        component_receipt_hashes[arm] = sha256_file(receipt_path)

    baseline_metrics = components["baseline_current"]["metrics"]
    candidate_metrics = components["edge_topk2_supervised"]["metrics"]
    routed_metrics = copy.deepcopy(baseline_metrics)
    routed_metrics["epitope"] = copy.deepcopy(candidate_metrics["epitope"])
    panel = json.loads(args.test_panel.read_text())
    panel_rows = {str(row["instance_id"]): row for row in panel["entries"]}
    replicates = int(protocol["execution"]["bootstrap_replicates"])
    base_seed = int(protocol["execution"]["bootstrap_seed"])
    metric_table: dict[str, dict[str, object]] = {}
    for index, task in enumerate(("contact", "epitope", "paratope")):
        task_metrics = routed_metrics[task]
        group_field = "antibody_cluster_id" if task == "paratope" else "antigen_cluster_id"
        groups = {
            identifier: str(row[group_field]) for identifier, row in panel_rows.items()
        }
        bootstrap = cluster_bootstrap(
            task_metrics["per_complex_auprc_by_instance"],
            groups,
            replicates=replicates,
            seed=base_seed + 1000 * (index + 1),
        )
        task_metrics["mean_per_complex_auprc_cluster_bootstrap_ci"] = bootstrap
        task_metrics["mean_per_complex_auprc_cluster_definition"] = (
            f"equal-weight mean over {group_field} means; positive-required "
            "per-complex AP excludes zero-positive complexes"
        )
        metric_table[task] = {
            key: task_metrics[key] for key in TABLE_FIELDS if key in task_metrics
        }
        metric_table[task]["cluster_weighted_mean_per_complex_auprc"] = bootstrap[
            "estimate"
        ]
        metric_table[task]["cluster_bootstrap_ci95"] = [
            bootstrap["ci95_lower"],
            bootstrap["ci95_upper"],
        ]
        metric_table[task]["source"] = TASK_SOURCE[task]

    result = {
        "format_version": "sabdab2_task_routed_exposed_historical_result_v1",
        "status": "PASS_EXPOSED_HISTORICAL_TASK_ROUTED_SEED17",
        "seed": int(protocol["seed"]),
        "claim_boundary": protocol["claim_boundary"],
        "task_routing": protocol["task_routing"],
        "metric_table": metric_table,
        "routed_metrics": routed_metrics,
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "component_result_sha256": {
                arm: sha256_file(
                    args.campaign_root
                    / "component-results"
                    / arm
                    / "COMPONENT_RESULT.json"
                )
                for arm in ARMS
            },
            "component_receipt_sha256": component_receipt_hashes,
            "test_complexes": int(protocol["data"]["test_complex_count"]),
            "candidate_grid": protocol["data"]["candidate_grid"],
            "calibration_refit_on_test": False,
            "model_or_checkpoint_selected_on_this_measurement": False,
            "model_parameters_updated": False,
            "old_test_previously_exposed": True,
            "fresh_blind_test": False,
            "historical_benchmark_only": True,
            "cumulative_test_access_ordinal": 2,
        },
    }
    result_path = args.campaign_root / "TASK_ROUTED_EXPOSED_HISTORICAL_RESULT.json"
    receipt_path = args.campaign_root / "TASK_ROUTED_EXPOSED_HISTORICAL_RECEIPT.json"
    for path in (result_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    result_path.write_bytes(canonical_bytes(result))
    receipt = {
        "format_version": "sabdab2_task_routed_exposed_historical_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(result_path),
        "protocol_sha256": args.protocol_sha256,
        "seed": result["seed"],
        "selection_performed_on_this_measurement": False,
        "old_test_previously_exposed": True,
        "fresh_blind_test": False,
        "historical_benchmark_only": True,
        "cumulative_test_access_ordinal": 2,
    }
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": metric_table,
                "result_sha256": receipt["result_sha256"],
                "receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
