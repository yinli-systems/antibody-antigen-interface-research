#!/usr/bin/env python3
"""Audit and route six ABF-30B exposed historical test components."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")
TASK_SOURCE = {"contact": "baseline", "epitope": "edge_topk2_supervised", "paratope": "baseline"}
TABLE_FIELDS = ("auroc", "auprc", "precision", "recall", "f1", "mcc", "brier", "ece_15", "mean_per_complex_auprc", "threshold_selected_on_validation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle: handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result): raise ValueError("non-finite metric")
    return result


def cluster_bootstrap(values: dict[str, Any], groups: dict[str, str], replicates: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for identifier, value in values.items():
        grouped.setdefault(groups[identifier], []).append(finite(value))
    group_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(np.mean(rng.choice(group_means, size=group_means.size, replace=True)))
    return {
        "estimate": float(group_means.mean()), "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)), "replicates": replicates,
        "seed": seed, "eligible_complex_count": len(values), "group_count": len(grouped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.campaign_root / "authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json"
    receipt_path = args.campaign_root / "authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1_RECEIPT.json"
    if sha256_file(protocol_path) != args.protocol_sha256: raise ValueError("protocol SHA differs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); protocol_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_OPENING"
        or protocol_receipt.get("protocol_sha256") != args.protocol_sha256
        or protocol.get("execution", {}).get("analyzer_sha256") != sha256_file(Path(__file__))
        or protocol.get("claim_boundary", {}).get("fresh_blind_test") is not False
        or sha256_file(args.test_panel) != protocol["data"]["test_panel_sha256"]
    ):
        raise ValueError("analysis authority differs")
    components: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    result_bindings: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            root = args.campaign_root / "component_results" / arm / f"seed-{seed}"
            result_path = root / "COMPONENT_RESULT.json"; component_receipt_path = root / "COMPONENT_RECEIPT.json"
            result = json.loads(result_path.read_text(encoding="utf-8")); receipt = json.loads(component_receipt_path.read_text(encoding="utf-8"))
            if (
                receipt.get("result_sha256") != sha256_file(result_path)
                or receipt.get("protocol_sha256") != args.protocol_sha256
                or receipt.get("arm") != arm or receipt.get("seed") != seed
                or receipt.get("selection_performed_on_test") is not False
                or receipt.get("fresh_blind_test") is not False
                or result.get("status") != "PASS_ABF30B_EXPOSED_HISTORICAL_TEST_COMPONENT"
                or result.get("audit", {}).get("model_parameters_updated") is not False
                or result.get("audit", {}).get("calibration_refit_on_test") is not False
                or result.get("audit", {}).get("experimental_bound_coordinates_as_model_input") is not False
            ):
                raise ValueError(f"component authority differs: {arm}/{seed}")
            components[arm][seed] = result["metrics"]
            result_bindings[arm][str(seed)] = {"result_sha256": sha256_file(result_path), "receipt_sha256": sha256_file(component_receipt_path), "checkpoint_sha256": receipt["checkpoint_sha256"]}
    panel = json.loads(args.test_panel.read_text(encoding="utf-8"))
    panel_rows = {str(row["instance_id"]): row for row in panel["entries"]}
    routed_by_seed: dict[int, dict[str, Any]] = {}
    metric_table_by_seed: dict[int, dict[str, Any]] = {}
    bootstrap_by_seed: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        routed = copy.deepcopy(components["baseline"][seed])
        routed["epitope"] = copy.deepcopy(components["edge_topk2_supervised"][seed]["epitope"])
        routed_by_seed[seed] = routed
        metric_table_by_seed[seed] = {}
        bootstrap_by_seed[seed] = {}
        for index, task in enumerate(("contact", "epitope", "paratope")):
            metrics = routed[task]
            metric_table_by_seed[seed][task] = {key: finite(metrics[key]) for key in TABLE_FIELDS if key in metrics}
            field = "antibody_cluster_id" if task == "paratope" else "antigen_cluster_id"
            groups = {identifier: str(row[field]) for identifier, row in panel_rows.items()}
            bootstrap_by_seed[seed][task] = cluster_bootstrap(
                metrics["per_complex_auprc_by_instance"], groups,
                int(protocol["execution"]["bootstrap_replicates"]),
                int(protocol["execution"]["bootstrap_seed"]) + seed * 100 + index,
            )
    median_table = {
        task: {field: float(median(metric_table_by_seed[seed][task][field] for seed in SEEDS)) for field in metric_table_by_seed[17][task]}
        for task in ("contact", "epitope", "paratope")
    }
    cluster_median = {
        task: {field: float(median(bootstrap_by_seed[seed][task][field] for seed in SEEDS)) for field in ("estimate", "ci95_lower", "ci95_upper")}
        for task in ("contact", "epitope", "paratope")
    }
    result = {
        "format_version": "abf30b_sabdab2_exposed_historical_test_analysis_v1",
        "status": "PASS_ABF30B_EXPOSED_HISTORICAL_TEST_THREE_SEED_ROUTED",
        "protocol_sha256": args.protocol_sha256, "seeds": list(SEEDS),
        "task_routing": protocol["task_routing"], "result_bindings": result_bindings,
        "metric_table_by_seed": {str(seed): metric_table_by_seed[seed] for seed in SEEDS},
        "median_metric_table": median_table,
        "cluster_bootstrap_by_seed": {str(seed): bootstrap_by_seed[seed] for seed in SEEDS},
        "median_cluster_weighted_summary": cluster_median,
        "claim_boundary": protocol["claim_boundary"],
        "test_complex_count": 549, "predicted_geometry_model_input": True,
        "experimental_bound_coordinates_as_model_input": False,
        "calibration_refit_on_test": False, "model_or_checkpoint_selected_on_test": False,
        "model_parameters_updated": False, "temporal_holdout_loaded": False,
    }
    output = args.campaign_root / "ABF30B_SABDAB2_EXPOSED_HISTORICAL_TEST_ANALYSIS_V1.json"
    receipt_output = args.campaign_root / "ABF30B_SABDAB2_EXPOSED_HISTORICAL_TEST_ANALYSIS_V1_RECEIPT.json"
    for path in (output, receipt_output):
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    analysis_sha = publish(output, result)
    receipt_sha = publish(receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_historical_test_analysis_receipt_v1",
        "status": result["status"], "analysis_sha256": analysis_sha,
        "protocol_sha256": args.protocol_sha256, "all_six_components_verified": True,
        "selection_performed_on_test": False, "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3, "temporal_holdout_loaded": False,
    })
    print(json.dumps({"status": result["status"], "median_metric_table": median_table, "median_cluster_weighted_summary": cluster_median, "analysis_sha256": analysis_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
