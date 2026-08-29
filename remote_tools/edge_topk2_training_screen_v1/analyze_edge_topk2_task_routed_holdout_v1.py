#!/usr/bin/env python3
"""Audit fresh156 component measurements and apply the frozen routing gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_BINDING_SHA256 = "accf813011342102fa2572aa27c2bde6796636cde3ac872756472c32b66f4ae1"
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


def cluster_bootstrap(
    deltas: dict[str, float],
    clusters: dict[str, str],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for identifier, delta in deltas.items():
        grouped.setdefault(clusters[identifier], []).append(delta)
    means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(means), (replicates, len(means)))
    samples = means[indices].mean(axis=1)
    return {
        "cluster_count": len(means),
        "point_estimate": float(means.mean()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    authority = args.campaign_root / "authority"
    protocol_path = authority / "PROTOCOL_TASK_ROUTED_FRESH156_V1.json"
    split_path = authority / "split-fresh156-measurement-only.csv"
    if sha256_file(protocol_path) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT_MEASUREMENT"
        or protocol.get("execution_sources", {}).get("analyzer_sha256")
        != sha256_file(Path(__file__))
        or protocol.get("checkpoint_binding_sha256") != CHECKPOINT_BINDING_SHA256
        or protocol.get("fresh_holdout_selection_performed") is not False
        or protocol.get("test_loaded") is not False
        or protocol.get("temporal_holdout_loaded") is not False
        or sha256_file(split_path) != protocol.get("holdout", {}).get("split_sha256")
    ):
        raise ValueError("frozen routing authority differs")
    with split_path.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    clusters = {
        str(row["instance_id"]): str(row["antigen_cluster_id"])
        for row in split_rows
        if row["role"] == "validation"
        and row.get("fresh_holdout_included") == "true"
    }
    if len(clusters) != 156 or len(set(clusters.values())) != 49:
        raise ValueError("fresh holdout split differs")

    component_runs: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    component_values: dict[str, dict[int, dict[str, float]]] = {
        arm: {} for arm in ARMS
    }
    component_per_complex: dict[
        str, dict[int, dict[str, dict[str, float]]]
    ] = {arm: {} for arm in ARMS}
    component_partner: dict[str, dict[int, Any]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            run_root = args.campaign_root / f"component-results/{arm}/seed-{seed}"
            result_path = run_root / "FRESH156_RESULT.json"
            receipt_path = run_root / "FRESH156_RECEIPT.json"
            result_sha256 = sha256_file(result_path)
            receipt_sha256 = sha256_file(receipt_path)
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            audit = result.get("audit", {})
            if (
                result.get("status") != "PASS_MEASUREMENT_ONLY_FRESH156"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or receipt.get("result_sha256") != result_sha256
                or receipt.get("protocol_sha256") != args.protocol_sha256
                or receipt.get("fresh_holdout_selection_performed") is not False
                or receipt.get("model_parameters_updated") is not False
                or receipt.get("test_loaded") is not False
                or receipt.get("temporal_holdout_loaded") is not False
                or audit.get("protocol_sha256") != args.protocol_sha256
                or audit.get("fresh_holdout_selection_performed") is not False
                or audit.get("calibration_refit_on_fresh_holdout") is not False
                or audit.get("model_parameters_updated") is not False
                or audit.get("test_loaded") is not False
                or audit.get("temporal_holdout_loaded") is not False
            ):
                raise ValueError(f"{arm} seed-{seed} component receipt differs")
            local = result.get("per_complex_auprc", {})
            if set(local) != set(clusters):
                raise ValueError(f"{arm} seed-{seed} component IDs differ")
            component_values[arm][seed] = {
                metric: finite(result["metrics"][metric]["mean_per_complex_auprc"])
                for metric in METRICS
            }
            component_per_complex[arm][seed] = {
                identifier: {metric: finite(row[metric]) for metric in METRICS}
                for identifier, row in local.items()
            }
            for metric in METRICS:
                observed = float(
                    np.mean(
                        [
                            row[metric]
                            for row in component_per_complex[arm][seed].values()
                        ]
                    )
                )
                if abs(observed - component_values[arm][seed][metric]) > 1e-10:
                    raise ValueError(f"{arm} seed-{seed} {metric} mean differs")
            component_partner[arm][seed] = result["metrics"].get(
                "partner_retrieval_by_direction"
            )
            component_runs[arm][str(seed)] = {
                "result_sha256": result_sha256,
                "receipt_sha256": receipt_sha256,
                "metrics": component_values[arm][seed],
            }

    baseline = {
        seed: dict(component_values["baseline_current"][seed]) for seed in SEEDS
    }
    routed = {
        seed: {
            "contact": component_values["baseline_current"][seed]["contact"],
            "paratope": component_values["baseline_current"][seed]["paratope"],
            "epitope": component_values["edge_topk2_supervised"][seed]["epitope"],
        }
        for seed in SEEDS
    }
    deltas = {
        str(seed): {
            metric: routed[seed][metric] - baseline[seed][metric]
            for metric in METRICS
        }
        for seed in SEEDS
    }
    medians = {
        metric: float(np.median([deltas[str(seed)][metric] for seed in SEEDS]))
        for metric in METRICS
    }
    minima = {
        metric: float(min(deltas[str(seed)][metric] for seed in SEEDS))
        for metric in METRICS
    }
    epitope_positive = sum(deltas[str(seed)]["epitope"] > 0 for seed in SEEDS)
    gate = protocol["promotion_gate"]
    checks = {
        "median_epitope_delta": medians["epitope"]
        >= finite(gate["median_epitope_delta_minimum"]),
        "epitope_positive_seed_count": epitope_positive
        >= int(gate["epitope_positive_seed_count_minimum"]),
        "minimum_seed_epitope_delta": minima["epitope"]
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
    epitope_delta_by_instance = {
        identifier: float(
            np.mean(
                [
                    component_per_complex["edge_topk2_supervised"][seed][identifier][
                        "epitope"
                    ]
                    - component_per_complex["baseline_current"][seed][identifier][
                        "epitope"
                    ]
                    for seed in SEEDS
                ]
            )
        )
        for identifier in clusters
    }
    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_edge_topk2_task_routed_holdout_analysis_v1",
        "status": (
            "PASS_FREEZE_FULL_DEVELOPMENT_TASK_ROUTING"
            if passed
            else "FAIL_RETAIN_SINGLE_MODEL_BASELINE"
        ),
        "protocol_sha256": args.protocol_sha256,
        "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
        "component_runs": component_runs,
        "task_routing": protocol["candidate"],
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in SEEDS},
        "candidate_minus_baseline_by_seed": deltas,
        "candidate_minus_baseline_median": medians,
        "candidate_minus_baseline_minimum_seed": minima,
        "epitope_positive_seed_count": epitope_positive,
        "epitope_paired_antigen_cluster_bootstrap": cluster_bootstrap(
            epitope_delta_by_instance, clusters, seed=20260829
        ),
        "routed_partner_retrieval_by_seed": {
            str(seed): component_partner["baseline_current"][seed] for seed in SEEDS
        },
        "promotion_gate": gate,
        "promotion_checks": checks,
        "full_development_protocol_authorized": passed,
        "fresh_holdout_selection_performed": False,
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
