#!/usr/bin/env python3
"""Associate frozen predicted-minus-bound per-complex AP with geometry quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_predicted_geometry_quality_mechanism_v1 import (
    _cluster_bootstrap_spearman,
    _quartiles,
)


PLAN_SHA256 = "86ef7dff66b4f636f929f28728cf300f360041577741c891059ff601f9c5f272"
MECHANISM_ANALYSIS_SHA256 = (
    "b8c8f3ea0ae35afec9341c5145d9467c92faf5fdcda09e8776bd562247f5c4f5"
)
SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")
GEOMETRY_SOURCES = ("predicted", "bound")
TASK_ARM = {
    "contact": "baseline",
    "epitope": "edge_topk2_supervised",
    "paratope": "baseline",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def load_metric_result(
    metrics_root: Path,
    geometry_source: str,
    arm: str,
    seed: int,
    plan: dict[str, Any],
) -> dict[str, Any]:
    root = metrics_root / geometry_source / arm / f"seed-{seed}"
    result_path = root / "PER_COMPLEX_METRICS.json"
    receipt_path = root / "RECEIPT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_authority = plan["evaluation_roles"][geometry_source]
    if (
        receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("status") != "PASS_FROZEN_DEVELOPMENT_CHECKPOINT_REEVALUATION"
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != seed
        or receipt.get("validation_complexes") != 549
        or receipt.get("test_loaded") is not False
        or receipt.get("temporal_holdout_loaded") is not False
        or result.get("status") != receipt["status"]
        or result.get("geometry_source") != geometry_source
        or result.get("arm") != arm
        or int(result.get("seed", -1)) != seed
        or result.get("plan_sha256") != PLAN_SHA256
        or result.get("selection_or_tuning") is not False
        or result["authority"].get("test_loaded") is not False
        or result["authority"].get("external_test_loaded") is not False
        or result["authority"].get("structure_manifest_sha256")
        != expected_authority["structure_manifest_sha256"]
        or result["authority"].get("split_manifest_sha256")
        != expected_authority["split_sha256"]
        or len(result.get("per_complex", {})) != 549
    ):
        raise ValueError(f"metric authority differs: {root}")
    return result


def cluster_weighted_mean(
    values: np.ndarray, cluster_ids: np.ndarray
) -> dict[str, Any]:
    clusters = np.unique(cluster_ids)
    cluster_means = np.asarray(
        [values[cluster_ids == cluster].mean() for cluster in clusters],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        samples[replicate] = rng.choice(
            cluster_means, size=len(cluster_means), replace=True
        ).mean()
    return {
        "estimate": float(cluster_means.mean()),
        "cluster_count": int(len(clusters)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "percentile_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mechanism-analysis", type=Path, required=True)
    parser.add_argument("--mechanism-receipt", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if sha256_file(args.plan) != PLAN_SHA256:
        raise ValueError("frozen association plan SHA-256 differs")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_BEFORE_PER_COMPLEX_REEVALUATION"
        or plan.get("geometry_mechanism_analysis_sha256")
        != MECHANISM_ANALYSIS_SHA256
        or plan.get("old_sabdab2_test_loaded") is not False
        or plan.get("temporal_holdout_loaded") is not False
        or plan.get("selection_or_tuning") is not False
        or tuple(plan.get("seeds", ())) != SEEDS
    ):
        raise ValueError("frozen association plan content differs")
    if sha256_file(args.mechanism_analysis) != MECHANISM_ANALYSIS_SHA256:
        raise ValueError("mechanism analysis SHA-256 differs")
    mechanism = json.loads(args.mechanism_analysis.read_text(encoding="utf-8"))
    mechanism_receipt = json.loads(args.mechanism_receipt.read_text(encoding="utf-8"))
    if (
        mechanism_receipt.get("analysis_sha256") != MECHANISM_ANALYSIS_SHA256
        or mechanism_receipt.get("test_loaded") is not False
        or mechanism_receipt.get("temporal_holdout_loaded") is not False
        or mechanism.get("old_sabdab2_test_loaded") is not False
        or mechanism.get("temporal_holdout_loaded") is not False
        or mechanism.get("selection_or_tuning_from_this_analysis") is not False
    ):
        raise ValueError("mechanism analysis authority differs")

    metrics = {
        geometry_source: {
            arm: {
                seed: load_metric_result(
                    args.metrics_root, geometry_source, arm, seed, plan
                )
                for seed in SEEDS
            }
            for arm in ARMS
        }
        for geometry_source in GEOMETRY_SOURCES
    }
    mechanism_rows = {
        str(row["instance_id"]): row for row in mechanism["per_complex"]
    }
    if len(mechanism_rows) != 549:
        raise ValueError("mechanism validation instances differ")
    expected_instances = set(mechanism_rows)
    for geometry_source in GEOMETRY_SOURCES:
        for arm in ARMS:
            for seed in SEEDS:
                if set(metrics[geometry_source][arm][seed]["per_complex"]) != expected_instances:
                    raise ValueError("per-complex metric instance set differs")

    rows: list[dict[str, Any]] = []
    for instance_id, geometry_row in mechanism_rows.items():
        task_deltas: dict[str, Any] = {}
        for task, arm in TASK_ARM.items():
            by_seed: dict[str, float] = {}
            for seed in SEEDS:
                predicted = metrics["predicted"][arm][seed]["per_complex"][instance_id][task]
                bound = metrics["bound"][arm][seed]["per_complex"][instance_id][task]
                if (
                    predicted["candidate_count"] != bound["candidate_count"]
                    or predicted["positive_count"] != bound["positive_count"]
                ):
                    raise ValueError(f"task label axis differs: {instance_id} {task}")
                by_seed[str(seed)] = finite(predicted["average_precision"]) - finite(
                    bound["average_precision"]
                )
            task_deltas[task] = {
                "predicted_minus_bound_by_seed": by_seed,
                "predicted_minus_bound_seed_mean": float(np.mean(list(by_seed.values()))),
            }
        rows.append(
            {
                "instance_id": instance_id,
                "antigen_cluster_id": geometry_row["antigen_cluster_id"],
                "predictors": {
                    "weighted_ca_rmsd_angstrom": geometry_row[
                        "weighted_ca_rmsd_angstrom"
                    ],
                    "combined_graph_jaccard_excluding_self": geometry_row[
                        "combined_graph_jaccard_excluding_self"
                    ],
                    "antibody_ca_rmsd_angstrom": geometry_row["geometry"][
                        "antibody"
                    ]["ca_rmsd_angstrom"],
                    "antigen_ca_rmsd_angstrom": geometry_row["geometry"]["antigen"][
                        "ca_rmsd_angstrom"
                    ],
                    "antibody_af3_ranking_score": geometry_row["af3_ranking_score"][
                        "antibody"
                    ],
                    "antigen_af3_ranking_score": geometry_row["af3_ranking_score"][
                        "antigen"
                    ],
                },
                "task_average_precision_delta": task_deltas,
            }
        )

    cluster_ids = np.asarray([row["antigen_cluster_id"] for row in rows])
    predictor_names = tuple(plan["predictors"])
    predictors = {
        name: np.asarray([finite(row["predictors"][name]) for row in rows])
        for name in predictor_names
    }
    outcomes = {
        task: np.asarray(
            [
                finite(
                    row["task_average_precision_delta"][task][
                        "predicted_minus_bound_seed_mean"
                    ]
                )
                for row in rows
            ]
        )
        for task in TASK_ARM
    }
    relationships = {
        predictor_name: {
            task: _cluster_bootstrap_spearman(
                predictor_values, outcome_values, cluster_ids
            )
            for task, outcome_values in outcomes.items()
        }
        for predictor_name, predictor_values in predictors.items()
    }
    quartile_summaries = {
        predictor_name: {
            task: _quartiles(predictor_values, outcome_values)
            for task, outcome_values in outcomes.items()
        }
        for predictor_name, predictor_values in predictors.items()
    }
    outcome_summaries = {
        task: {
            "complex_weighted_mean": float(values.mean()),
            "complex_weighted_median": float(np.median(values)),
            "cluster_weighted_mean": cluster_weighted_mean(values, cluster_ids),
        }
        for task, values in outcomes.items()
    }
    metric_authorities = {
        geometry_source: {
            arm: {
                str(seed): {
                    "result_sha256": sha256_file(
                        args.metrics_root
                        / geometry_source
                        / arm
                        / f"seed-{seed}"
                        / "PER_COMPLEX_METRICS.json"
                    ),
                    "receipt_sha256": sha256_file(
                        args.metrics_root
                        / geometry_source
                        / arm
                        / f"seed-{seed}"
                        / "RECEIPT.json"
                    ),
                }
                for seed in SEEDS
            }
            for arm in ARMS
        }
        for geometry_source in GEOMETRY_SOURCES
    }
    payload = {
        "format_version": "sabdab2_predicted_geometry_task_association_analysis_v1",
        "status": "PASS_DEVELOPMENT_ONLY_PAIRED_GEOMETRY_TASK_ASSOCIATION",
        "authority": {
            "plan_sha256": PLAN_SHA256,
            "mechanism_analysis_sha256": MECHANISM_ANALYSIS_SHA256,
            "mechanism_receipt_sha256": sha256_file(args.mechanism_receipt),
            "metric_exports": metric_authorities,
        },
        "counts": {
            "validation_complexes": len(rows),
            "antigen_clusters": int(len(np.unique(cluster_ids))),
            "seeds": len(SEEDS),
            "checkpoint_evaluations": 12,
        },
        "routing": plan["routing"],
        "outcome_summaries": outcome_summaries,
        "relationships": relationships,
        "quartile_summaries": quartile_summaries,
        "per_complex": rows,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "causal_interpretation_authorized": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_analysis": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_predicted_geometry_task_association_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": sha256_file(args.output),
        "all_metric_result_sha256_verified": True,
        "all_metric_receipt_sha256_verified": True,
        "validation_complexes": len(rows),
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": sha256_file(args.receipt_output),
                "outcome_summaries": outcome_summaries,
                "relationships": relationships,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
