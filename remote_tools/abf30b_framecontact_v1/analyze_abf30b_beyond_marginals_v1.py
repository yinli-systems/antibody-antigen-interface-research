#!/usr/bin/env python3
"""Aggregate the frozen three-seed ABF-30B beyond-marginals development audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


SEEDS = (17, 29, 43)
SCORES = (
    "full_contact",
    "unary_only",
    "residue_head_outer_product",
    "raw_interaction",
    "gauge_fixed_residual",
)
COMPARATORS = SCORES[1:]
METRICS = (
    "average_precision",
    "reciprocal_rank_first_positive",
    "hits_at_1",
    "recall_at_P",
    "recall_at_ceil_P_over_2",
    "recall_at_ceil_P_over_5",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def grouped_estimand(
    values: dict[str, float],
    group_by_instance: dict[str, str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for instance_id, value in values.items():
        grouped.setdefault(group_by_instance[instance_id], []).append(finite(value))
    groups = sorted(grouped)
    means = np.array([np.mean(grouped[group]) for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(means), size=len(means))
        draws[index] = float(means[sampled].mean())
    return {
        "point_estimate": float(means.mean()),
        "confidence_interval_95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "group_count": len(groups),
        "complex_count": len(values),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    protocol_sha = sha256_file(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    protocol_receipt = json.loads(args.protocol_receipt.read_text())
    if (
        protocol.get("status") != "FROZEN_BEFORE_READ_ONLY_DEVELOPMENT_COMPONENT_AUDIT"
        or protocol_receipt.get("protocol_sha256") != protocol_sha
        or protocol.get("tools", {}).get("analyzer_sha256") != sha256_file(Path(__file__).resolve())
        or protocol.get("estimand", {}).get("conditional_on_fixed_trained_seeds") is not True
    ):
        raise ValueError("frozen audit protocol differs")

    components: dict[int, dict[str, Any]] = {}
    component_bindings: dict[str, Any] = {}
    for seed in SEEDS:
        result_path = args.component_root / f"seed-{seed}/ABF30B_BEYOND_MARGINALS_COMPONENT_V1.json"
        receipt_path = args.component_root / f"seed-{seed}/ABF30B_BEYOND_MARGINALS_COMPONENT_V1_RECEIPT.json"
        if any(path.is_symlink() or not path.is_file() for path in (result_path, receipt_path)):
            raise ValueError(f"component artifact missing for seed {seed}")
        result = json.loads(result_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        result_sha = sha256_file(result_path)
        if (
            result.get("status") != "PASS_READ_ONLY_DEVELOPMENT_COMPONENT_MEASUREMENT"
            or result.get("seed") != seed
            or result.get("protocol_sha256") != protocol_sha
            or result.get("checkpoint_sha256")
            != protocol["run_bindings"][str(seed)]["checkpoint_sha256"]
            or result.get("counts", {}).get("complexes") != protocol["data"]["instance_count"]
            or result.get("authority", {}).get("model_training") is not False
            or result.get("authority", {}).get("checkpoint_selection") is not False
            or result.get("authority", {}).get("test_loaded") is not False
            or result.get("authority", {}).get("temporal_holdout_loaded") is not False
            or result.get("authority", {}).get("sable_external_labels_loaded") is not False
            or result.get("authority", {}).get("raw_pair_predictions_persisted") is not False
            or result.get("authority", {}).get("model_state_sha256_before")
            != result.get("authority", {}).get("model_state_sha256_after")
            or receipt.get("result_sha256") != result_sha
            or receipt.get("seed") != seed
            or receipt.get("protocol_sha256") != protocol_sha
        ):
            raise ValueError(f"component authority differs for seed {seed}")
        components[seed] = result
        component_bindings[str(seed)] = {
            "result_sha256": result_sha,
            "receipt_sha256": sha256_file(receipt_path),
            "checkpoint_sha256": result["checkpoint_sha256"],
        }

    instance_sets = [set(value["per_instance"]) for value in components.values()]
    if any(ids != instance_sets[0] for ids in instance_sets[1:]) or len(instance_sets[0]) != 549:
        raise ValueError("component instance sets differ")
    instance_ids = sorted(instance_sets[0])
    cluster_by_instance: dict[str, str] = {}
    pdb_by_instance: dict[str, str] = {}
    for instance_id in instance_ids:
        rows = [components[seed]["per_instance"][instance_id] for seed in SEEDS]
        identity = {
            (
                row["antigen_cluster_id"],
                row["pdb_assembly_id"],
                row["pair_count"],
                row["positive_pair_count"],
            )
            for row in rows
        }
        if len(identity) != 1:
            raise ValueError(f"instance metadata differs across seeds: {instance_id}")
        cluster_by_instance[instance_id] = rows[0]["antigen_cluster_id"]
        pdb_by_instance[instance_id] = rows[0]["pdb_assembly_id"]

    score_summary: dict[str, Any] = {}
    for score in SCORES:
        pooled = [
            finite(components[seed]["score_summary"][score]["pooled_average_precision"])
            for seed in SEEDS
        ]
        mean_complex = [
            finite(
                components[seed]["score_summary"][score][
                    "mean_per_complex_average_precision"
                ]
            )
            for seed in SEEDS
        ]
        score_summary[score] = {
            "pooled_average_precision_by_seed": dict(zip(map(str, SEEDS), pooled)),
            "pooled_average_precision_median": float(median(pooled)),
            "mean_per_complex_average_precision_by_seed": dict(
                zip(map(str, SEEDS), mean_complex)
            ),
            "mean_per_complex_average_precision_median": float(median(mean_complex)),
        }

    comparisons: dict[str, Any] = {}
    replicates = int(protocol["estimand"]["bootstrap_replicates"])
    bootstrap_seed = int(protocol["estimand"]["bootstrap_seed"])
    for comparator_index, comparator in enumerate(COMPARATORS):
        by_metric: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            instance_delta: dict[str, float] = {}
            by_seed: dict[str, float] = {}
            for seed in SEEDS:
                seed_values = []
                for instance_id in instance_ids:
                    row = components[seed]["per_instance"][instance_id]["metrics_by_score"]
                    if row["full_contact"] is None or row[comparator] is None:
                        continue
                    seed_values.append(
                        finite(row["full_contact"][metric]) - finite(row[comparator][metric])
                    )
                by_seed[str(seed)] = float(np.mean(seed_values))
            for instance_id in instance_ids:
                deltas = []
                for seed in SEEDS:
                    row = components[seed]["per_instance"][instance_id]["metrics_by_score"]
                    if row["full_contact"] is None or row[comparator] is None:
                        continue
                    deltas.append(
                        finite(row["full_contact"][metric]) - finite(row[comparator][metric])
                    )
                if deltas:
                    instance_delta[instance_id] = float(np.mean(deltas))
            if not instance_delta:
                raise ValueError(f"no eligible instances for {comparator}/{metric}")
            stream_seed = bootstrap_seed + 1000 * comparator_index + metric_index
            by_metric[metric] = {
                "unweighted_mean_delta_by_seed": by_seed,
                "antigen_cluster_equal_weighted": grouped_estimand(
                    instance_delta,
                    cluster_by_instance,
                    replicates=replicates,
                    seed=stream_seed,
                ),
                "pdb_assembly_equal_weighted": grouped_estimand(
                    instance_delta,
                    pdb_by_instance,
                    replicates=replicates,
                    seed=stream_seed + 500000,
                ),
            }
        comparisons[f"full_contact_minus_{comparator}"] = by_metric

    gauge_by_seed = {
        str(seed): {
            "mean_partner_specific_variance_share": finite(
                components[seed]["gauge_summary"]["mean_partner_specific_variance_share"]
            ),
            "median_partner_specific_variance_share": finite(
                components[seed]["gauge_summary"]["median_partner_specific_variance_share"]
            ),
            "maximum_reconstruction_error": finite(
                components[seed]["gauge_summary"]["maximum_reconstruction_error"]
            ),
            "maximum_centering_error": finite(
                components[seed]["gauge_summary"]["maximum_centering_error"]
            ),
            "maximum_variance_partition_relative_error": finite(
                components[seed]["gauge_summary"][
                    "maximum_variance_partition_relative_error"
                ]
            ),
        }
        for seed in SEEDS
    }
    output = {
        "format_version": "abf30b_beyond_marginals_analysis_v1",
        "status": "PASS_READ_ONLY_DEVELOPMENT_BEYOND_MARGINALS_AUDIT",
        "status_scope": "integrity and protocol completion only",
        "protocol_sha256": protocol_sha,
        "component_bindings": component_bindings,
        "score_summary": score_summary,
        "comparisons": comparisons,
        "gauge_summary_by_seed": gauge_by_seed,
        "counts": {
            "seeds": len(SEEDS),
            "complexes": len(instance_ids),
            "antigen_clusters": len(set(cluster_by_instance.values())),
            "pdb_assemblies": len(set(pdb_by_instance.values())),
            "zero_positive_complexes_by_seed": {
                str(seed): components[seed]["counts"]["zero_positive_complexes"]
                for seed in SEEDS
            },
        },
        "authority": {
            "development_only": True,
            "independent_test": False,
            "model_training": False,
            "checkpoint_selection": False,
            "calibration_or_threshold_fitting": False,
            "conditional_on_fixed_trained_seeds": True,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "sable_external_labels_loaded": False,
            "raw_pair_predictions_persisted": False,
        },
    }
    analysis_sha = publish(args.output, output)
    receipt_sha = publish(
        args.receipt_output,
        {
            "format_version": "abf30b_beyond_marginals_analysis_receipt_v1",
            "status": output["status"],
            "analysis_sha256": analysis_sha,
            "protocol_sha256": protocol_sha,
            "component_result_sha256": {
                seed: value["result_sha256"] for seed, value in component_bindings.items()
            },
            "development_only": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "sable_external_labels_loaded": False,
        },
    )
    print(json.dumps({"analysis_sha256": analysis_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
