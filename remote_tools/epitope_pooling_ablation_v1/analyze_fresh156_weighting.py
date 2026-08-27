#!/usr/bin/env python3
"""Explain the fresh156 complex- versus cluster-weighted paired estimates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SEEDS = (17, 29, 43)


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


def load_epitope(root: Path, arm: str, seed: int) -> dict[str, float]:
    path = root / f"component-results/{arm}/seed-{seed}/FRESH156_RESULT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["seed"] != seed or payload["arm"] != arm:
        raise ValueError(f"result identity differs: {path}")
    if payload["audit"]["test_loaded"] is not False:
        raise ValueError(f"test access differs: {path}")
    return {
        instance: float(metrics["epitope"])
        for instance, metrics in payload["per_complex_auprc"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.campaign_root / "authority/FRESH156_PARTNER_METADATA.csv"
    metadata = {
        row["instance_id"]: row
        for row in csv.DictReader(metadata_path.open(encoding="utf-8"))
    }
    per_instance: dict[str, list[float]] = defaultdict(list)
    per_seed_complex_mean: dict[str, float] = {}
    inputs: dict[str, str] = {str(metadata_path): sha256_file(metadata_path)}
    for seed in SEEDS:
        baseline = load_epitope(args.campaign_root, "baseline_current", seed)
        expert = load_epitope(args.campaign_root, "edge_topk2_supervised", seed)
        if set(baseline) != set(expert) or len(baseline) != 156:
            raise ValueError(f"seed {seed} instance coverage differs")
        deltas = {key: expert[key] - baseline[key] for key in baseline}
        per_seed_complex_mean[str(seed)] = statistics.fmean(deltas.values())
        for instance, delta in deltas.items():
            per_instance[instance].append(delta)
        for arm in ("baseline_current", "edge_topk2_supervised"):
            path = (
                args.campaign_root
                / f"component-results/{arm}/seed-{seed}/FRESH156_RESULT.json"
            )
            inputs[str(path)] = sha256_file(path)

    seed_averaged = {
        instance: statistics.fmean(deltas)
        for instance, deltas in per_instance.items()
    }
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for instance, delta in seed_averaged.items():
        row = metadata[instance]
        if row["role"] != "validation":
            raise ValueError(f"fresh156 role differs: {instance}")
        by_cluster[row["antigen_cluster_id"]].append(delta)
    cluster_rows = [
        {
            "antigen_cluster_id": cluster,
            "complex_count": len(values),
            "seed_averaged_mean_delta": statistics.fmean(values),
        }
        for cluster, values in sorted(by_cluster.items())
    ]
    if len(cluster_rows) != 49:
        raise ValueError("fresh156 antigen-cluster count differs")

    complex_weighted_seed_averaged = statistics.fmean(seed_averaged.values())
    cluster_equal_seed_averaged = statistics.fmean(
        row["seed_averaged_mean_delta"] for row in cluster_rows
    )
    mean_size = statistics.fmean(row["complex_count"] for row in cluster_rows)
    mean_gain = cluster_equal_seed_averaged
    covariance = statistics.fmean(
        (row["complex_count"] - mean_size)
        * (row["seed_averaged_mean_delta"] - mean_gain)
        for row in cluster_rows
    )
    variance_size = statistics.fmean(
        (row["complex_count"] - mean_size) ** 2 for row in cluster_rows
    )
    variance_gain = statistics.fmean(
        (row["seed_averaged_mean_delta"] - mean_gain) ** 2
        for row in cluster_rows
    )
    correlation = covariance / math.sqrt(variance_size * variance_gain)

    size_buckets = {}
    for name, predicate in (
        ("singleton", lambda n: n == 1),
        ("two_to_three", lambda n: 2 <= n <= 3),
        ("four_or_more", lambda n: n >= 4),
    ):
        selected = [
            row["seed_averaged_mean_delta"]
            for row in cluster_rows
            if predicate(row["complex_count"])
        ]
        size_buckets[name] = {
            "cluster_count": len(selected),
            "equal_cluster_mean_delta": statistics.fmean(selected),
        }

    payload = {
        "format_version": "fresh156_weighting_explanation_v1",
        "status": "PASS_REPRODUCED_FROZEN_ESTIMATES",
        "claim_boundary": {
            "conditional_on_fixed_seeds": list(SEEDS),
            "selection_performed": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "counts": {
            "complexes": len(seed_averaged),
            "antigen_clusters": len(cluster_rows),
            "cluster_size_distribution": dict(
                sorted(Counter(row["complex_count"] for row in cluster_rows).items())
            ),
        },
        "estimates": {
            "per_seed_complex_weighted_mean_delta": per_seed_complex_mean,
            "median_of_per_seed_complex_weighted_means": statistics.median(
                per_seed_complex_mean.values()
            ),
            "seed_averaged_complex_weighted_mean_delta": complex_weighted_seed_averaged,
            "seed_averaged_cluster_equal_mean_delta": cluster_equal_seed_averaged,
            "cluster_minus_complex_weighted_difference": (
                cluster_equal_seed_averaged - complex_weighted_seed_averaged
            ),
        },
        "diagnostics": {
            "cluster_size_gain_covariance": covariance,
            "cluster_size_gain_correlation": correlation,
            "size_buckets": size_buckets,
            "largest_clusters": sorted(
                cluster_rows,
                key=lambda row: (-row["complex_count"], row["antigen_cluster_id"]),
            )[:10],
        },
        "inputs": inputs,
    }
    encoded = canonical_bytes(payload)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
