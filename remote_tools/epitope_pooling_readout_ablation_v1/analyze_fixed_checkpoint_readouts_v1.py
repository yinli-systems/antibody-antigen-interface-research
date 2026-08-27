#!/usr/bin/env python3
"""Analyze paired fixed-checkpoint readout diagnostics without model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROTOCOL_SHA256 = "b361bcf437fc73e9611b1564cdec06fbc1f4da633066753cef6276755b3eb928"
RESULT_SHA256 = "848a548a3a041fe2c97e8522d2542bbc5a93b8e7e75fe95a6634643b12ac90fc"
RECEIPT_SHA256 = "115db8ca56dbcbad1d3728d0518b7077d56e1cd7a2b83882726c1a0ecf545f49"
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 10_000


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


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = (
        args.campaign_root / "authority/PROTOCOL_FIXED_CHECKPOINT_EPITOPE_READOUT_V1.json"
    )
    result_path = args.campaign_root / "results/FIXED_CHECKPOINT_EPITOPE_READOUT_RESULT_V1.json"
    receipt_path = args.campaign_root / "results/FIXED_CHECKPOINT_EPITOPE_READOUT_RECEIPT_V1.json"
    output_path = args.campaign_root / "FIXED_CHECKPOINT_EPITOPE_READOUT_ANALYSIS_V1.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    for path, expected in (
        (protocol_path, PROTOCOL_SHA256),
        (result_path, RESULT_SHA256),
        (receipt_path, RECEIPT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen artifact differs: {path}")
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("result_sha256") != RESULT_SHA256
        or result.get("status")
        != "PASS_FIXED_CHECKPOINT_DETERMINISTIC_READOUT_DIAGNOSTIC"
        or result.get("audit", {}).get("model_parameters_updated") is not False
        or result.get("audit", {}).get("model_selection_performed") is not False
        or result.get("audit", {}).get("test_loaded") is not False
        or result.get("audit", {}).get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("result claim boundary differs")

    split_path = Path(protocol["paths"]["split_manifest"])
    if sha256_file(split_path) != protocol["sha256"]["split_manifest"]:
        raise ValueError("split differs")
    with split_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    cluster_by_instance = {
        row["instance_id"]: row["antigen_cluster_id"]
        for row in rows
        if row["role"] == "validation"
    }
    per_complex = result["per_complex_average_precision"]
    if set(per_complex) != set(cluster_by_instance) or len(per_complex) != 549:
        raise ValueError("validation identity set differs")

    arms = tuple(protocol["readouts"])
    reference = "topk2"
    identifiers = tuple(sorted(per_complex))
    clusters = tuple(sorted(set(cluster_by_instance.values())))
    cluster_members = {
        cluster: tuple(
            identifier
            for identifier in identifiers
            if cluster_by_instance[identifier] == cluster
        )
        for cluster in clusters
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = {}
    for arm in arms:
        differences = {
            identifier: float(
                per_complex[identifier][arm] - per_complex[identifier][reference]
            )
            for identifier in identifiers
        }
        complex_values = np.asarray(list(differences.values()), dtype=np.float64)
        cluster_values = np.asarray(
            [
                np.mean([differences[identifier] for identifier in cluster_members[cluster]])
                for cluster in clusters
            ],
            dtype=np.float64,
        )
        complex_draws = rng.choice(
            complex_values,
            size=(BOOTSTRAP_REPLICATES, complex_values.size),
            replace=True,
        ).mean(axis=1)
        cluster_draws = rng.choice(
            cluster_values,
            size=(BOOTSTRAP_REPLICATES, cluster_values.size),
            replace=True,
        ).mean(axis=1)
        comparisons[arm] = {
            "reference": reference,
            "complex_weighted_mean_delta": float(complex_values.mean()),
            "complex_bootstrap_95_percentile_interval": interval(complex_draws),
            "equal_antigen_cluster_weighted_mean_delta": float(cluster_values.mean()),
            "antigen_cluster_bootstrap_95_percentile_interval": interval(cluster_draws),
            "positive_complex_count": int((complex_values > 0).sum()),
            "negative_complex_count": int((complex_values < 0).sum()),
            "tie_complex_count": int((complex_values == 0).sum()),
        }

    payload = {
        "format_version": "sabdab2_fixed_checkpoint_epitope_readout_analysis_v1",
        "status": "PASS_PAIRED_FIXED_CHECKPOINT_READOUT_ANALYSIS",
        "reference": reference,
        "metrics": result["metrics"],
        "paired_comparisons": comparisons,
        "sampling": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "validation_complexes": len(identifiers),
            "antigen_clusters": len(clusters),
        },
        "audit": {
            "protocol_sha256": PROTOCOL_SHA256,
            "result_sha256": RESULT_SHA256,
            "receipt_sha256": RECEIPT_SHA256,
            "seed": 17,
            "selected_checkpoint_fixed": True,
            "model_parameters_updated": False,
            "model_selection_performed": False,
            "learned_pooling_evaluated": False,
            "development_only": True,
            "full_validation_is_checkpoint_selection_data": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    output_path.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "analysis_sha256": sha256_file(output_path),
                "comparisons": comparisons,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
