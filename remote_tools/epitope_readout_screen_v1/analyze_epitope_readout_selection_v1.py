#!/usr/bin/env python3
"""Select one readout on the frozen antigen-cluster selection panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def cluster_bootstrap(
    deltas: dict[str, float],
    instance_to_cluster: dict[str, str],
    *,
    replicates: int,
    seed: int,
    lower_quantile: float,
) -> dict[str, object]:
    cluster_values: dict[str, list[float]] = {}
    for identifier, delta in deltas.items():
        cluster_values.setdefault(instance_to_cluster[identifier], []).append(delta)
    clusters = sorted(cluster_values)
    cluster_sums = np.asarray(
        [np.sum(cluster_values[cluster]) for cluster in clusters], dtype=np.float64
    )
    cluster_sizes = np.asarray(
        [len(cluster_values[cluster]) for cluster in clusters], dtype=np.float64
    )
    cluster_means = cluster_sums / cluster_sizes
    generator = np.random.default_rng(seed)
    complex_weighted = np.empty(replicates, dtype=np.float64)
    cluster_balanced = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = generator.integers(0, len(clusters), size=len(clusters))
        complex_weighted[index] = cluster_sums[sample].sum() / cluster_sizes[
            sample
        ].sum()
        cluster_balanced[index] = cluster_means[sample].mean()
    return {
        "cluster_count": len(clusters),
        "complex_weighted_point": float(np.mean(list(deltas.values()))),
        "cluster_balanced_point": float(cluster_means.mean()),
        "complex_weighted_interval": [
            float(np.quantile(complex_weighted, lower_quantile)),
            float(np.quantile(complex_weighted, 1.0 - lower_quantile)),
        ],
        "cluster_balanced_interval": [
            float(np.quantile(cluster_balanced, lower_quantile)),
            float(np.quantile(cluster_balanced, 1.0 - lower_quantile)),
        ],
        "replicates": replicates,
        "bootstrap_seed": seed,
        "lower_quantile": lower_quantile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--confirmation-split", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-receipt", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--confirmation-protocol-output", type=Path, required=True)
    parser.add_argument("--confirmation-receipt-output", type=Path, required=True)
    args = parser.parse_args()

    protocol = read_json(args.protocol)
    protocol_receipt = read_json(args.protocol_receipt)
    partition = read_json(args.partition)
    result = read_json(args.result)
    result_receipt = read_json(args.result_receipt)
    protocol_sha256 = sha256_file(args.protocol)
    partition_sha256 = sha256_file(args.partition)
    if (
        protocol.get("phase") != "selection"
        or protocol.get("status") != "frozen_before_measurement"
        or protocol_receipt.get("protocol_sha256") != protocol_sha256
        or protocol_receipt.get("frozen_before_measurement") is not True
        or protocol["data"].get("partition_manifest_sha256") != partition_sha256
        or result.get("status") != "PASS_MEASUREMENT_ONLY_REPLAY"
        or result.get("phase") != "selection"
        or result.get("panel_role") != "selection"
        or int(result.get("seed", -1)) != 17
        or result.get("readout_order") != protocol.get("readout_order")
        or result.get("audit", {}).get("protocol_sha256") != protocol_sha256
        or result.get("audit", {}).get("test_loaded") is not False
        or result.get("audit", {}).get("temporal_holdout_loaded") is not False
        or result.get("audit", {}).get("model_parameters_updated") is not False
        or result.get("audit", {}).get("candidate_sidecar_only") is not True
        or result.get("audit", {}).get("model_parameter_versions_unchanged") is not True
        or result_receipt.get("result_sha256") != sha256_file(args.result)
        or result_receipt.get("protocol_sha256") != protocol_sha256
        or result_receipt.get("partition") != "gpu_4090"
        or result_receipt.get("test_loaded") is not False
        or result_receipt.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("selection authority differs")

    role_entries = partition.get("roles", {}).get("selection")
    if not isinstance(role_entries, list):
        raise ValueError("selection partition is missing")
    instance_to_cluster = {
        str(entry["instance_id"]): str(entry["antigen_cluster_id"])
        for entry in role_entries
    }
    baseline_name = protocol["selection"]["baseline"]
    baseline = result["metrics"][baseline_name]["per_complex_auprc_by_instance"]
    if set(baseline) != set(instance_to_cluster):
        raise ValueError("selection metric instances differ from partition")
    candidate_summaries = {}
    passing = []
    for candidate in protocol["selection"]["candidates"]:
        candidate_values = result["metrics"][candidate][
            "per_complex_auprc_by_instance"
        ]
        if set(candidate_values) != set(baseline):
            raise ValueError(f"{candidate}: metric instances differ")
        deltas = {
            identifier: float(candidate_values[identifier]) - float(baseline[identifier])
            for identifier in baseline
        }
        bootstrap = cluster_bootstrap(
            deltas,
            instance_to_cluster,
            replicates=int(protocol["selection"]["cluster_bootstrap_replicates"]),
            seed=int(protocol["selection"]["cluster_bootstrap_seed"]),
            lower_quantile=float(
                protocol["selection"]["bonferroni_one_sided_lower_quantile"]
            ),
        )
        checks = {
            "minimum_complex_weighted_delta": bootstrap["complex_weighted_point"]
            >= float(protocol["selection"]["minimum_complex_weighted_delta"]),
            "bonferroni_cluster_bootstrap_lower_positive": bootstrap[
                "complex_weighted_interval"
            ][0]
            > 0.0,
            "cluster_balanced_direction_positive": bootstrap[
                "cluster_balanced_point"
            ]
            > 0.0,
        }
        candidate_summaries[candidate] = {
            "baseline_mean_per_complex_auprc": result["metrics"][baseline_name][
                "mean_per_complex_auprc"
            ],
            "candidate_mean_per_complex_auprc": result["metrics"][candidate][
                "mean_per_complex_auprc"
            ],
            "bootstrap": bootstrap,
            "checks": checks,
            "passes": all(checks.values()),
        }
        if all(checks.values()):
            passing.append(candidate)

    preference = protocol["selection"]["tie_preference"]
    winner = None
    if passing:
        winner = max(
            passing,
            key=lambda name: (
                candidate_summaries[name]["bootstrap"]["complex_weighted_point"],
                -preference.index(name),
            ),
        )
    selection_payload = {
        "format_version": "sabdab2_epitope_readout_selection_result_v1",
        "status": "PASS_AUTHORIZE_CONFIRMATION" if winner else "FAIL_CLEAN",
        "winner": winner,
        "candidates": candidate_summaries,
        "authorization": {
            "confirmation_panel": winner is not None,
            "full_development": False,
            "old_test": False,
            "temporal_holdout": False,
        },
        "authority": {
            "protocol_sha256": protocol_sha256,
            "protocol_receipt_sha256": sha256_file(args.protocol_receipt),
            "partition_manifest_sha256": partition_sha256,
            "selection_result_sha256": sha256_file(args.result),
            "selection_result_receipt_sha256": sha256_file(args.result_receipt),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    selection_sha256 = write_new(args.analysis_output, selection_payload)
    if winner is not None:
        confirmation_protocol = {
            "format_version": "sabdab2_epitope_readout_protocol_v1",
            "phase": "confirmation",
            "status": "frozen_before_measurement",
            "research_basis": protocol["research_basis"],
            "data": {
                **protocol["data"],
                "panel_role": "confirmation",
                "role_split_sha256": sha256_file(args.confirmation_split),
                "complex_count": partition["counts"]["confirmation"]["complexes"],
                "antigen_cluster_count": partition["counts"]["confirmation"][
                    "antigen_clusters"
                ],
            },
            "source": protocol["source"],
            "runs": protocol["runs"],
            "readout_order": [baseline_name, winner],
            "selection": {
                "winner": winner,
                "selection_result_sha256": selection_sha256,
                "selection_protocol_sha256": protocol_sha256,
                "confirmation_cannot_change_winner": True,
            },
            "promotion_gate": {
                "each_seed_delta_strictly_positive": True,
                "minimum_median_delta": 0.005,
                "paired_cluster_bootstrap_lower_95_ci_strictly_positive": True,
                "cluster_balanced_delta_strictly_positive": True,
                "cluster_bootstrap_replicates": 50000,
                "cluster_bootstrap_seed": 20260826,
                "next_scope": "full_2565_549_development_readout_replay_only",
            },
            "authority": {
                "measurement_only": True,
                "model_parameters_updated": False,
                "test_loaded": False,
                "temporal_holdout_loaded": False,
            },
        }
        confirmation_sha256 = write_new(
            args.confirmation_protocol_output, confirmation_protocol
        )
        confirmation_receipt = {
            "format_version": "sabdab2_epitope_readout_protocol_receipt_v1",
            "phase": "confirmation",
            "protocol_sha256": confirmation_sha256,
            "partition_manifest_sha256": partition_sha256,
            "selection_result_sha256": selection_sha256,
            "winner": winner,
            "frozen_before_measurement": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        write_new(args.confirmation_receipt_output, confirmation_receipt)
    print(
        json.dumps(
            {
                "status": selection_payload["status"],
                "winner": winner,
                "selection_result_sha256": selection_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
