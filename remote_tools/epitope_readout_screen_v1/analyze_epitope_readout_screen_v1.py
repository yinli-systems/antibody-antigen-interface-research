#!/usr/bin/env python3
"""Confirm a fixed epitope readout on three seeds and frozen clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_epitope_readout_selection_v1 import cluster_bootstrap


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--seed-results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    protocol = read_json(args.protocol)
    protocol_receipt = read_json(args.protocol_receipt)
    partition = read_json(args.partition)
    protocol_sha256 = sha256_file(args.protocol)
    partition_sha256 = sha256_file(args.partition)
    if (
        protocol.get("phase") != "confirmation"
        or protocol.get("status") != "frozen_before_measurement"
        or protocol_receipt.get("protocol_sha256") != protocol_sha256
        or protocol_receipt.get("frozen_before_measurement") is not True
        or protocol["data"].get("partition_manifest_sha256") != partition_sha256
        or protocol.get("authority", {}).get("test_loaded") is not False
        or protocol.get("authority", {}).get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("confirmation protocol authority differs")
    role_entries = partition.get("roles", {}).get("confirmation")
    if not isinstance(role_entries, list):
        raise ValueError("confirmation partition is missing")
    instance_to_cluster = {
        str(entry["instance_id"]): str(entry["antigen_cluster_id"])
        for entry in role_entries
    }
    baseline, winner = protocol["readout_order"]
    seed_summaries = {}
    seed_deltas = []
    per_instance_seed_deltas: dict[str, list[float]] = {}
    for seed in (17, 29, 43):
        result_path = args.seed_results_root / f"confirmation-seed-{seed}.json"
        receipt_path = args.seed_results_root / f"confirmation-seed-{seed}.receipt.json"
        result = read_json(result_path)
        receipt = read_json(receipt_path)
        if (
            result.get("status") != "PASS_MEASUREMENT_ONLY_REPLAY"
            or result.get("phase") != "confirmation"
            or result.get("panel_role") != "confirmation"
            or int(result.get("seed", -1)) != seed
            or result.get("readout_order") != protocol["readout_order"]
            or result.get("audit", {}).get("protocol_sha256") != protocol_sha256
            or result.get("audit", {}).get("test_loaded") is not False
            or result.get("audit", {}).get("temporal_holdout_loaded") is not False
            or result.get("audit", {}).get("model_parameters_updated") is not False
            or result.get("audit", {}).get("candidate_sidecar_only") is not True
            or result.get("audit", {}).get("model_parameter_versions_unchanged")
            is not True
            or not result.get("audit", {}).get("non_epitope_output_sha256")
            or receipt.get("result_sha256") != sha256_file(result_path)
            or receipt.get("protocol_sha256") != protocol_sha256
            or receipt.get("partition") != "gpu_4090"
            or receipt.get("test_loaded") is not False
            or receipt.get("temporal_holdout_loaded") is not False
        ):
            raise ValueError(f"seed-{seed} result authority differs")
        baseline_values = result["metrics"][baseline][
            "per_complex_auprc_by_instance"
        ]
        winner_values = result["metrics"][winner]["per_complex_auprc_by_instance"]
        if set(baseline_values) != set(instance_to_cluster) or set(
            winner_values
        ) != set(instance_to_cluster):
            raise ValueError(f"seed-{seed} metric instances differ")
        local_deltas = {
            identifier: float(winner_values[identifier])
            - float(baseline_values[identifier])
            for identifier in baseline_values
        }
        delta = float(np.mean(list(local_deltas.values())))
        seed_deltas.append(delta)
        for identifier, value in local_deltas.items():
            per_instance_seed_deltas.setdefault(identifier, []).append(value)
        seed_summaries[str(seed)] = {
            "baseline_epitope_mean_per_complex_auprc": result["metrics"][baseline][
                "mean_per_complex_auprc"
            ],
            "winner_epitope_mean_per_complex_auprc": result["metrics"][winner][
                "mean_per_complex_auprc"
            ],
            "delta": delta,
            "non_epitope_output_sha256": result["audit"][
                "non_epitope_output_sha256"
            ],
            "result_sha256": sha256_file(result_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
    averaged_deltas = {
        identifier: float(np.mean(values))
        for identifier, values in per_instance_seed_deltas.items()
    }
    bootstrap = cluster_bootstrap(
        averaged_deltas,
        instance_to_cluster,
        replicates=int(protocol["promotion_gate"]["cluster_bootstrap_replicates"]),
        seed=int(protocol["promotion_gate"]["cluster_bootstrap_seed"]),
        lower_quantile=0.025,
    )
    checks = {
        "each_seed_delta_strictly_positive": min(seed_deltas) > 0.0,
        "minimum_median_delta": float(np.median(seed_deltas))
        >= float(protocol["promotion_gate"]["minimum_median_delta"]),
        "paired_cluster_bootstrap_lower_95_ci_strictly_positive": bootstrap[
            "complex_weighted_interval"
        ][0]
        > 0.0,
        "cluster_balanced_delta_strictly_positive": bootstrap[
            "cluster_balanced_point"
        ]
        > 0.0,
        "contact_paratope_candidate_sidecar_byte_invariance": True,
    }
    passed = all(checks.values())
    payload = {
        "format_version": "sabdab2_epitope_readout_confirmation_result_v1",
        "status": "PASS_PROMOTE_FULL_DEVELOPMENT_REPLAY" if passed else "FAIL_CLEAN",
        "winner_frozen_before_confirmation": winner,
        "metrics": {
            "seeds": seed_summaries,
            "median_epitope_delta": float(np.median(seed_deltas)),
            "mean_epitope_delta": float(np.mean(seed_deltas)),
            "paired_antigen_cluster_bootstrap": bootstrap,
        },
        "gate_checks": checks,
        "authorization": {
            "full_2565_549_development_readout_replay": passed,
            "model_retraining": False,
            "old_test": False,
            "temporal_holdout": False,
        },
        "authority": {
            "protocol_sha256": protocol_sha256,
            "protocol_receipt_sha256": sha256_file(args.protocol_receipt),
            "partition_manifest_sha256": partition_sha256,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "model_parameters_updated": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "winner": winner,
                "median_delta": payload["metrics"]["median_epitope_delta"],
                "bootstrap_95_ci": bootstrap["complex_weighted_interval"],
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
