#!/usr/bin/env python3
"""Audit and summarize the once-only temporal double-cold measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("baseline_current", "edge_topk2_supervised")
SEEDS = (17, 29, 43)
TASK_ROUTE = {
    "contact": "baseline_current",
    "paratope": "baseline_current",
    "epitope": "edge_topk2_supervised",
}
METRIC_KEYS = (
    "average_precision",
    "auroc",
    "brier",
    "ece_15",
    "precision",
    "recall",
    "f1",
    "mcc",
    "mean_per_eligible_complex_average_precision",
    "eligible_complex_count",
    "total_complex_count",
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260827


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def pdb_block_bootstrap(
    per_seed_ap: dict[int, dict[str, float]],
    pdb_by_instance: dict[str, str],
) -> dict[str, object]:
    instance_sets = [set(values) for values in per_seed_ap.values()]
    if not instance_sets or any(values != instance_sets[0] for values in instance_sets[1:]):
        raise ValueError("eligible instances differ across seeds")
    instances = sorted(instance_sets[0])
    by_pdb: dict[str, list[float]] = {}
    for instance in instances:
        pdb_id = pdb_by_instance[instance]
        seed_mean = float(np.mean([per_seed_ap[seed][instance] for seed in SEEDS]))
        by_pdb.setdefault(pdb_id, []).append(seed_mean)
    pdb_values = np.asarray(
        [np.mean(by_pdb[pdb_id]) for pdb_id in sorted(by_pdb)], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(pdb_values), size=len(pdb_values))
        draws[index] = float(pdb_values[sample].mean())
    return {
        "point_estimate": float(pdb_values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "eligible_instance_count": len(instances),
        "eligible_unique_pdb_count": len(pdb_values),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "weighting": "mean over the three fixed seeds per complex, mean within PDB, then equal weight per PDB",
        "uncertainty_boundary": "conditional on the three fixed trained seeds; retraining uncertainty is not included",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.protocol, args.protocol_sha256, "protocol")
    data_manifest_path = args.data_root / "TEMPORAL_DOUBLE_COLD_DATA_MANIFEST_V1.json"
    require_sha(data_manifest_path, args.data_manifest_sha256, "data manifest")
    protocol = json.loads(args.protocol.read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    if (
        protocol.get("status") != "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_SINGLE_LABEL_OPENING"
        or protocol.get("execution", {}).get("analyzer_sha256") != sha256_file(Path(__file__))
        or data_manifest.get("status") != "PASS_SINGLE_OPENING_TEMPORAL_DATA_MATERIALIZED"
        or data_manifest.get("opening", {}).get("opening_count") != 1
        or data_manifest.get("opening", {}).get("selection_performed_on_temporal") is not False
    ):
        raise ValueError("analysis authority differs")
    pdb_by_instance = {
        str(instance): str(pdb_id)
        for instance, pdb_id in data_manifest["pdb_by_instance"].items()
    }
    components: dict[str, dict[int, dict[str, object]]] = {arm: {} for arm in ARMS}
    bindings: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            result_path = args.components_root / arm / f"seed-{seed}" / "RESULT.json"
            receipt_path = args.components_root / arm / f"seed-{seed}" / "RECEIPT.json"
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            if (
                result.get("status") != "PASS_TEMPORAL_DOUBLE_COLD_COMPONENT_MEASUREMENT"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or result.get("audit", {}).get("protocol_sha256") != args.protocol_sha256
                or result.get("audit", {}).get("data_manifest_sha256") != args.data_manifest_sha256
                or result.get("audit", {}).get("opening_count") != 1
                or result.get("audit", {}).get("calibration_refit") is not False
                or result.get("audit", {}).get("selection_performed") is not False
                or result.get("audit", {}).get("old_exposed_test_access") is not False
                or receipt.get("result_sha256") != sha256_file(result_path)
                or receipt.get("opening_count") != 1
                or receipt.get("selection_performed") is not False
            ):
                raise ValueError(f"component audit failed: {arm}/seed-{seed}")
            components[arm][seed] = result
            bindings[arm][str(seed)] = {
                "result_sha256": sha256_file(result_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_sha256": result["audit"]["checkpoint_sha256"],
                "model_state_sha256": result["audit"]["model_state_sha256"],
            }

    routed_by_task_seed: dict[str, dict[str, object]] = {}
    median_by_task: dict[str, dict[str, object]] = {}
    bootstrap_by_task: dict[str, object] = {}
    for task, arm in TASK_ROUTE.items():
        routed_by_task_seed[task] = {}
        per_seed_ap: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            metric = components[arm][seed]["metrics"][task]
            routed_by_task_seed[task][str(seed)] = {
                key: metric[key] for key in METRIC_KEYS
            }
            per_seed_ap[seed] = {
                str(instance): float(value)
                for instance, value in metric["per_complex_average_precision"].items()
            }
        median_by_task[task] = {
            key: (
                int(np.median([routed_by_task_seed[task][str(seed)][key] for seed in SEEDS]))
                if key in {"eligible_complex_count", "total_complex_count"}
                else float(np.median([routed_by_task_seed[task][str(seed)][key] for seed in SEEDS]))
            )
            for key in METRIC_KEYS
            if all(routed_by_task_seed[task][str(seed)][key] is not None for seed in SEEDS)
        }
        bootstrap_by_task[task] = pdb_block_bootstrap(per_seed_ap, pdb_by_instance)

    payload = {
        "format_version": "framecontact_temporal_double_cold_measurement_result_v1",
        "status": "PASS_ONCE_ONLY_TEMPORAL_DOUBLE_COLD_MEASUREMENT_COMPLETE",
        "protocol_sha256": args.protocol_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "task_routing": TASK_ROUTE,
        "routed_metrics_by_task_seed": routed_by_task_seed,
        "median_across_three_seeds": median_by_task,
        "unique_pdb_weighted_bootstrap": bootstrap_by_task,
        "component_bindings": bindings,
        "evidence_boundary": {
            "prospective_relative_to_training_snapshot": True,
            "single_opening": True,
            "opening_count": 1,
            "required_interventions_complete": True,
            "selection_performed_on_temporal": False,
            "calibration_refit": False,
            "result_feedback_into_development": False,
            "old_exposed_test_access": False,
            "not_an_official_sabdab2_split": True,
            "bound_complex_derived_intra_chain_backbone_geometry": True,
            "cross_chain_coordinates_are_not_model_inputs": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "framecontact_temporal_double_cold_measurement_result_receipt_v1",
        "status": payload["status"],
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "opening_count": 1,
        "required_interventions_complete": True,
        "selection_performed": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
