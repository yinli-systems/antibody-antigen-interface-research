#!/usr/bin/env python3
"""Analyze six fixed ABF-30B SABLE measurements with seed-matched routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")
TASK_ARM = {
    "contact": "baseline",
    "paratope": "baseline",
    "epitope": "edge_topk2_supervised",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260830


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root differs: {path}")
    return value


def cluster_weighted_estimate(
    per_seed: dict[int, dict[str, float]], pdb_by_entry: dict[str, str]
) -> dict[str, Any]:
    entry_sets = [set(per_seed[seed]) for seed in SEEDS]
    if not entry_sets or any(entries != entry_sets[0] for entries in entry_sets[1:]):
        raise ValueError("seed-specific eligible entry sets differ")
    eligible = sorted(entry_sets[0])
    if not eligible or set(eligible) - set(pdb_by_entry):
        raise ValueError("cluster-weighted entry authority differs")
    seed_average = {
        entry_id: float(np.mean([per_seed[seed][entry_id] for seed in SEEDS]))
        for entry_id in eligible
    }
    by_pdb: dict[str, list[float]] = {}
    for entry_id, value in seed_average.items():
        by_pdb.setdefault(pdb_by_entry[entry_id], []).append(value)
    values = np.asarray(
        [float(np.mean(by_pdb[pdb_id])) for pdb_id in sorted(by_pdb)], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)), endpoint=False
    )
    replicates = values[indices].mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "ci95": [
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
        ],
        "eligible_entry_count": len(eligible),
        "unique_pdb_id_count": len(values),
        "seed_count": len(SEEDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--component-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    plan = load_json(args.measurement_plan)
    plan_sha = sha256_file(args.measurement_plan)
    authorization = load_json(args.authorization)
    authorization_sha = sha256_file(args.authorization)
    binding = load_json(args.checkpoint_binding)
    binding_sha = sha256_file(args.checkpoint_binding)
    label_manifest_path = args.label_root / "SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = args.label_root / "SABLE_BOUND_LABEL_RECEIPT.json"
    label_manifest = load_json(label_manifest_path)
    label_receipt = load_json(label_receipt_path)
    if (
        plan.get("status")
        != "FROZEN_BEFORE_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_MEASUREMENT"
        or plan.get("metrics", {}).get("pdb_equal_weighted_bootstrap", {}).get(
            "replicates"
        )
        != BOOTSTRAP_REPLICATES
        or plan.get("metrics", {}).get("pdb_equal_weighted_bootstrap", {}).get("seed")
        != BOOTSTRAP_SEED
        or authorization.get("status")
        != "AUTHORIZED_MODEL_SPECIFIC_ANALYST_EXPOSED_MEASUREMENT_ONLY"
        or authorization.get("measurement_plan_sha256") != plan_sha
        or authorization.get("checkpoint_binding_sha256") != binding_sha
        or authorization.get("fresh_blind_test") is not False
        or authorization.get("measurement_only") is not True
        or authorization.get("model_unseen_before_this_measurement") is not True
        or authorization.get("selection_or_tuning") is not False
        or binding.get("status") != "PASS_EXACT_ABF30B_DEVELOPMENT_CHECKPOINT_BINDING"
        or label_manifest.get("status") != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or sha256_file(label_manifest_path) != authorization["bound_label_manifest_sha256"]
        or sha256_file(label_receipt_path) != authorization["bound_label_receipt_sha256"]
        or label_receipt.get("manifest_sha256") != sha256_file(label_manifest_path)
    ):
        raise ValueError("SABLE analysis authority differs")
    pdb_by_entry = {
        str(row["entry_id"]): str(row["pdb_id"])
        for row in label_manifest.get("records", [])
    }
    if len(pdb_by_entry) != int(label_manifest["counts"]["analyzable_entries"]):
        raise ValueError("SABLE label entry mapping differs")

    components: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    component_bindings: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    common_data: dict[str, Any] | None = None
    for arm in ARMS:
        for seed in SEEDS:
            run_root = args.component_root / arm / f"seed-{seed}"
            result_path = run_root / "SABLE_COMPONENT_RESULT.json"
            receipt_path = run_root / "SABLE_COMPONENT_RECEIPT.json"
            result = load_json(result_path)
            receipt = load_json(receipt_path)
            result_sha = sha256_file(result_path)
            audit = result.get("audit", {})
            if (
                result.get("status")
                != "PASS_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_COMPONENT_MEASUREMENT"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or receipt.get("status") != result.get("status")
                or receipt.get("arm") != arm
                or receipt.get("seed") != seed
                or receipt.get("result_sha256") != result_sha
                or receipt.get("fresh_blind_test") is not False
                or receipt.get("model_unseen_before_this_measurement") is not True
                or receipt.get("measurement_only") is not True
                or receipt.get("selection_performed") is not False
                or audit.get("measurement_plan_sha256") != plan_sha
                or audit.get("authorization_sha256") != authorization_sha
                or audit.get("checkpoint_binding_sha256") != binding_sha
                or audit.get("sable_external_labels_loaded") is not True
                or audit.get("fresh_blind_test") is not False
                or audit.get("analyst_has_seen_prior_model_results_on_dataset") is not True
                or audit.get("model_unseen_before_this_measurement") is not True
                or audit.get("measurement_only") is not True
                or audit.get("experimental_bound_coordinates_used_as_model_input")
                is not False
                or audit.get("distance_labels_loaded") is not False
                or audit.get("calibration_refit") is not False
                or audit.get("model_parameters_updated") is not False
                or audit.get("selection_performed") is not False
                or audit.get("feedback_to_model_or_protocol") is not False
                or audit.get("old_sabdab2_test_loaded") is not False
                or audit.get("temporal_holdout_loaded") is not False
            ):
                raise ValueError(f"component authority differs: {arm}/seed-{seed}")
            current_data = {
                key: audit[key]
                for key in (
                    "predicted_structure_manifest_sha256",
                    "mixed_embedding_manifest_sha256",
                    "bound_label_manifest_sha256",
                    "entry_count",
                )
            }
            if common_data is None:
                common_data = current_data
            elif current_data != common_data:
                raise ValueError("component data authorities differ")
            components[arm][seed] = result
            component_bindings[arm][str(seed)] = {
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }

    routed: dict[str, Any] = {}
    metric_names = (
        "mean_per_eligible_entry_average_precision",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "f1",
        "mcc",
        "eligible_entry_count",
        "total_entry_count",
        "observed_label_count",
        "positive_label_count",
    )
    summary_names = metric_names[:7]
    for task, arm in TASK_ARM.items():
        seed_metrics = {
            str(seed): {
                key: components[arm][seed]["metrics"][task][key]
                for key in metric_names
            }
            for seed in SEEDS
        }
        per_seed_ap = {
            seed: {
                str(key): float(value)
                for key, value in components[arm][seed]["metrics"][task][
                    "per_entry_average_precision"
                ].items()
            }
            for seed in SEEDS
        }
        routed[task] = {
            "source_arm": arm,
            "seed_metrics": seed_metrics,
            "median_across_seeds": {
                key: float(np.median([seed_metrics[str(seed)][key] for seed in SEEDS]))
                for key in summary_names
            },
            "pdb_equal_weighted_seed_averaged_per_entry_ap": cluster_weighted_estimate(
                per_seed_ap, pdb_by_entry
            ),
            "zero_positive_entry_count": len(
                components[arm][SEEDS[0]]["metrics"][task]["zero_positive_entry_ids"]
            ),
        }

    payload = {
        "format_version": "abf30b_sable_model_specific_measurement_analysis_v1",
        "status": "PASS_ABF30B_ANALYST_EXPOSED_SABLE_MODEL_SPECIFIC_MEASUREMENT",
        "dataset": {
            "name": "SABLE",
            "version": "0.2.0",
            "doi": "10.5281/zenodo.21759053",
            "official_test_split": True,
        },
        "seeds": list(SEEDS),
        "task_routing": TASK_ARM,
        "routed_metrics": routed,
        "component_bindings": component_bindings,
        "audit": {
            "measurement_plan_sha256": plan_sha,
            "authorization_sha256": authorization_sha,
            "checkpoint_binding_sha256": binding_sha,
            "bound_label_manifest_sha256": sha256_file(label_manifest_path),
            "bound_label_receipt_sha256": sha256_file(label_receipt_path),
            "entry_count": len(pdb_by_entry),
            "unique_pdb_id_count": len(set(pdb_by_entry.values())),
            "bound_label_opening_count": 1,
            "new_bound_label_opening": False,
            "sable_external_labels_loaded": True,
            "fresh_blind_test": False,
            "analyst_has_seen_prior_model_results_on_dataset": True,
            "model_unseen_before_this_measurement": True,
            "measurement_only": True,
            "formal_blind_test_claim_authorized": False,
            "experimental_bound_coordinates_used_as_model_input": False,
            "experimental_bound_coordinates_used_for_labels_only": True,
            "cross_partner_geometry_used_as_model_input": False,
            "distance_labels_loaded": False,
            "model_parameters_updated": False,
            "calibration_refit": False,
            "selection_performed": False,
            "feedback_to_model_or_protocol": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    analysis_sha = publish(args.output, payload)
    receipt = {
        "format_version": "abf30b_sable_model_specific_measurement_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": analysis_sha,
        "all_six_components_verified": True,
        "fresh_blind_test": False,
        "model_unseen_before_this_measurement": True,
        "measurement_only": True,
        "selection_performed": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_medians": {
                    task: routed[task]["median_across_seeds"] for task in TASK_ARM
                },
                "analysis_sha256": analysis_sha,
                "receipt_sha256": receipt_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
