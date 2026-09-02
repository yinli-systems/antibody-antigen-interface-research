#!/usr/bin/env python3
"""Analyze the fixed seed-matched SABLE external task routing once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")
TASK_ARM = {
    "contact": "baseline",
    "paratope": "baseline",
    "epitope": "edge_topk2_supervised",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260830


class AnalysisError(ValueError):
    """Raised when an external result or frozen authority differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON root is not an object: {path}")
    return value


def cluster_weighted_estimate(
    per_seed: dict[int, dict[str, float]], pdb_by_entry: dict[str, str]
) -> dict[str, Any]:
    entry_sets = [set(per_seed[seed]) for seed in SEEDS]
    if not entry_sets or any(entries != entry_sets[0] for entries in entry_sets[1:]):
        raise AnalysisError("seed-specific eligible entry sets differ")
    eligible_entries = sorted(entry_sets[0])
    if not eligible_entries:
        raise AnalysisError("cluster-weighted estimand has no eligible entries")
    missing_pdb = sorted(set(eligible_entries) - set(pdb_by_entry))
    if missing_pdb:
        raise AnalysisError(f"PDB mapping lacks eligible entry {missing_pdb[0]}")
    seed_averaged = {
        entry_id: float(np.mean([per_seed[seed][entry_id] for seed in SEEDS]))
        for entry_id in eligible_entries
    }
    by_pdb: dict[str, list[float]] = {}
    for entry_id, value in seed_averaged.items():
        by_pdb.setdefault(pdb_by_entry[entry_id], []).append(value)
    pdb_means = np.asarray(
        [float(np.mean(by_pdb[pdb_id])) for pdb_id in sorted(by_pdb)], dtype=np.float64
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(pdb_means), size=(BOOTSTRAP_REPLICATES, len(pdb_means)), endpoint=False
    )
    replicates = pdb_means[indices].mean(axis=1)
    return {
        "point_estimate": float(pdb_means.mean()),
        "ci95": [
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
        ],
        "eligible_entry_count": len(eligible_entries),
        "unique_pdb_id_count": len(pdb_means),
        "seed_count": len(SEEDS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-plan", type=Path, required=True)
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
    if sha256_file(args.external_plan) != EXPECTED_EXTERNAL_PLAN_SHA256:
        raise AnalysisError("external execution plan SHA256 differs")
    external_plan = load_json(args.external_plan)
    authorization = load_json(args.authorization)
    authorization_sha256 = sha256_file(args.authorization)
    checkpoint_binding = load_json(args.checkpoint_binding)
    checkpoint_binding_sha256 = sha256_file(args.checkpoint_binding)
    label_manifest_path = args.label_root / "SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = args.label_root / "SABLE_BOUND_LABEL_RECEIPT.json"
    label_manifest = load_json(label_manifest_path)
    label_receipt = load_json(label_receipt_path)
    if (
        external_plan.get("status") != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or external_plan.get("analysis", {}).get("bootstrap", {}).get("replicates")
        != BOOTSTRAP_REPLICATES
        or external_plan.get("analysis", {}).get("bootstrap", {}).get("seed")
        != BOOTSTRAP_SEED
        or authorization.get("status") != "AUTHORIZED_ONE_TIME_BOUND_LABEL_OPENING"
        or authorization.get("one_time_evaluation_authorized") is not True
        or authorization.get("selection_or_tuning") is not False
        or checkpoint_binding.get("status")
        != "PASS_EXACT_NO_DISTANCE_CHECKPOINT_BINDING"
        or checkpoint_binding.get("selection_or_tuning") is not False
        or label_manifest.get("status")
        != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or label_manifest.get("authority", {}).get("authorization_sha256")
        != authorization_sha256
        or label_manifest.get("mapping_gate", {}).get("passed") is not True
        or label_manifest.get("authority", {}).get("selection_or_tuning") is not False
        or label_receipt.get("manifest_sha256") != sha256_file(label_manifest_path)
        or label_receipt.get("mapping_gate_passed") is not True
        or label_receipt.get("performance_read") is not False
    ):
        raise AnalysisError("external analysis authority differs")
    pdb_by_entry = {
        str(row["entry_id"]): str(row["pdb_id"])
        for row in label_manifest.get("records", [])
    }
    if len(pdb_by_entry) != int(label_manifest["counts"]["analyzable_entries"]):
        raise AnalysisError("label manifest entry mapping differs")

    components: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    component_bindings: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    audit_reference: dict[str, Any] | None = None
    for arm in ARMS:
        for seed in SEEDS:
            run_root = args.component_root / arm / f"seed-{seed}"
            result_path = run_root / "SABLE_COMPONENT_RESULT.json"
            receipt_path = run_root / "SABLE_COMPONENT_RECEIPT.json"
            result = load_json(result_path)
            receipt = load_json(receipt_path)
            result_sha256 = sha256_file(result_path)
            if (
                result.get("status") != "PASS_SINGLE_OPENING_SABLE_COMPONENT_MEASUREMENT"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or receipt.get("status") != result.get("status")
                or receipt.get("arm") != arm
                or receipt.get("seed") != seed
                or receipt.get("result_sha256") != result_sha256
                or receipt.get("opening_count") != 1
                or receipt.get("selection_performed") is not False
                or receipt.get("test_loaded") is not False
                or receipt.get("temporal_holdout_loaded") is not False
                or result.get("audit", {}).get("external_execution_plan_sha256")
                != EXPECTED_EXTERNAL_PLAN_SHA256
                or result.get("audit", {}).get("authorization_sha256")
                != authorization_sha256
                or result.get("audit", {}).get("checkpoint_binding_sha256")
                != checkpoint_binding_sha256
                or result.get("audit", {}).get("opening_count") != 1
                or result.get("audit", {}).get(
                    "experimental_bound_coordinates_used_as_model_input"
                )
                is not False
                or result.get("audit", {}).get("distance_labels_loaded") is not False
                or result.get("audit", {}).get("model_parameters_updated") is not False
                or result.get("audit", {}).get("selection_performed") is not False
                or result.get("audit", {}).get("old_sabdab2_test_loaded") is not False
                or result.get("audit", {}).get("temporal_holdout_loaded") is not False
            ):
                raise AnalysisError(f"component authority differs: {arm}/seed-{seed}")
            current_reference = {
                key: result["audit"][key]
                for key in (
                    "predicted_structure_manifest_sha256",
                    "esm_embedding_manifest_sha256",
                    "bound_label_manifest_sha256",
                    "entry_count",
                )
            }
            if audit_reference is None:
                audit_reference = current_reference
            elif current_reference != audit_reference:
                raise AnalysisError("component external-data authorities differ")
            components[arm][seed] = result
            component_bindings[arm][str(seed)] = {
                "result_path": str(result_path),
                "result_sha256": result_sha256,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }

    routed: dict[str, Any] = {}
    for task, arm in TASK_ARM.items():
        seed_metrics = {
            str(seed): {
                key: components[arm][seed]["metrics"][task][key]
                for key in (
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
                for key in (
                    "mean_per_eligible_entry_average_precision",
                    "average_precision",
                    "auroc",
                    "precision",
                    "recall",
                    "f1",
                    "mcc",
                )
            },
            "pdb_equal_weighted_seed_averaged_per_entry_ap": cluster_weighted_estimate(
                per_seed_ap, pdb_by_entry
            ),
            "zero_positive_entry_count": len(
                components[arm][SEEDS[0]]["metrics"][task]["zero_positive_entry_ids"]
            ),
        }

    payload = {
        "format_version": "sable_v0_2_0_framecontact_external_analysis_v1",
        "status": "PASS_SABLE_V0_2_0_SINGLE_OPENING_EXTERNAL_EVALUATION",
        "dataset": {"name": "SABLE", "version": "0.2.0", "doi": "10.5281/zenodo.21759053"},
        "seeds": list(SEEDS),
        "task_routing": TASK_ARM,
        "routed_metrics": routed,
        "component_bindings": component_bindings,
        "audit": {
            "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
            "authorization_sha256": authorization_sha256,
            "checkpoint_binding_sha256": checkpoint_binding_sha256,
            "bound_label_manifest_sha256": sha256_file(label_manifest_path),
            "bound_label_receipt_sha256": sha256_file(label_receipt_path),
            "entry_count": len(pdb_by_entry),
            "unique_pdb_id_count": len(set(pdb_by_entry.values())),
            "opening_count": 1,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sable_v0_2_0_framecontact_external_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": sha256_file(args.output),
        "authorization_sha256": authorization_sha256,
        "all_six_components_verified": True,
        "opening_count": 1,
        "selection_performed": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {**receipt, "receipt_sha256": sha256_file(args.receipt_output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
