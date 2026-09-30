#!/usr/bin/env python3
"""Aggregate a complete frozen SaProt/ProSST development layer sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def per_complex_ap(path: Path) -> dict[str, float]:
    frame = pd.read_parquet(path, columns=["instance_id", "label", "score"])
    output: dict[str, float] = {}
    for instance_id, group in frame.groupby("instance_id", sort=True):
        labels = group["label"].to_numpy(dtype=np.uint8)
        if not labels.any():
            continue
        scores = group["score"].to_numpy(dtype=np.float64)
        output[str(instance_id)] = float(average_precision_score(labels, scores))
    return output


def crossed_bootstrap(
    differences_by_seed: dict[int, dict[str, float]],
    component_by_instance: dict[str, str],
    *,
    seed: int = 20260930,
    repeats: int = 10000,
) -> dict[str, Any]:
    training_seeds = sorted(differences_by_seed)
    shared = sorted(set.intersection(*(set(differences_by_seed[s]) for s in training_seeds)))
    if not shared:
        raise ValueError("no complete paired complex effects")
    components = sorted({component_by_instance[instance] for instance in shared})
    if len(components) < 2:
        raise ValueError("at least two authority components required")
    matrix = np.zeros((len(training_seeds), len(components)), dtype=np.float64)
    for seed_index, training_seed in enumerate(training_seeds):
        effects = differences_by_seed[training_seed]
        for component_index, component in enumerate(components):
            values = [effects[instance] for instance in shared if component_by_instance[instance] == component]
            matrix[seed_index, component_index] = float(np.mean(values))
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    component_only = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled_seeds = rng.integers(matrix.shape[0], size=matrix.shape[0])
        sampled_components = rng.integers(matrix.shape[1], size=matrix.shape[1])
        samples[index] = float(matrix[sampled_seeds][:, sampled_components].mean())
        component_only[index] = float(matrix[:, sampled_components].mean())
    raw = np.array([
        differences_by_seed[training_seed][instance]
        for training_seed in training_seeds for instance in shared
    ])
    return {
        "estimand": "equal_component_equal_training_seed_mean_paired_difference",
        "mean_difference": float(matrix.mean()),
        "complex_weighted_mean_difference": float(raw.mean()),
        "crossed_seed_component_ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "component_ci95_conditional_on_fixed_seeds": [
            float(value) for value in np.quantile(component_only, [0.025, 0.975])
        ],
        "training_seeds": training_seeds,
        "complex_count": len(shared),
        "component_count": len(components),
        "positive_component_fraction": float(np.mean(matrix.mean(axis=0) > 0.0)),
        "negative_component_fraction": float(np.mean(matrix.mean(axis=0) < 0.0)),
        "bootstrap_repeats": repeats,
        "scope": "development-only pointwise exploratory interval",
    }


def metric_snapshot(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "instance_count": int(evaluation["instance_count"]),
        "pair_macro_ap": float(evaluation["pair"]["macro"]["auprc"]),
        "pair_micro_ap": float(evaluation["pair"]["micro"]["auprc"]),
        "pair_macro_auroc": float(evaluation["pair"]["macro"]["auroc"]),
        "epitope_macro_ap": float(evaluation["nodes"]["epitope"]["macro"]["auprc"]),
        "paratope_macro_ap": float(evaluation["nodes"]["paratope"]["macro"]["auprc"]),
        "partner_mrr": float(evaluation["partner_retrieval"]["mrr"]),
        "partner_recall_at_1": float(evaluation["partner_retrieval"]["recall_at_1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    if sha256(plan_path) != args.plan_sha256:
        raise RuntimeError("plan digest mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("test_authorized") or plan.get("ood_authorized"):
        raise RuntimeError("plan unexpectedly authorizes closed data")
    split = json.loads(Path(plan["paths"]["split_manifest"]).read_text(encoding="utf-8"))
    validation_ids = set(map(str, split["splits"]["validation"]))
    component_by_instance = {
        str(instance): str(component)
        for instance, component in split["component_id_by_instance"].items()
        if str(instance) in validation_ids
    }
    if set(component_by_instance) != validation_ids:
        raise RuntimeError("component authority does not cover validation exactly")

    task_records: list[dict[str, Any]] = []
    per_complex: dict[tuple[str, int], dict[str, float]] = {}
    receipts: dict[str, str] = {}
    task_by_id = {int(task["task_id"]): task for task in plan["tasks"]}
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        run_root = args.attempt_root.resolve() / "runs" / f"task_{task_id:02d}"
        complete_path = run_root / "COMPLETE.json"
        if not complete_path.is_file():
            raise RuntimeError(f"missing terminal receipt: {complete_path}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "COMPLETE_DEVELOPMENT_TRAINING":
            raise RuntimeError(f"task not complete: {complete_path}")
        if complete.get("task") != task or complete.get("plan_sha256") != args.plan_sha256:
            raise RuntimeError(f"task/plan mismatch: {complete_path}")
        best_path = run_root / "validation_best" / "evaluation.json"
        last_path = run_root / "validation_last" / "evaluation.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        last = json.loads(last_path.read_text(encoding="utf-8"))
        if int(best["instance_count"]) != 217 or int(last["instance_count"]) != 217:
            raise RuntimeError(f"incomplete validation read: task {task_id}")
        pair_path = run_root / "validation_best" / "pair_predictions.parquet"
        effects = per_complex_ap(pair_path)
        if set(effects) != validation_ids:
            missing = sorted(validation_ids - set(effects))
            raise RuntimeError(f"task {task_id} missing positive AP cases: {missing[:10]}")
        per_complex[(task["arm"], int(task["seed"]))] = effects
        receipts[str(complete_path)] = sha256(complete_path)
        task_records.append({
            "task_id": task_id, "arm": task["arm"], "family": task["family"],
            "layer": task["layer"], "kind": task["kind"], "seed": int(task["seed"]),
            "parameter_count": int(complete["parameter_count"]),
            "wall_seconds": float(complete["wall_seconds"]),
            "best": metric_snapshot(best), "last": metric_snapshot(last),
            "best_minus_last_pair_macro_ap": float(best["pair"]["macro"]["auprc"] - last["pair"]["macro"]["auprc"]),
            "complete_receipt": str(complete_path),
        })

    if len(task_records) != 36:
        raise AssertionError(len(task_records))
    arm_records: dict[str, list[dict[str, Any]]] = {}
    for record in task_records:
        arm_records.setdefault(record["arm"], []).append(record)
    if any(len(records) != 3 for records in arm_records.values()):
        raise RuntimeError("every arm must have exactly three seeds")

    arm_summaries = []
    for arm, records in sorted(arm_records.items()):
        records = sorted(records, key=lambda item: item["seed"])
        parameter_counts = {record["parameter_count"] for record in records}
        if len(parameter_counts) != 1:
            raise RuntimeError(f"parameter count varies within {arm}")
        arm_summaries.append({
            "arm": arm, "family": records[0]["family"], "layer": records[0]["layer"],
            "kind": records[0]["kind"], "seeds": [record["seed"] for record in records],
            "parameter_count": next(iter(parameter_counts)),
            "best_pair_macro_ap_by_seed": {str(record["seed"]): record["best"]["pair_macro_ap"] for record in records},
            "best_pair_macro_ap_mean": float(np.mean([record["best"]["pair_macro_ap"] for record in records])),
            "best_pair_macro_ap_std": float(np.std([record["best"]["pair_macro_ap"] for record in records], ddof=1)),
            "last_pair_macro_ap_mean": float(np.mean([record["last"]["pair_macro_ap"] for record in records])),
            "best_epitope_macro_ap_mean": float(np.mean([record["best"]["epitope_macro_ap"] for record in records])),
            "best_paratope_macro_ap_mean": float(np.mean([record["best"]["paratope_macro_ap"] for record in records])),
            "best_partner_mrr_mean": float(np.mean([record["best"]["partner_mrr"] for record in records])),
            "best_partner_recall_at_1_mean": float(np.mean([record["best"]["partner_recall_at_1"] for record in records])),
            "total_wall_seconds": float(sum(record["wall_seconds"] for record in records)),
        })

    controls = {
        "saprot": "saprot_random_width_control",
        "prosst": "prosst_random_width_control",
    }
    comparisons = []
    for summary in arm_summaries:
        if summary["kind"] != "real":
            continue
        control_arm = controls[summary["family"]]
        differences_by_seed = {}
        seed_rows = []
        for seed in (1, 2, 3):
            real = per_complex[(summary["arm"], seed)]
            control = per_complex[(control_arm, seed)]
            if set(real) != set(control):
                raise RuntimeError("paired complex sets differ")
            differences_by_seed[seed] = {instance: real[instance] - control[instance] for instance in real}
            real_mean = float(np.mean(list(real.values())))
            control_mean = float(np.mean(list(control.values())))
            seed_rows.append({
                "seed": seed, "real_macro_ap": real_mean, "control_macro_ap": control_mean,
                "difference": real_mean - control_mean,
            })
        comparisons.append({
            "arm": summary["arm"], "family": summary["family"], "layer": summary["layer"],
            "control_arm": control_arm, "per_seed": seed_rows,
            "paired_complex_component_analysis": crossed_bootstrap(
                differences_by_seed, component_by_instance
            ),
        })

    selections = {}
    for family in ("saprot", "prosst"):
        candidates = [summary for summary in arm_summaries if summary["family"] == family and summary["kind"] == "real"]
        candidates.sort(key=lambda item: (-item["best_pair_macro_ap_mean"], int(item["layer"])))
        top = candidates[0]
        tied = [item for item in candidates if top["best_pair_macro_ap_mean"] - item["best_pair_macro_ap_mean"] <= 0.001]
        selected = min(tied, key=lambda item: int(item["layer"]))
        selections[family] = {
            "selected_arm": selected["arm"], "selected_layer": selected["layer"],
            "selected_mean_pair_macro_ap": selected["best_pair_macro_ap_mean"],
            "raw_highest_arm": top["arm"], "raw_highest_layer": top["layer"],
            "raw_highest_mean_pair_macro_ap": top["best_pair_macro_ap_mean"],
            "tie_window": 0.001,
            "role": "development selection only; no blind or SOTA status",
        }

    output = {
        "format_version": "structural_plm_training_analysis_v1",
        "status": "COMPLETE_DEVELOPMENT_LAYER_SWEEP_ANALYSIS",
        "plan_sha256": args.plan_sha256,
        "source_commit": plan["source_commit"],
        "task_count": len(task_records), "arm_count": len(arm_summaries),
        "validation_complex_count": len(validation_ids),
        "component_count": len(set(component_by_instance.values())),
        "primary_endpoint": plan["primary_endpoint"],
        "task_receipt_sha256": receipts,
        "tasks": task_records,
        "arms": arm_summaries,
        "real_vs_width_matched_control": comparisons,
        "development_layer_selection": selections,
        "test_labels_read": False, "ood_labels_read": False,
        "blind_claim_authorized": False, "sota_claim_authorized": False,
        "limitations": [
            "Three training seeds per arm quantify optimization repeatability, not population uncertainty.",
            "All layer selection occurs on previously used development validation data.",
            "Pointwise exploratory intervals are not multiplicity-adjusted confirmatory tests.",
            "Historical ESM-2 and ESM-IF1 controls are not contemporaneous same-run arms.",
            "Partner ranking is secondary because score/contact dissociation has been observed.",
        ],
    }
    exclusive_json(args.output.resolve(), output)

    table = pd.DataFrame([{key: value for key, value in summary.items() if not isinstance(value, dict)} for summary in arm_summaries])
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps({
        "status": output["status"], "tasks": len(task_records), "arms": len(arm_summaries),
        "selections": selections, "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
