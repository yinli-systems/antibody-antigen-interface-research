#!/usr/bin/env python3
"""Recover fixed-0.5 counts from the exact MIPE paper-row prediction exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROLES = ("primary_epoch_800", "secondary_validation_best")
TASKS = {"contact": ("mean_bidirectional_score", "pair_target"),
         "epitope": ("epitope_score", "epitope_target"),
         "paratope": ("paratope_score", "paratope_target")}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def micro_metrics(values) -> dict:
    tp, fp, tn, fn = map(int, values)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "mcc": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0}


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_root.exists(), "Refusing to overwrite recovery output")
    require(sha256(args.evaluation) == args.evaluation_sha256, "Frozen evaluation SHA mismatch")
    evaluation = json.loads(args.evaluation.read_text())
    require(evaluation["status"] == "complete_create_only_without_policy_refit", "Incomplete evaluation")
    binding = evaluation["heldout_export_binding"]
    root = args.heldout_root
    require(sha256(root / "manifest.json") == binding["export_manifest_sha256"], "Top manifest changed")
    summaries, per_complex, checkpoint_bindings = {}, [], {}
    checked_files = 0
    for role in ROLES:
        totals = {task: np.zeros(4, dtype=np.int64) for task in TASKS}
        indices, codes = set(), set()
        checkpoint_bindings[role] = []
        for fold in range(5):
            name = f"{role}/fold_{fold}"
            bound = binding["bundles"][name]
            folder = root / name
            manifest_path = folder / "manifest.json"
            require(sha256(manifest_path) == bound["manifest_sha256"]
                    == binding["bundle_manifests"][f"{name}/manifest.json"], "Fold manifest changed")
            manifest = json.loads(manifest_path.read_text())
            require(manifest["files"] == bound["prediction_files"], "Prediction hash bindings differ")
            require(len(manifest["files"]) == 112, "Fold must contain 112 predictions")
            require(set(p.name for p in folder.glob("*.npz")) == set(manifest["files"]), "Prediction inventory differs")
            require(manifest["checkpoint_sha256"] == bound["checkpoint_sha256"]
                    and manifest["checkpoint_epoch"] == bound["checkpoint_epoch"], "Checkpoint bindings differ")
            checkpoint_bindings[role].append({
                "fold": fold, "epoch": bound["checkpoint_epoch"],
                "checkpoint_sha256": bound["checkpoint_sha256"],
                "prediction_manifest_sha256": bound["manifest_sha256"],
            })
            observed_fold_indices = set()
            for filename, expected_hash in sorted(manifest["files"].items()):
                path = folder / filename
                require(path.resolve().is_relative_to(root.resolve()), "Prediction path escapes held-out root")
                require(sha256(path) == expected_hash, f"Prediction hash changed: {name}/{filename}")
                with np.load(path, allow_pickle=False) as arrays:
                    index, code = int(arrays["dataset_index"]), str(arrays["complex_code"].item())
                    require(index not in indices and code not in codes, "Duplicate held-out sample")
                    indices.add(index); codes.add(code); observed_fold_indices.add(index)
                    require(int(arrays["fold"]) == fold and int(arrays["data_fold"]) == fold
                            and str(arrays["checkpoint_role"].item()) == role
                            and int(arrays["checkpoint_epoch"]) == bound["checkpoint_epoch"]
                            and str(arrays["data_split"].item()) == "heldout", "Prediction metadata mismatch")
                    pair = arrays["pair_target"]
                    require(pair.ndim == 2, "Pair target must be two dimensional")
                    require(np.array_equal(arrays["epitope_target"], pair.max(axis=1))
                            and np.array_equal(arrays["paratope_target"], pair.max(axis=0)), "Marginal labels differ")
                    ag, ab = arrays["ag_to_ab_score"], arrays["ab_to_ag_score"]
                    require(ag.shape == ab.T.shape == pair.shape, "Attention axes mismatch")
                    require(np.allclose(arrays["mean_bidirectional_score"], (ag + ab.T) / 2, rtol=0, atol=1e-6),
                            "Stored bidirectional mean differs")
                    for task, (score_name, target_name) in TASKS.items():
                        score, target = arrays[score_name], arrays[target_name]
                        require(score.shape == target.shape and target.size > 0
                                and np.isfinite(score).all() and np.isin(target, (0, 1)).all()
                                and ((score >= 0) & (score <= 1)).all(), "Invalid score/target arrays")
                        # Encoded confusion: TN=0, FP=1, FN=2, TP=3.
                        bins = np.bincount((target.astype(np.int64) * 2 + (score >= 0.5)).ravel(), minlength=4)
                        counts = bins[[3, 1, 0, 2]]
                        totals[task] += counts
                        per_complex.append({"role": role, "fold": fold, "dataset_index": index,
                                            "complex_code": code, "task": task,
                                            **micro_metrics(counts)})
                checked_files += 1
            require(observed_fold_indices == set(bound["dataset_indices"]), "Held-out fold membership differs")
        require(indices == set(range(560)) and len(codes) == 560, "Incomplete held-out role")
        original = evaluation["role_results"][role]["summary"]
        summaries[role] = {}
        for task in TASKS:
            metrics = micro_metrics(totals[task])
            if task == "contact":
                rank = original["pair"]["directions"]["mean_bidirectional"]["pooled_raw_ranking_descriptive_length_confounded"]["metrics"]
            else:
                source = original["marginal"][task]["micro"]
                rank = source["ranking"]
                reference = source["mcc_at_released_default_0_5"]
                for short, long in (("tp", "true_positives"), ("fp", "false_positives"),
                                    ("tn", "true_negatives"), ("fn", "false_negatives")):
                    require(metrics[short] == reference[long], "Fixed-0.5 residue counts differ from original evaluation")
                require(math.isclose(metrics["mcc"], reference["mcc"], abs_tol=1e-12), "Residue MCC mismatch")
            require(sum(int(v) for v in totals[task]) == rank["items"]
                    and metrics["tp"] + metrics["fn"] == rank["positives"], "Ranking and threshold cohorts differ")
            summaries[role][task] = {**metrics, "auroc": rank["auroc"], "pooled_ap": rank["auprc"],
                                     "threshold": 0.5, "complexes": len(codes)}
        summaries[role]["historical_validation_fitted_residue"] = {}
        for task in ("epitope", "paratope"):
            counts = original["marginal"][task]["micro"]["mcc_at_fold_specific_validation_fitted_threshold"]
            summaries[role]["historical_validation_fitted_residue"][task] = micro_metrics(
                [counts[k] for k in ("true_positives", "false_positives", "true_negatives", "false_negatives")])
    result = {
        "status": "PASS_MIPE_PAPER_TRACK_FIXED05_RECOVERY",
        "source_evaluation_sha256": args.evaluation_sha256,
        "source_evaluation": str(args.evaluation), "source_heldout_root": str(root),
        "export_manifest_sha256": binding["export_manifest_sha256"],
        "script_sha256": sha256(Path(__file__)), "track_id": evaluation["track_id"],
        "checkpoint_bindings": checkpoint_bindings, "verified_prediction_artifacts": checked_files,
        "results": summaries,
        "threshold_comparison": ">= 0.5", "aggregation": "micro counts pooled before computing ratios",
        "ranking_metrics": "Copied from SHA-bound original evaluation for the same verified raw predictions; AP is pooled average precision.",
        "scope": "Original MIPE paper-row five-fold held-out cohort, 560 complexes per role; not the SAbDab2 test-455 cohort",
        "contact_score": "mean of two head-averaged row-softmax attention maps after axis transpose",
        "model_inference_rerun": False, "training_performed": False, "threshold_selection_performed": False,
        "claim_boundary": "Descriptive historical replay. The normalized contact readout must be named explicitly. Keep 560-cohort values separate from 455-cohort baselines.",
    }
    args.output_root.mkdir(parents=True)
    result_path = args.output_root / "MIPE_PAPER_TRACK_FIXED05_RESULT.json"
    csv_path = args.output_root / "MIPE_PAPER_TRACK_FIXED05_METRICS.csv"
    detail_path = args.output_root / "MIPE_PAPER_TRACK_FIXED05_PER_COMPLEX.csv"
    write_json(result_path, result)
    table = [{"role": role, "task": task, **summaries[role][task]} for role in ROLES for task in TASKS]
    for path, rows in ((csv_path, table), (detail_path, per_complex)):
        with path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(args.output_root / "MIPE_PAPER_TRACK_FIXED05_RECEIPT.json", {
        "status": result["status"], "source_evaluation_sha256": args.evaluation_sha256,
        "script_sha256": result["script_sha256"], "verified_prediction_artifacts": checked_files,
        "outputs": {path.name: sha256(path) for path in (result_path, csv_path, detail_path)},
    })
    print(json.dumps({"status": result["status"], "results": summaries,
                      "checkpoint_bindings": checkpoint_bindings}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
