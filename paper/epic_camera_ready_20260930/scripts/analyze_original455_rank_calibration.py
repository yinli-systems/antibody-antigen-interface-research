#!/usr/bin/env python3
"""Audit practical ranking, calibration, uncertainty, and recall aggregation.

Reads only completed raw prediction NPZ files. Performs no model inference and
never selects a checkpoint or threshold using test labels. Fixed 0.5 metrics are
reported only as historical operating-point diagnostics; ranking metrics are
threshold-free. The test cohort is historical/exposed and is not presented as
fresh blind evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TASKS = ("contact", "epitope", "paratope")
SEEDS = (17, 29, 43)
FIXED_K = (10, 20, 50, 100)
RECALL_TARGETS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
PRECISION_TARGETS = (0.005, 0.01, 0.02, 0.05, 0.10)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def safe_ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if int(y.sum()) else float("nan")


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    rows = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & ((p <= hi) if i == bins - 1 else (p < hi))
        n = int(mask.sum())
        if n:
            mp = float(p[mask].mean())
            yf = float(y[mask].mean())
            value += n / total * abs(mp - yf)
        else:
            mp = yf = None
        rows.append({"lower": lo, "upper": hi, "count": n, "mean_probability": mp, "positive_fraction": yf})
    return float(value), rows


def precision_recall_curve_exact(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order].astype(np.int64)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    thresholds = s[order]
    # Keep last index of each score tie so all equal scores receive same threshold.
    keep = np.r_[thresholds[1:] != thresholds[:-1], True]
    return precision[keep], recall[keep], thresholds[keep]


def precision_at_recall_from_curve(p: np.ndarray, r: np.ndarray, t: np.ndarray, target: float) -> dict[str, float | None]:
    idx = np.flatnonzero(r >= target)
    if not len(idx):
        return {"precision": None, "recall": None, "threshold": None}
    # Among thresholds reaching target, report highest precision; deterministic earliest tie.
    local = idx[np.argmax(p[idx])]
    return {"precision": float(p[local]), "recall": float(r[local]), "threshold": float(t[local])}


def recall_at_precision_from_curve(p: np.ndarray, r: np.ndarray, t: np.ndarray, target: float) -> dict[str, float | None]:
    idx = np.flatnonzero(p >= target)
    if not len(idx):
        return {"precision": None, "recall": 0.0, "threshold": None}
    local = idx[np.argmax(r[idx])]
    return {"precision": float(p[local]), "recall": float(r[local]), "threshold": float(t[local])}


def expected_precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> tuple[float, float, int]:
    k = min(max(int(k), 1), len(y))
    order = np.argsort(-s, kind="mergesort")
    cutoff = float(s[order[k - 1]])
    above = s > cutoff
    tied = s == cutoff
    slots = k - int(above.sum())
    hits = int(y[above].sum()) + slots * float(y[tied].mean())
    return float(hits / k), float(hits / max(int(y.sum()), 1)), k


def load_seed(seed_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(seed_root.glob("*.npz")):
        z = np.load(path, allow_pickle=False)
        iid = str(z["instance_id"].item())
        if iid in records:
            raise ValueError(f"duplicate {iid} in {seed_root}")
        rec: dict[str, Any] = {"source": str(path), "sha256": sha256(path)}
        for task in TASKS:
            logits = z[f"{task}_logits"].astype(np.float64)
            labels = z[f"{task}_labels"].astype(np.uint8)
            mask = z[f"{task}_mask"].astype(bool)
            if task == "contact":
                if mask.ndim == 2:
                    mask_flat = mask.reshape(-1)
                    # Stored contact logits/labels are already compact valid-pair vectors.
                    if len(logits) == int(mask_flat.sum()):
                        pass
                    elif len(logits) == len(mask_flat):
                        logits = logits[mask_flat]
                        labels = labels[mask_flat]
                    else:
                        raise ValueError(f"contact mask/logit mismatch {iid}")
            else:
                logits = logits[mask]
                labels = labels[mask]
            if len(logits) != len(labels) or not len(labels):
                raise ValueError(f"bad {task} arrays {iid}")
            rec[task] = {"logits": logits, "probabilities": sigmoid(logits), "labels": labels}
        records[iid] = rec
    if len(records) != 455:
        raise ValueError(f"expected 455 records, got {len(records)} in {seed_root}")
    return records


def nanmean(values: list[float]) -> float:
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def summarize_task(task: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(records)
    per_complex = []
    all_y, all_p = [], []
    for iid in ids:
        x = records[iid][task]
        y, p = x["labels"], x["probabilities"]
        row: dict[str, Any] = {
            "instance_id": iid,
            "candidate_count": int(len(y)),
            "positive_count": int(y.sum()),
            "prevalence": float(y.mean()),
            "average_precision": safe_ap(y, p),
            "auroc": safe_auc(y, p),
        }
        pred = p >= 0.5
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        row["precision_at_05"] = float(tp / (tp + fp)) if tp + fp else 0.0
        row["recall_at_05"] = float(tp / (tp + fn)) if tp + fn else float("nan")
        if task == "contact":
            cuts: dict[str, Any] = {}
            for k in FIXED_K:
                prec, rec, used = expected_precision_at_k(y, p, k)
                cuts[str(k)] = {"precision": prec, "recall": rec, "k": used}
            ab_len, ag_len = records[iid].get("contact_shape", (None, None))
            # Contact mask shape was not persisted in compact record; infer Lmin from source NPZ.
            z = np.load(records[iid]["source"], allow_pickle=False)
            shape = z["contact_mask"].shape
            lmin = min(map(int, shape))
            for name, k in (("lmin", lmin), ("true_count", int(y.sum())), ("top_1pct", max(1, math.ceil(0.01 * len(y))))):
                prec, rec, used = expected_precision_at_k(y, p, k)
                cuts[name] = {"precision": prec, "recall": rec, "k": used, "enrichment_over_prevalence": float(prec / y.mean())}
            row["ranked_cutoffs"] = cuts
        per_complex.append(row)
        all_y.append(y)
        all_p.append(p)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    ece_value, bins = ece(y, p, 15)
    pred = p >= 0.5
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum((~pred) & (y == 1)))
    curve_precision, curve_recall, curve_threshold = precision_recall_curve_exact(y, p)
    pooled = {
        "candidate_count": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": safe_ap(y, p),
        "auroc": safe_auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_15": ece_value,
        "reliability_bins_15": bins,
        "precision_at_05": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_at_05": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "precision_at_recall": {str(target): precision_at_recall_from_curve(curve_precision, curve_recall, curve_threshold, target) for target in RECALL_TARGETS},
        "recall_at_precision": {str(target): recall_at_precision_from_curve(curve_precision, curve_recall, curve_threshold, target) for target in PRECISION_TARGETS},
    }
    macro = {
        "average_precision": nanmean([r["average_precision"] for r in per_complex]),
        "auroc": nanmean([r["auroc"] for r in per_complex]),
        "precision_at_05": nanmean([r["precision_at_05"] for r in per_complex]),
        "recall_at_05": nanmean([r["recall_at_05"] for r in per_complex]),
    }
    if task == "contact":
        names = [str(k) for k in FIXED_K] + ["lmin", "true_count", "top_1pct"]
        macro["ranked_cutoffs"] = {
            name: {
                "mean_precision": nanmean([r["ranked_cutoffs"][name]["precision"] for r in per_complex]),
                "mean_recall": nanmean([r["ranked_cutoffs"][name]["recall"] for r in per_complex]),
                "mean_k": nanmean([float(r["ranked_cutoffs"][name]["k"]) for r in per_complex]),
                **({"mean_enrichment_over_prevalence": nanmean([r["ranked_cutoffs"][name]["enrichment_over_prevalence"] for r in per_complex])} if name in {"lmin", "true_count", "top_1pct"} else {}),
            }
            for name in names
        }
    return {"pooled": pooled, "macro": macro, "per_complex": per_complex}


def seed_averaged_bootstrap(per_seed: dict[str, dict[str, Any]], task: str, reps: int = 10000) -> dict[str, Any]:
    ids = sorted(set.intersection(*(set(v) for v in per_seed.values())))
    seed_names = sorted(per_seed)
    matrix_ap = np.array([[per_seed[s][i][task]["average_precision"] for i in ids] for s in seed_names], dtype=np.float64)
    matrix_rec = np.array([[per_seed[s][i][task]["recall_at_05"] for i in ids] for s in seed_names], dtype=np.float64)
    rng = np.random.default_rng(20260930)
    ap_by_complex = np.nanmean(matrix_ap, axis=0)
    rec_by_complex = np.nanmean(matrix_rec, axis=0)
    draws_ap = np.empty(reps)
    draws_rec = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(len(ids), size=len(ids))
        draws_ap[b] = np.nanmean(ap_by_complex[idx])
        draws_rec[b] = np.nanmean(rec_by_complex[idx])
    return {
        "estimand": "equal_complex_mean_after_averaging_three_fixed_seeds",
        "complex_count": len(ids),
        "seeds": seed_names,
        "bootstrap_repeats": reps,
        "macro_average_precision": float(np.nanmean(ap_by_complex)),
        "macro_average_precision_ci95": list(map(float, np.quantile(draws_ap, [0.025, 0.975]))),
        "macro_recall_at_05": float(np.nanmean(rec_by_complex)),
        "macro_recall_at_05_ci95": list(map(float, np.quantile(draws_rec, [0.025, 0.975]))),
        "conditional_on_fixed_seeds": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_seed_records = {str(seed): load_seed(args.test_root / f"seed-{seed}") for seed in SEEDS}
    ids = sorted(per_seed_records[str(SEEDS[0])])
    if any(sorted(x) != ids for x in [per_seed_records[str(s)] for s in SEEDS[1:]]):
        raise ValueError("seed instance sets differ")

    summaries: dict[str, Any] = {}
    compact_for_bootstrap: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        key = str(seed)
        summaries[key] = {task: summarize_task(task, per_seed_records[key]) for task in TASKS}
        compact_for_bootstrap[key] = {
            iid: {
                task: {
                    "average_precision": next(r for r in summaries[key][task]["per_complex"] if r["instance_id"] == iid)["average_precision"],
                    "recall_at_05": next(r for r in summaries[key][task]["per_complex"] if r["instance_id"] == iid)["recall_at_05"],
                }
                for task in TASKS
            }
            for iid in ids
        }

    # Ensemble probabilities: average probabilities seedwise for the same instance/task.
    ensemble_records: dict[str, dict[str, Any]] = {}
    for iid in ids:
        ensemble_records[iid] = {"source": per_seed_records["17"][iid]["source"]}
        for task in TASKS:
            labels = per_seed_records["17"][iid][task]["labels"]
            probs = np.mean([per_seed_records[str(seed)][iid][task]["probabilities"] for seed in SEEDS], axis=0)
            if any(not np.array_equal(labels, per_seed_records[str(seed)][iid][task]["labels"]) for seed in SEEDS[1:]):
                raise ValueError(f"label mismatch {iid} {task}")
            ensemble_records[iid][task] = {"labels": labels, "probabilities": probs, "logits": np.log(np.clip(probs, 1e-12, 1 - 1e-12) / np.clip(1 - probs, 1e-12, 1))}
    ensemble = {task: summarize_task(task, ensemble_records) for task in TASKS}

    training = {}
    for seed in SEEDS:
        result_path = args.train_root / f"seed-{seed}" / "RESULT.json"
        d = json.loads(result_path.read_text())
        training[str(seed)] = {
            "result_sha256": sha256(result_path),
            "best_epoch": d["best_epoch"],
            "checkpoint": d["checkpoint"],
            "checkpoint_sha256": d["checkpoint_sha256"],
            "epochs_executed": len(d["trace"]),
            "aggregate_epoch_seconds": float(sum(float(row["seconds"]) for row in d["trace"])),
            "selection": "validation_epitope_mpc_ap_then_contact_mpc_ap_then_earlier_epoch",
            "test_selection_performed": bool(d["test_selection_performed"]),
        }

    source_hashes = {}
    for seed in SEEDS:
        result = args.test_root / f"seed-{seed}" / "RESULT.json"
        source_hashes[str(result)] = sha256(result)
        source_hashes[str(args.train_root / f"seed-{seed}" / "RESULT.json")] = sha256(args.train_root / f"seed-{seed}" / "RESULT.json")

    output = {
        "format_version": "epic_original455_postsubmission_rank_calibration_v1",
        "status": "COMPLETE_POSTSUBMISSION_HISTORICAL_ANALYSIS",
        "evidence_class": "post-submission audited retrain on the official 455 historical split; not the exact OpenDDE submission checkpoint and not fresh blind evidence",
        "source_hashes": source_hashes,
        "seed_summaries": summaries,
        "three_seed_ensemble": ensemble,
        "fixed_seed_complex_bootstrap": {task: seed_averaged_bootstrap(compact_for_bootstrap, task) for task in TASKS},
        "training_cost": training,
        "metric_harmonization": {
            "macro_recall_at_05": "mean of per-complex recalls, equal complex weight",
            "pooled_recall_at_05": "aggregate TP/(TP+FN) over all examples",
            "paper_asymmetry_resolution": "both are reported; neither is silently substituted for the other",
        },
        "claim_boundaries": {
            "exact_submission_model": False,
            "exact_submission_geometry_predictor": False,
            "fresh_blind_test": False,
            "new_model_inference_performed": False,
            "test_labels_used_for_model_selection": False,
            "sota_claim_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "ensemble_contact": ensemble["contact"]["macro"],
        "ensemble_contact_pooled": {k: ensemble["contact"]["pooled"][k] for k in ("prevalence", "average_precision", "auroc", "precision_at_05", "recall_at_05", "brier", "ece_15")},
        "bootstrap_contact": output["fixed_seed_complex_bootstrap"]["contact"],
    }, indent=2))

if __name__ == "__main__":
    main()
