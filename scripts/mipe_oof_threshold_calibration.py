#!/usr/bin/env python3
"""Fit residue decision thresholds on MIPE OOF predictions and apply once to test."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


ROLES = ("primary_epoch_800", "secondary_validation_best")
SIDES = {
    "epitope": ("epitope_score", "epitope_target"),
    "paratope": ("paratope_score", "paratope_target"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arrays(files: list[str], score_key: str, target_key: str) -> tuple[np.ndarray, np.ndarray]:
    targets, scores = [], []
    for filename in files:
        with np.load(filename) as arrays:
            targets.append(arrays[target_key].reshape(-1))
            scores.append(arrays[score_key].reshape(-1))
    return np.concatenate(targets).astype(np.uint8), np.concatenate(scores).astype(float)


def best_mcc_threshold(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-score, kind="mergesort")
    target = target[order]
    score = score[order]
    tp = np.cumsum(target)
    fp = np.cumsum(1 - target)
    indices = np.r_[np.where(np.diff(score) != 0)[0], len(score) - 1]
    true_positive = tp[indices].astype(float)
    false_positive = fp[indices].astype(float)
    false_negative = float(tp[-1]) - true_positive
    true_negative = float(fp[-1]) - false_positive
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    values = np.divide(
        true_positive * true_negative - false_positive * false_negative,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(values))
    index = int(indices[best])
    next_score = score[index + 1] if index + 1 < len(score) else -np.inf
    threshold = (
        float((score[index] + next_score) / 2)
        if np.isfinite(next_score)
        else float(np.nextafter(score[index], -np.inf))
    )
    return threshold, float(values[best])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--independent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    output_root = arguments.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    results = {}
    for role in ROLES:
        results[role] = {}
        for side, (score_key, target_key) in SIDES.items():
            oof_files = sorted(glob.glob(str(arguments.oof_root / role / "fold_*" / "*.npz")))
            test_files = sorted(
                glob.glob(
                    str(
                        arguments.independent_root
                        / "predictions"
                        / role
                        / "five_fold_mean_ensemble"
                        / "*.npz"
                    )
                )
            )
            if len(oof_files) != 560 or len(test_files) != 64:
                raise ValueError(f"Unexpected prediction count for {role} {side}")
            oof_target, oof_score = load_arrays(oof_files, score_key, target_key)
            test_target, test_score = load_arrays(test_files, score_key, target_key)
            threshold, oof_mcc = best_mcc_threshold(oof_target, oof_score)
            results[role][side] = {
                "threshold_fit_scope": "560 CV out-of-fold complexes only",
                "score_threshold": threshold,
                "oof_mcc": oof_mcc,
                "independent_test": {
                    "auroc": float(roc_auc_score(test_target, test_score)),
                    "auprc": float(average_precision_score(test_target, test_score)),
                    "mcc_at_oof_fitted_threshold": float(
                        matthews_corrcoef(test_target, test_score >= threshold)
                    ),
                    "predicted_positives": int(np.sum(test_score >= threshold)),
                    "true_positives": int(test_target.sum()),
                    "residues": int(test_target.size),
                },
            }

    report = {
        "schema_version": "mipe-oof-threshold-calibration-v1",
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "selection_policy": "Thresholds maximize MCC on CV out-of-fold predictions; independent-test labels are never used for threshold fitting.",
        "inputs": {
            "oof_manifest": sha256_file(arguments.oof_root / "manifest.json"),
            "independent_report": sha256_file(
                arguments.independent_root / "MIPE_OFFICIAL_INDEPENDENT_TEST.json"
            ),
        },
        "results": results,
        "implementation": {"script_sha256": sha256_file(Path(__file__).resolve())},
    }
    (output_root / "MIPE_OOF_THRESHOLD_CALIBRATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MIPE OOF-fitted threshold calibration",
        "",
        report["selection_policy"],
        "",
        "| Checkpoints | Side | Threshold | OOF MCC | Independent AUROC | Independent AUPRC | Independent MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        for side in SIDES:
            item = results[role][side]
            test = item["independent_test"]
            lines.append(
                f"| {role} | {side} | {item['score_threshold']:.6f} | {item['oof_mcc']:.6f} | "
                f"{test['auroc']:.6f} | {test['auprc']:.6f} | {test['mcc_at_oof_fitted_threshold']:.6f} |"
            )
    (output_root / "MIPE_OOF_THRESHOLD_CALIBRATION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
