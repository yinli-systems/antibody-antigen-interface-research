#!/usr/bin/env python3
"""Independently check retained MIPE threshold counts and source semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite verification")
    root, recovery = args.interaction_root, args.recovery_root
    result = json.loads((recovery / "MIPE_CONTACT_THRESHOLD_RESULT.json").read_text())
    receipt = json.loads((recovery / "MIPE_CONTACT_THRESHOLD_RECEIPT.json").read_text())
    require(digest(recovery / "MIPE_CONTACT_THRESHOLD_RESULT.json") == receipt["result_sha256"],
            "Result hash mismatch")
    require(digest(recovery / "MIPE_CONTACT_THRESHOLD_PER_COMPLEX.csv") == receipt["per_complex_csv_sha256"],
            "Per-complex CSV hash mismatch")
    require(result["threshold"] == receipt["threshold"] == 0.5, "Threshold is not fixed 0.5")
    summary = json.loads((root / "summary.json").read_text())
    require(digest(root / "summary.json") == result["source"]["summary_sha256"], "Source summary changed")
    with (recovery / "MIPE_CONTACT_THRESHOLD_PER_COMPLEX.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(r["fold"], r["complex_code"]): r for r in rows}
    require(len(rows) == len(by_key) == 560, "CSV count or uniqueness mismatch")
    names = {"mean_bidirectional": "interaction_score_mean",
             "antigen_to_antibody": "ag_to_ab", "antibody_to_antigen": "ab_to_ag"}
    totals = {name: np.zeros(4, dtype=np.int64) for name in names}
    tree = hashlib.sha256()
    seen = set()
    maximum_mean_error = maximum_row_error = 0.0
    files = sorted(root.glob("fold_*/*.npz"))
    require(len(files) == 560, "Unexpected number of raw maps")
    for fold in range(5):
        require(sum(p.parent.name == f"fold_{fold}" for p in files) == 112, "Unexpected fold count")
    for path in files:
        tree.update(f"{digest(path)}  {path.stat().st_size}  {path.relative_to(root)}\n".encode())
        with np.load(path, allow_pickle=False) as arrays:
            code = str(arrays["complex_code"].item())
            require(code not in seen, "Duplicate held-out complex")
            seen.add(code)
            row = by_key[(path.parent.name, code)]
            target = arrays["interaction_target"]
            require(target.ndim == 2 and target.size > 0 and np.isin(target, [0, 1]).all(),
                    "Invalid contact labels")
            require(int(row["pairs"]) == target.size and int(row["positives"]) == int(target.sum()),
                    "CSV target totals mismatch")
            forward, reverse = arrays["ag_to_ab"], arrays["ab_to_ag"]
            require(forward.shape == reverse.T.shape == target.shape, "Directional axes mismatch")
            mean_error = float(np.max(np.abs(arrays["interaction_score_mean"] - (forward + reverse.T) / 2)))
            row_error = max(float(np.max(np.abs(forward.sum(1) - 1))),
                            float(np.max(np.abs(reverse.sum(1) - 1))))
            require(mean_error <= 1e-6 and row_error <= 1e-4, "Mean or attention normalization mismatch")
            maximum_mean_error = max(maximum_mean_error, mean_error)
            maximum_row_error = max(maximum_row_error, row_error)
            for name, field in names.items():
                score = arrays[field].T if name == "antibody_to_antigen" else arrays[field]
                require(np.isfinite(score).all() and ((0 <= score) & (score <= 1)).all(), "Invalid score range")
                # Independent encoded-confusion implementation: 0=TN, 1=FP, 2=FN, 3=TP.
                bins = np.bincount((target.astype(np.int64) * 2 + (score >= 0.5)).ravel(), minlength=4)
                counts = bins[[3, 1, 0, 2]]
                require(counts.tolist() == [int(row[f"{name}_{k}"]) for k in ("tp", "fp", "tn", "fn")],
                        "Per-complex confusion mismatch")
                totals[name] += counts
    require(tree.hexdigest() == result["source"]["artifact_tree_sha256"] == receipt["artifact_tree_sha256"],
            "Raw source tree hash mismatch")
    for name, values in totals.items():
        tp, fp, tn, fn = map(int, values)
        require(tp + fp + tn + fn == 11431418 and tp + fn == 18316, "Pooled pair totals mismatch")
        expected = result["overall"][name]
        require(values.tolist() == [expected[k] for k in ("tp", "fp", "tn", "fn")], "Aggregate counts differ")
        calculated = {"recall": tp / (tp + fn), "precision": tp / (tp + fp),
                      "mcc": (tp * tn - fp * fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))}
        for key, value in calculated.items():
            require(math.isclose(value, expected[key], rel_tol=1e-12, abs_tol=1e-12), "Micro metric mismatch")
    audit = {
        "status": "PASS_MIPE_THRESHOLD_RECOVERY_SOURCE_AUDIT",
        "verifier_sha256": digest(Path(__file__)),
        "source_summary_sha256": digest(root / "summary.json"),
        "result_sha256": receipt["result_sha256"], "source_artifact_tree_sha256": tree.hexdigest(),
        "complexes": len(seen), "pairs": 11431418, "positive_pairs": 18316,
        "threshold": 0.5, "threshold_comparison": ">=", "aggregation": "micro pooled confusion counts",
        "source_folds": [{k: f[k] for k in ("fold", "checkpoint", "checkpoint_epoch")}
                         for f in summary["folds"]],
        "checkpoint_provenance": "Paths and epochs from retained export summary; original checkpoint byte hashes were not recorded there.",
        "score_source_notes": summary["notes"],
        "maximum_bidirectional_mean_error": maximum_mean_error,
        "maximum_directional_row_sum_error": maximum_row_error,
        "historical_pooled_rank_metrics": summary["overall_interaction_metrics"],
        "scope": "560 held-out complexes in formal_seed42 retained export; not the SAbDab2 455 test set",
        "reporting": "May be reported as MIPE attention-readout contact adaptation at fixed 0.5, with source cohort and checkpoint track stated. No threshold refit was performed.",
        "residue_table_threshold_correction": "Historical Recall 0.5523/0.7954 uses fold-specific validation-fitted thresholds, not fixed 0.5.",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
