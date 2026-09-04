#!/usr/bin/env python3
"""Recover fixed-threshold diagnostics from retained MIPE interaction maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SCORE_FIELDS = {
    "mean_bidirectional": "interaction_score_mean",
    "antigen_to_antibody": "ag_to_ab",
    "antibody_to_antigen": "ab_to_ag",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else None
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "mcc": mcc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-complexes", type=int, default=560)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interaction_root = args.interaction_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1]")
    files = sorted(interaction_root.glob("fold_*/*.npz"))
    if len(files) != args.expected_complexes:
        raise ValueError(f"Expected {args.expected_complexes} NPZ files, got {len(files)}")

    totals = {name: [0, 0, 0, 0] for name in SCORE_FIELDS}
    rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            target = np.asarray(archive["interaction_target"], dtype=bool)
            if target.ndim != 2 or target.size == 0:
                raise ValueError(f"Invalid target matrix: {path}")
            if not np.isfinite(archive["interaction_score_mean"]).all():
                raise ValueError(f"Non-finite score matrix: {path}")
            complex_code = str(np.asarray(archive["complex_code"]).item())
            row: dict[str, Any] = {
                "fold": path.parent.name,
                "complex_code": complex_code,
                "pairs": int(target.size),
                "positives": int(target.sum()),
            }
            for name, field in SCORE_FIELDS.items():
                score = np.asarray(archive[field], dtype=np.float64)
                if name == "antibody_to_antigen":
                    score = score.T
                if score.shape != target.shape or not np.isfinite(score).all():
                    raise ValueError(f"Score/target mismatch for {field}: {path}")
                prediction = score >= args.threshold
                tp = int(np.logical_and(prediction, target).sum())
                fp = int(np.logical_and(prediction, ~target).sum())
                tn = int(np.logical_and(~prediction, ~target).sum())
                fn = int(np.logical_and(~prediction, target).sum())
                values = metrics(tp, fp, tn, fn)
                totals[name] = [left + right for left, right in zip(totals[name], (tp, fp, tn, fn))]
                for key, value in values.items():
                    row[f"{name}_{key}"] = value
                row[f"{name}_maximum_score"] = float(score.max())
            rows.append(row)
        artifact_records.append(
            {
                "path": str(path.relative_to(interaction_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    output_root.mkdir(parents=True)
    csv_path = output_root / "MIPE_CONTACT_THRESHOLD_PER_COMPLEX.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    artifact_tree = hashlib.sha256()
    for record in artifact_records:
        artifact_tree.update(
            f"{record['sha256']}  {record['size_bytes']}  {record['path']}\n".encode()
        )
    result = {
        "format_version": "mipe_contact_threshold_diagnostic_v1",
        "status": "PASS_RETAINED_MIPE_ATTENTION_THRESHOLD_DIAGNOSTIC",
        "scope": "retained five-fold held-out MIPE interaction maps",
        "threshold": args.threshold,
        "complexes": len(rows),
        "score_semantics": (
            "Head-averaged, row-normalized bidirectional cross-attention weights. "
            "These scores are not sigmoid-calibrated residue-pair contact probabilities."
        ),
        "claim_boundary": (
            "Diagnostic reconstruction only. It must not be reported as an official MIPE "
            "contact head or combined with a different MIPE checkpoint track."
        ),
        "overall": {
            name: metrics(*values) for name, values in totals.items()
        },
        "source": {
            "interaction_root": str(interaction_root),
            "artifact_count": len(artifact_records),
            "artifact_tree_sha256": artifact_tree.hexdigest(),
            "summary_sha256": (
                sha256_file(interaction_root / "summary.json")
                if (interaction_root / "summary.json").is_file()
                else None
            ),
        },
    }
    result_path = output_root / "MIPE_CONTACT_THRESHOLD_RESULT.json"
    atomic_write(result_path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    receipt = {
        "format_version": "mipe_contact_threshold_diagnostic_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(result_path),
        "per_complex_csv_sha256": sha256_file(csv_path),
        "artifact_tree_sha256": artifact_tree.hexdigest(),
        "complexes": len(rows),
        "threshold": args.threshold,
    }
    atomic_write(
        output_root / "MIPE_CONTACT_THRESHOLD_RECEIPT.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
