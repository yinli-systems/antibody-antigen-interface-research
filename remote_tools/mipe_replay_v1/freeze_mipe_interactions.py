#!/usr/bin/env python3
"""Validate and freeze MIPE held-out interaction exports without rerunning MIPE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_KEYS = {
    "ab_to_ag",
    "ag_to_ab",
    "antibody_score",
    "antibody_target",
    "antigen_score",
    "antigen_target",
    "complex_code",
    "interaction_score_mean",
    "interaction_target",
}
EXPECTED_FOLDS = 5
EXPECTED_PER_FOLD = 112
EXPECTED_TOTAL = EXPECTED_FOLDS * EXPECTED_PER_FOLD


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scalar_text(array: np.ndarray, label: str) -> str:
    if array.shape != ():
        raise ValueError(f"{label} must be a scalar; got {array.shape}")
    return str(array.item())


def validate_binary(array: np.ndarray, label: str) -> None:
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{label} is not binary: {unique.tolist()}")


def validate_finite(array: np.ndarray, label: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")


def flat_rank_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    scores_flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    target_flat = np.asarray(target, dtype=np.uint8).reshape(-1)
    positives = int(target_flat.sum())
    if positives <= 0:
        raise ValueError("Cannot rank a complex with zero positive contact pairs")
    # Stable row-major tie handling makes the result exactly reproducible.
    order = np.argsort(-scores_flat, kind="stable")
    ranked_target = target_flat[order]
    positive_ranks = np.flatnonzero(ranked_target) + 1
    hit_at_1 = int(ranked_target[0])
    first_positive_rank = int(positive_ranks[0])
    return {
        "pairs": int(target_flat.size),
        "positives": positives,
        "hit_at_1": hit_at_1,
        "recall_at_1": hit_at_1 / positives,
        "first_positive_rank": first_positive_rank,
        "reciprocal_rank": 1.0 / first_positive_rank,
    }


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return float(sum(values) / len(values))


def summarize_rank(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    return {
        "complexes": len(records),
        "pairs": sum(int(record["pairs"]) for record in records),
        "positives": sum(int(record["positives"]) for record in records),
        "hit_at_1_complexes": sum(int(record["hit_at_1"]) for record in records),
        "hit_at_1_rate_macro": mean([float(record["hit_at_1"]) for record in records]),
        "recall_at_1_macro": mean([float(record["recall_at_1"]) for record in records]),
        "recall_at_1_micro": (
            sum(int(record["hit_at_1"]) for record in records)
            / sum(int(record["positives"]) for record in records)
        ),
        "mrr_first_positive_macro": mean(
            [float(record["reciprocal_rank"]) for record in records]
        ),
        "mean_first_positive_rank": mean(
            [float(record["first_positive_rank"]) for record in records]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction-root", type=Path, required=True)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interaction_root = args.interaction_root.resolve()
    evaluation_json = args.evaluation_json.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {output_root}")
    if not evaluation_json.is_file():
        raise FileNotFoundError(evaluation_json)

    files_by_fold: dict[str, list[Path]] = {}
    all_files: list[Path] = []
    for fold_index in range(EXPECTED_FOLDS):
        fold_name = f"fold_{fold_index}"
        files = sorted((interaction_root / fold_name).glob("*.npz"))
        if len(files) != EXPECTED_PER_FOLD:
            raise ValueError(f"{fold_name}: expected {EXPECTED_PER_FOLD}, got {len(files)}")
        files_by_fold[fold_name] = files
        all_files.extend(files)
    unexpected = sorted(
        path for path in interaction_root.glob("fold_*/*.npz") if path not in set(all_files)
    )
    if unexpected or len(all_files) != EXPECTED_TOTAL:
        raise ValueError(f"Unexpected NPZ inventory: total={len(all_files)}, extras={unexpected}")

    file_records: list[dict[str, Any]] = []
    rank_records: dict[str, list[dict[str, float | int]]] = {
        "ag_to_ab": [],
        "ab_to_ag_transposed": [],
        "mean_bidirectional": [],
    }
    complex_codes: set[str] = set()
    max_mean_error = 0.0
    max_ag_row_sum_error = 0.0
    max_ab_row_sum_error = 0.0
    total_bytes = 0

    for fold_name, files in files_by_fold.items():
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                if keys != EXPECTED_KEYS:
                    raise ValueError(
                        f"{path}: keys differ; missing={sorted(EXPECTED_KEYS - keys)}, "
                        f"extra={sorted(keys - EXPECTED_KEYS)}"
                    )
                code = scalar_text(archive["complex_code"], f"{path}:complex_code")
                if code in complex_codes:
                    raise ValueError(f"Duplicate complex_code: {code}")
                complex_codes.add(code)

                target = archive["interaction_target"]
                mean_score = archive["interaction_score_mean"]
                ag_to_ab = archive["ag_to_ab"]
                ab_to_ag = archive["ab_to_ag"]
                antigen_score = archive["antigen_score"]
                antibody_score = archive["antibody_score"]
                antigen_target = archive["antigen_target"]
                antibody_target = archive["antibody_target"]
                if target.ndim != 2:
                    raise ValueError(f"{path}: interaction_target must be 2-D")
                antigen_length, antibody_length = target.shape
                expected_shapes = {
                    "interaction_score_mean": target.shape,
                    "ag_to_ab": target.shape,
                    "ab_to_ag": (antibody_length, antigen_length),
                    "antigen_score": (antigen_length,),
                    "antigen_target": (antigen_length,),
                    "antibody_score": (antibody_length,),
                    "antibody_target": (antibody_length,),
                }
                for key, shape in expected_shapes.items():
                    if archive[key].shape != shape:
                        raise ValueError(f"{path}:{key}: expected {shape}, got {archive[key].shape}")
                for key in (
                    "interaction_score_mean",
                    "ag_to_ab",
                    "ab_to_ag",
                    "antigen_score",
                    "antibody_score",
                ):
                    validate_finite(archive[key], f"{path}:{key}")
                for key in ("interaction_target", "antigen_target", "antibody_target"):
                    validate_binary(archive[key], f"{path}:{key}")
                if not np.array_equal(antigen_target, target.max(axis=1)):
                    raise ValueError(f"{path}: antigen target disagrees with contact rows")
                if not np.array_equal(antibody_target, target.max(axis=0)):
                    raise ValueError(f"{path}: antibody target disagrees with contact columns")

                reconstructed_mean = (ag_to_ab + ab_to_ag.T) / 2.0
                mean_error = float(np.max(np.abs(mean_score - reconstructed_mean)))
                ag_row_error = float(np.max(np.abs(ag_to_ab.sum(axis=1) - 1.0)))
                ab_row_error = float(np.max(np.abs(ab_to_ag.sum(axis=1) - 1.0)))
                if mean_error > 1e-6:
                    raise ValueError(f"{path}: bidirectional mean max error {mean_error}")
                if ag_row_error > 1e-4 or ab_row_error > 1e-4:
                    raise ValueError(
                        f"{path}: attention normalization errors row={ag_row_error}, "
                        f"reverse_row={ab_row_error}"
                    )
                max_mean_error = max(max_mean_error, mean_error)
                max_ag_row_sum_error = max(max_ag_row_sum_error, ag_row_error)
                max_ab_row_sum_error = max(max_ab_row_sum_error, ab_row_error)

                directions = {
                    "ag_to_ab": ag_to_ab,
                    "ab_to_ag_transposed": ab_to_ag.T,
                    "mean_bidirectional": mean_score,
                }
                rank_by_direction = {
                    name: flat_rank_metrics(scores, target) for name, scores in directions.items()
                }
                for name, record in rank_by_direction.items():
                    rank_records[name].append(record)

                size = path.stat().st_size
                total_bytes += size
                file_records.append(
                    {
                        "relative_path": str(path.relative_to(interaction_root)),
                        "fold": fold_name,
                        "complex_code": code,
                        "sha256": sha256_file(path),
                        "bytes": size,
                        "antigen_length": antigen_length,
                        "antibody_length": antibody_length,
                        "contact_pairs": int(target.sum()),
                        "rank_metrics": rank_by_direction,
                    }
                )

    evaluated = json.loads(evaluation_json.read_text(encoding="utf-8"))
    evaluation_status = evaluated.get("status")
    if evaluation_status not in {"complete", "complete_create_only_without_policy_refit"}:
        raise ValueError(f"Evaluation status is not complete: {evaluation_status!r}")
    summary = {
        name: summarize_rank(records) for name, records in rank_records.items()
    }
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "mipe-frozen-interactions-v1",
        "status": "complete",
        "created_at_utc": created_at,
        "scope": "MIPE official five-fold held-out exports at epoch 800",
        "source": {
            "interaction_root": str(interaction_root),
            "evaluation_json": str(evaluation_json),
            "evaluation_json_sha256": sha256_file(evaluation_json),
        },
        "inventory": {
            "folds": EXPECTED_FOLDS,
            "files_per_fold": EXPECTED_PER_FOLD,
            "files": len(file_records),
            "unique_complex_codes": len(complex_codes),
            "total_bytes": total_bytes,
        },
        "validation": {
            "expected_keys_exact": True,
            "all_numeric_arrays_finite": True,
            "all_targets_binary": True,
            "residue_targets_equal_contact_marginals": True,
            "max_bidirectional_mean_abs_error": max_mean_error,
            "max_ag_to_ab_row_sum_abs_error": max_ag_row_sum_error,
            "max_ab_to_ag_row_sum_abs_error": max_ab_row_sum_error,
        },
        "metric_definition": {
            "ranking_unit": "all antigen-antibody residue pairs within one complex",
            "tie_break": "stable descending score, then row-major pair order",
            "recall_at_1": "positive pairs among the top one divided by positive pairs in the complex",
            "hit_at_1": "whether the top-scored pair is a true contact",
            "mrr_first_positive": "reciprocal rank of the first true contact in the flattened matrix",
            "aggregation": "macro over the 560 held-out complexes unless marked micro",
            "claim_boundary": "attention-derived ranking scores; not calibrated contact probabilities",
        },
        "flat_pair_rank_summary": summary,
        "files": file_records,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        manifest_path = temporary / "MIPE_FROZEN_INTERACTION_MANIFEST.json"
        report_path = temporary / "MIPE_FROZEN_INTERACTION_REPORT.md"
        write_json(manifest_path, manifest)
        lines = [
            "# MIPE frozen interaction audit",
            "",
            f"Created: `{created_at}`",
            "",
            f"Validated `{len(file_records)}` NPZ files: five folds x 112 held-out complexes.",
            "Every file has an individual SHA-256 in the manifest.",
            "",
            "## Validation",
            "",
            "- Exact expected arrays, shapes, finite values, and binary targets: PASS",
            "- Residue labels equal row/column marginals of the contact matrix: PASS",
            "- Bidirectional mean and directional attention normalization: PASS",
            "",
            "## Flattened pair ranking",
            "",
            "| score | pair Recall@1 macro | pair Hit@1 | pair MRR | mean first-positive rank |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in ("ag_to_ab", "ab_to_ag_transposed", "mean_bidirectional"):
            values = summary[name]
            lines.append(
                f"| {name} | {values['recall_at_1_macro']:.8f} | "
                f"{values['hit_at_1_rate_macro']:.8f} | "
                f"{values['mrr_first_positive_macro']:.8f} | "
                f"{values['mean_first_positive_rank']:.3f} |"
            )
        lines.extend(
            [
                "",
                "Recall@1 and MRR rank all residue pairs within each complex. "
                "Ties use stable row-major order.",
                "",
                "## Claim boundary",
                "",
                "These are attention-derived ranking scores, not calibrated pair-contact probabilities. "
                "The linked held-out evaluation remains authoritative for AUPRC, AUROC, calibration, "
                "residue metrics, and bootstrap intervals.",
                "",
            ]
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        receipt = {
            "schema_version": "mipe-frozen-interactions-receipt-v1",
            "status": "complete",
            "created_at_utc": created_at,
            "output_root": str(output_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "artifacts": {
                manifest_path.name: sha256_file(manifest_path),
                report_path.name: sha256_file(report_path),
            },
        }
        write_json(temporary / "RUN_RECEIPT.json", receipt)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({"output_root": str(output_root), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
