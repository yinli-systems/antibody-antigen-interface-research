#!/usr/bin/env python3
"""Verify and aggregate ten frozen FrameContact validation pair dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEEDS = (17, 42, 73, 101, 137)
METRICS = (
    "hit_at_1_rate_macro",
    "recall_at_1_macro",
    "recall_at_1_micro",
    "mrr_first_positive_macro",
    "mean_first_positive_rank",
)
BOOTSTRAP_SEED = 20_260_821
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap(values: list[float]) -> dict[str, float | int | str]:
    generator = random.Random(BOOTSTRAP_SEED)
    replicates = [
        statistics.fmean(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "estimate": statistics.fmean(values),
        "ci95_lower": quantile(replicates, 0.025),
        "ci95_upper": quantile(replicates, 0.975),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "paired_training_seed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dump-root", type=Path, required=True)
    parser.add_argument("--decoy-aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_root = args.pair_dump_root.resolve()
    decoy_aggregate = args.decoy_aggregate.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    if not decoy_aggregate.is_file():
        raise FileNotFoundError(decoy_aggregate)

    members: dict[str, dict[str, Any]] = {"candidate": {}, "control": {}}
    values: dict[str, dict[str, list[float]]] = {
        arm: {metric: [] for metric in METRICS} for arm in members
    }
    for arm in members:
        for seed in SEEDS:
            root = pair_root / arm / f"seed-{seed}"
            manifest_path = root / "FROZEN_PAIR_MANIFEST.json"
            receipt_path = root / "RUN_RECEIPT.json"
            prediction_path = root / "FROZEN_PAIR_PREDICTIONS.npz"
            manifest = read_json(manifest_path)
            receipt = read_json(receipt_path)
            if manifest.get("status") != "complete" or receipt.get("status") != "complete":
                raise ValueError(f"Incomplete member: {root}")
            if manifest.get("arm") != arm or int(manifest.get("seed")) != seed:
                raise ValueError(f"Arm/seed mismatch: {root}")
            expected_artifacts = receipt["artifacts"]
            actual_manifest_hash = sha256_file(manifest_path)
            actual_prediction_hash = sha256_file(prediction_path)
            if actual_manifest_hash != expected_artifacts[manifest_path.name]:
                raise ValueError(f"Manifest hash mismatch: {root}")
            if actual_prediction_hash != expected_artifacts[prediction_path.name]:
                raise ValueError(f"Prediction hash mismatch: {root}")
            summary = manifest["flat_pair_rank_summary"]
            if int(summary["complexes"]) != 549 or int(summary["positive_eligible_complexes"]) != 549:
                raise ValueError(f"Unexpected validation inventory: {root}")
            for metric in METRICS:
                value = float(summary[metric])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {metric}: {root}")
                values[arm][metric].append(value)
            members[arm][str(seed)] = {
                "root": str(root),
                "manifest_sha256": actual_manifest_hash,
                "prediction_sha256": actual_prediction_hash,
                "prediction_bytes": prediction_path.stat().st_size,
                "source_result_sha256": manifest["source"]["result_sha256"],
                "selected_checkpoint_sha256": manifest["source"][
                    "selected_checkpoint_sha256"
                ],
                "summary": summary,
            }

    arm_summary = {
        arm: {metric: summarize(metric_values) for metric, metric_values in metrics.items()}
        for arm, metrics in values.items()
    }
    paired_deltas: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [
            values["candidate"][metric][index] - values["control"][metric][index]
            for index in range(len(SEEDS))
        ]
        paired_deltas[metric] = {
            "candidate_minus_control_by_seed": {
                str(seed): delta for seed, delta in zip(SEEDS, deltas)
            },
            "summary": summarize(deltas),
            "paired_bootstrap": bootstrap(deltas),
        }

    created_at = datetime.now(timezone.utc).isoformat()
    aggregate = {
        "format_version": "framecontact_pair_dump_aggregate_v1",
        "status": "complete",
        "created_at_utc": created_at,
        "scope": "SAbDab2 double-cold development validation; five paired seeds",
        "claim_boundary": (
            "Test and external roles remained sealed; no benchmark-improvement claim is authorized."
        ),
        "metric_definition": {
            "ranking_unit": "all antibody-antigen residue pairs within one complex",
            "recall_at_1": "top-one true contacts divided by all contacts in the complex",
            "mrr_first_positive": "reciprocal rank of the first true contact",
            "tie_break": "stable descending score, then antibody-major row order",
        },
        "authority": {
            "pair_dump_root": str(pair_root),
            "decoy_aggregate": str(decoy_aggregate),
            "decoy_aggregate_sha256": sha256_file(decoy_aggregate),
            "all_prediction_and_manifest_hashes_verified": True,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "members": members,
        "arm_summary": arm_summary,
        "paired_deltas": paired_deltas,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        aggregate_path = temporary / "FRAMECONTACT_PAIR_RANK_AGGREGATE.json"
        report_path = temporary / "FRAMECONTACT_PAIR_RANK_REPORT.md"
        write_json(aggregate_path, aggregate)
        lines = [
            "# FrameContact pair-rank audit",
            "",
            "Five paired hard-decoy/control seeds on sealed double-cold development validation.",
            "All ten raw-logit/label dumps and manifests passed SHA-256 verification.",
            "",
            "| metric | hard-decoy mean | control mean | paired delta | 95% bootstrap CI |",
            "|---|---:|---:|---:|---:|",
        ]
        for metric in METRICS:
            delta = paired_deltas[metric]["paired_bootstrap"]
            lines.append(
                f"| {metric} | {arm_summary['candidate'][metric]['mean']:.8f} | "
                f"{arm_summary['control'][metric]['mean']:.8f} | "
                f"{delta['estimate']:+.8f} | [{delta['ci95_lower']:+.8f}, "
                f"{delta['ci95_upper']:+.8f}] |"
            )
        lines.extend(
            [
                "",
                "Positive delta favors hard-decoy for Hit@1, Recall@1, and MRR; "
                "negative delta favors hard-decoy for mean first-positive rank.",
                "",
                "Test and external roles remained sealed. This is development evidence only.",
                "",
            ]
        )
        report_path.write_text("\n".join(lines), encoding="utf-8")
        receipt = {
            "format_version": "framecontact_pair_dump_aggregate_receipt_v1",
            "status": "complete",
            "created_at_utc": created_at,
            "output_root": str(output_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "artifacts": {
                aggregate_path.name: sha256_file(aggregate_path),
                report_path.name: sha256_file(report_path),
            },
        }
        write_json(temporary / "RUN_RECEIPT.json", receipt)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output_root": str(output_root), "paired_deltas": paired_deltas}, indent=2))


if __name__ == "__main__":
    main()
