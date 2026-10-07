#!/usr/bin/env python3
"""Estimate prospective PDB-sample precision from exposed SABLE AP dispersion.

SABLE is already analyst-exposed and is used here only as a design prior.  The
program reads aggregate per-entry AP dictionaries already emitted by the frozen
measurement; it does not read structures, labels, logits, or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


SEEDS = (17, 29, 43)
ROUTE = {
    "contact": "baseline",
    "epitope": "edge_topk2_supervised",
    "paratope": "baseline",
}
SAMPLE_SIZES = (10, 20, 30, 40, 60, 80)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_routed_pdb_values(root: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    per_task_entries: dict[str, list[dict[str, float]]] = {task: [] for task in ROUTE}
    source_sha: dict[str, str] = {}
    for task, arm in ROUTE.items():
        for seed in SEEDS:
            path = root / arm / f"seed-{seed}" / "SABLE_COMPONENT_RESULT.json"
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"component result is missing or unsafe: {path}")
            result = json.loads(path.read_text())
            if (
                result.get("status")
                != "PASS_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_COMPONENT_MEASUREMENT"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or result.get("audit", {}).get("fresh_blind_test") is not False
                or result.get("audit", {}).get("measurement_only") is not True
            ):
                raise ValueError(f"component authority differs: {path}")
            values = result["metrics"][task]["per_entry_average_precision"]
            if not values or any(not math.isfinite(float(value)) for value in values.values()):
                raise ValueError(f"per-entry AP differs: {path}")
            per_task_entries[task].append({key: float(value) for key, value in values.items()})
            source_sha[f"{arm}/seed-{seed}"] = sha256_file(path)

    routed: dict[str, dict[str, float]] = {}
    for task, seed_values in per_task_entries.items():
        entry_ids = set(seed_values[0])
        if any(set(values) != entry_ids for values in seed_values[1:]):
            raise ValueError(f"seed entry sets differ: {task}")
        pdb_entries: dict[str, list[float]] = {}
        for entry_id in sorted(entry_ids):
            pdb_id = entry_id.split("_", 1)[0].lower()
            seed_average = mean(values[entry_id] for values in seed_values)
            pdb_entries.setdefault(pdb_id, []).append(seed_average)
        routed[task] = {
            pdb_id: mean(values) for pdb_id, values in sorted(pdb_entries.items())
        }
    if len({tuple(values) for values in (sorted(task) for task in routed.values())}) != 1:
        raise ValueError("routed task PDB sets differ")
    return routed, source_sha


def simulate(
    values: dict[str, float], sample_size: int, rng: np.random.Generator
) -> dict[str, float]:
    population = np.asarray([values[pdb_id] for pdb_id in sorted(values)], dtype=np.float64)
    full_mean = float(population.mean())
    widths: list[float] = []
    absolute_errors: list[float] = []
    trials = 500
    bootstrap_replicates = 500
    for _ in range(trials):
        sample = rng.choice(population, size=sample_size, replace=False)
        absolute_errors.append(abs(float(sample.mean()) - full_mean))
        bootstrap_indices = rng.integers(
            0, sample_size, size=(bootstrap_replicates, sample_size)
        )
        bootstrap = sample[bootstrap_indices].mean(axis=1)
        widths.append(
            float(np.quantile(bootstrap, 0.975) - np.quantile(bootstrap, 0.025))
        )
    return {
        "sample_size_unique_pdbs": sample_size,
        "trials": trials,
        "bootstrap_replicates_per_trial": bootstrap_replicates,
        "median_ci95_width": median(widths),
        "ci95_width_p90": quantile(widths, 0.90),
        "median_absolute_error_to_full_exposed_sable_mean": median(absolute_errors),
        "absolute_error_p90": quantile(absolute_errors, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    routed, source_sha = load_routed_pdb_values(args.components_root)
    unique_pdbs = len(next(iter(routed.values())))
    sizes = [size for size in SAMPLE_SIZES if size <= unique_pdbs]
    analysis = {
        "format_version": "framecontact_prospective_temporal_gate_precision_audit_v2",
        "status": "PASS_EXPOSED_SABLE_DESIGN_PRIOR_ONLY",
        "evidence_boundary": {
            "sable_is_analyst_exposed": True,
            "scientific_test_result": False,
            "used_to_select_or_tune_model": False,
            "future_temporal_labels_read": False,
            "bound_structures_or_labels_read_by_this_audit": False,
            "protocol_threshold_automatically_changed": False,
        },
        "design": {
            "source_unique_pdbs": unique_pdbs,
            "tasks_use_frozen_seed_matched_route": ROUTE,
            "three_fixed_seeds_averaged_before_pdb_aggregation": True,
            "multiple_entries_from_same_pdb_averaged_before_sampling": True,
            "subsampling_without_replacement": True,
            "bootstrap_with_replacement_within_each_subsample": True,
            "random_seed": 20260831,
        },
        "full_exposed_sable_pdb_equal_means": {
            task: mean(values.values()) for task, values in routed.items()
        },
        "sample_size_precision": {},
        "source_component_sha256": source_sha,
    }
    for task, values in routed.items():
        task_rng = np.random.default_rng(
            20260831 + list(ROUTE).index(task) * 100000
        )
        analysis["sample_size_precision"][task] = [
            simulate(values, size, task_rng) for size in sizes
        ]
    analysis_path = args.output_root / "TEMPORAL_GATE_PRECISION_AUDIT_V2.json"
    analysis_sha = publish(analysis_path, analysis)
    receipt = {
        "format_version": "framecontact_prospective_temporal_gate_precision_audit_receipt_v2",
        "status": analysis["status"],
        "analysis_sha256": analysis_sha,
        "source_unique_pdbs": unique_pdbs,
        "future_temporal_labels_read": False,
        "protocol_threshold_automatically_changed": False,
    }
    receipt_sha = publish(
        args.output_root / "TEMPORAL_GATE_PRECISION_AUDIT_V2_RECEIPT.json", receipt
    )
    print(json.dumps({"analysis_sha256": analysis_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
