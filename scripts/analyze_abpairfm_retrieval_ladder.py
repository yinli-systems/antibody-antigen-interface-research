#!/usr/bin/env python3
"""Compare frozen development retrieval predictions with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable


METRICS = ("mrr", "recall_at_1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def average_tie_rank(cognate_score: float, scores: Iterable[float]) -> float:
    values = list(scores)
    greater = sum(value > cognate_score for value in values)
    tied_decoys = sum(value == cognate_score for value in values) - 1
    if tied_decoys < 0:
        raise ValueError("Cognate score is absent from candidates")
    return 1.0 + greater + 0.5 * tied_decoys


def _score(row: dict[str, Any]) -> float:
    available = [
        name
        for name in ("score", "raw_logit", "probability_logit")
        if name in row
    ]
    if len(available) != 1:
        raise ValueError(f"Prediction row has ambiguous score fields: {available}")
    value = float(row[available[0]])
    if not math.isfinite(value):
        raise ValueError("Prediction score is not finite")
    return value


def canonical_pair_metrics(path: Path, dataset: str) -> dict[str, dict[str, float]]:
    """Reproduce unique-endpoint bidirectional metrics per canonical exact pair."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != dataset:
                raise ValueError("Prediction dataset differs")
            key = (
                str(row["canonical_pair_sha256"]),
                str(row["anchor_instance_id"]),
                str(row["direction"]),
            )
            grouped.setdefault(key, []).append(row)
    if not grouped:
        raise ValueError("Prediction file is empty")

    anchors: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for (canonical, anchor, direction), rows in grouped.items():
        cognates = [row for row in rows if row["is_cognate"] is True]
        if len(cognates) != 1:
            raise ValueError("Every anchor direction must have exactly one cognate")
        cognate_score = _score(cognates[0])
        collapsed: dict[str, float] = {}
        for row in rows:
            endpoint = str(row["candidate_endpoint_sha256"])
            value = _score(row)
            previous = collapsed.setdefault(endpoint, value)
            if previous != value:
                raise ValueError("Exact endpoint duplicates received different scores")
        rank = average_tie_rank(cognate_score, collapsed.values())
        anchors.setdefault((canonical, anchor), {})[direction] = {
            "mrr": 1.0 / rank,
            "recall_at_1": float(rank == 1.0),
        }

    required_directions = {
        "fixed_antibody_wrong_antigen",
        "fixed_antigen_wrong_antibody",
    }
    canonical_values: dict[str, dict[str, list[float]]] = {}
    for (canonical, _anchor), directions in anchors.items():
        if set(directions) != required_directions:
            raise ValueError("Anchor is missing one retrieval direction")
        target = canonical_values.setdefault(canonical, {metric: [] for metric in METRICS})
        for metric in METRICS:
            target[metric].append(
                math.fsum(directions[direction][metric] for direction in required_directions)
                / 2.0
            )
    return {
        canonical: {
            metric: math.fsum(values[metric]) / len(values[metric])
            for metric in METRICS
        }
        for canonical, values in canonical_values.items()
    }


def paired_bootstrap(
    reference: dict[str, dict[str, float]],
    comparator: dict[str, dict[str, float]],
    metric: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(reference)
    if keys != sorted(comparator):
        raise ValueError("Paired models have different canonical exact-pair units")
    deltas = [reference[key][metric] - comparator[key][metric] for key in keys]
    estimate = math.fsum(deltas) / len(deltas)
    generator = random.Random(seed)
    draws = [
        math.fsum(deltas[generator.randrange(len(deltas))] for _ in deltas)
        / len(deltas)
        for _ in range(replicates)
    ]
    return {
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "delta_reference_minus_comparator": estimate,
        "paired_unit_count": len(keys),
        "probability_delta_gt_zero": sum(draw > 0.0 for draw in draws) / replicates,
    }


def _model_argument(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("Model must be NAME=ASEP_PATH,SABDAB2_PATH")
    paths = parts[1].split(",")
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("Model must provide exactly two paths")
    return parts[0], Path(paths[0]), Path(paths[1])


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=_model_argument, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    names = [name for name, _asep, _sabdab2 in arguments.model]
    if len(names) != len(set(names)) or arguments.reference not in names:
        raise ValueError("Model names must be unique and include the reference")

    values: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "asep": {},
        "sabdab2": {},
    }
    inputs: dict[str, dict[str, Any]] = {}
    for name, asep_path, sabdab2_path in arguments.model:
        inputs[name] = {}
        for dataset, path in (("asep", asep_path), ("sabdab2", sabdab2_path)):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Prediction input is not a regular file: {path}")
            values[dataset][name] = canonical_pair_metrics(path, dataset)
            inputs[name][dataset] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    output: dict[str, Any] = {
        "authorization": {"development_only": True, "test_access": False},
        "claim_boundary": "Paired comparison of frozen development retrieval predictions. Scores from different model families are used only for within-panel ranking; no probability calibration or test claim.",
        "format_version": "abpairfm_retrieval_ladder_analysis_v1",
        "inputs": inputs,
        "models": {},
        "paired_deltas": {},
        "reference": arguments.reference,
        "status": "PASS_PAIRED_DEVELOPMENT_ANALYSIS",
    }
    for dataset, models in values.items():
        output["models"][dataset] = {}
        for name, pairs in models.items():
            output["models"][dataset][name] = {
                metric: math.fsum(row[metric] for row in pairs.values()) / len(pairs)
                for metric in METRICS
            } | {"canonical_exact_pair_count": len(pairs)}
        reference = models[arguments.reference]
        output["paired_deltas"][dataset] = {}
        for name, comparator in models.items():
            if name == arguments.reference:
                continue
            output["paired_deltas"][dataset][name] = {
                metric: paired_bootstrap(
                    reference,
                    comparator,
                    metric,
                    replicates=arguments.replicates,
                    seed=arguments.seed,
                )
                for metric in METRICS
            }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
