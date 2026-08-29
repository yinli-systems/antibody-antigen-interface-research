#!/usr/bin/env python3
"""Aggregate the development-only pair-to-node attribution matrix.

The three conditions differ only in the registered reduction from the pair
contact matrix to residue marginals.  Every run is authenticated before any
summary is produced.  This script never reads a frozen, official, or external
test artifact and never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


FORMAT = "framecontact_asep_joint_attribution_dev_v1_aggregate"
GROUPS = ("contact", "epitope", "paratope")
METRICS = (
    "auprc",
    "mean_per_complex_auprc",
    "auroc",
    "precision",
    "recall",
    "f1",
    "mcc",
)
RETRIEVAL_DIRECTIONS = (
    "fixed_antibody_wrong_antigen",
    "fixed_antigen_wrong_antibody",
)
RETRIEVAL_FIELDS = (
    "cluster_weighted_cognate_compatibility_margin",
    "top1_retrieval_accuracy",
    "mean_reciprocal_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {label}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}")
    return value


def _summary(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("Every condition requires exactly five seeds")
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _paired_bootstrap(
    left: list[float], right: list[float], *, resamples: int, seed: int
) -> dict[str, Any]:
    if len(left) != 5 or len(right) != 5:
        raise ValueError("Paired bootstrap requires five aligned seeds")
    if resamples < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples")
    deltas = [right[index] - left[index] for index in range(5)]
    rng = random.Random(seed)
    sampled = sorted(
        statistics.mean(deltas[rng.randrange(5)] for _ in range(5))
        for _ in range(resamples)
    )
    return {
        "right_minus_left_mean": statistics.mean(deltas),
        "confidence_interval_95": [
            _percentile(sampled, 0.025),
            _percentile(sampled, 0.975),
        ],
        "paired_seed_deltas": deltas,
        "resampling_unit": "seed",
        "resamples": resamples,
        "random_seed": seed,
    }


def _protocol(protocol_path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if any(path.is_symlink() or not path.is_file() for path in (protocol_path, receipt_path)):
        raise ValueError("Protocol and receipt must be regular files")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "framecontact_asep_joint_attribution_development_protocol_v1"
        or protocol.get("data", {}).get("test_role_forbidden") is not True
        or protocol.get("model_selection", {}).get("test_role_forbidden") is not True
        or receipt.get("protocol_file_sha256") != _sha256(protocol_path)
        or receipt.get("official_test_used_for_selection") is not False
        or receipt.get("official_test_scores_read_for_this_protocol") is not False
        or receipt.get("external_test_used_for_selection") is not False
    ):
        raise ValueError("Attribution protocol authority is ineligible")
    return protocol, receipt


def _history(result: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    rows = result.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Run lacks validation history: {path}")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("epoch"), bool):
            raise ValueError(f"Malformed validation history: {path}")
        epoch = int(row.get("epoch", 0))
        validation = row.get("validation")
        if epoch < 1 or epoch in output or not isinstance(validation, dict):
            raise ValueError(f"Malformed validation epoch: {path}")
        metrics = {
            group: {
                "auprc": _number(validation.get(group, {}).get("auprc"), f"{group}.auprc"),
                "mean_per_complex_auprc": _number(
                    validation.get(group, {}).get("mean_per_complex_auprc"),
                    f"{group}.mean_per_complex_auprc",
                ),
            }
            for group in GROUPS
        }
        directions = validation.get("partner_retrieval_by_direction")
        if not isinstance(directions, dict) or set(directions) != set(RETRIEVAL_DIRECTIONS):
            raise ValueError(f"Validation retrieval directions differ: {path}")
        output[epoch] = {
            "metrics": metrics,
            "minimum_bidirectional_margin": min(
                _number(
                    directions[direction].get("cluster_weighted_cognate_compatibility_margin"),
                    f"{direction}.margin",
                )
                for direction in RETRIEVAL_DIRECTIONS
            ),
        }
    return output


def _member(
    run_root: Path,
    *,
    condition: str,
    seed: int,
    protocol_sha: str,
    receipt_sha: str,
) -> dict[str, Any]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    run_receipt_path = run_root / "RUN_RECEIPT.json"
    if any(path.is_symlink() or not path.is_file() for path in (result_path, run_receipt_path)):
        raise ValueError(f"Incomplete development run: {run_root}")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if (
        run_receipt.get("result_sha256") != _sha256(result_path)
        or run_receipt.get("test_loaded") is not False
        or run_receipt.get("external_test_loaded") is not False
        or run_receipt.get("seed") != seed
    ):
        raise ValueError(f"Run receipt failed authentication: {run_root}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("dataset_backend") != "asep"
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or authority.get("protocol_file_sha256") != protocol_sha
        or authority.get("protocol_receipt_sha256") != receipt_sha
        or result.get("experiment") != condition
        or result.get("configuration", {}).get("seed") != seed
    ):
        raise ValueError(f"Run falls outside attribution authority: {run_root}")
    selected = result.get("selected_checkpoint_validation")
    if not isinstance(selected, dict):
        raise ValueError(f"Run lacks selected validation metrics: {run_root}")
    metrics = {
        group: {
            metric: _number(selected.get(group, {}).get(metric), f"{group}.{metric}")
            for metric in METRICS
        }
        for group in GROUPS
    }
    retrieval = selected.get("partner_retrieval_by_direction")
    if not isinstance(retrieval, dict) or set(retrieval) != set(RETRIEVAL_DIRECTIONS):
        raise ValueError(f"Selected retrieval directions differ: {run_root}")
    retrieval = {
        direction: {
            field: _number(retrieval[direction].get(field), f"{direction}.{field}")
            for field in RETRIEVAL_FIELDS
        }
        for direction in RETRIEVAL_DIRECTIONS
    }
    per_complex: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        raw = selected.get(group, {}).get("per_complex_auprc_by_instance")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Run lacks per-complex {group} AUPRC: {run_root}")
        per_complex[group] = {str(key): _number(value, f"{group}.{key}") for key, value in raw.items()}
        if not math.isclose(
            statistics.mean(per_complex[group].values()),
            metrics[group]["mean_per_complex_auprc"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Per-complex AUPRC does not reproduce macro mean: {run_root}")
    selected_epoch = int(result.get("selection", {}).get("best_epoch", 0))
    history = _history(result, run_root)
    if selected_epoch not in history:
        raise ValueError(f"Selected epoch is absent from validation history: {run_root}")
    return {
        "condition": condition,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "result_sha256": run_receipt["result_sha256"],
        "selected_checkpoint_sha256": run_receipt["selected_checkpoint_sha256"],
        "metrics": metrics,
        "retrieval": retrieval,
        "minimum_bidirectional_margin": min(
            retrieval[direction]["cluster_weighted_cognate_compatibility_margin"]
            for direction in RETRIEVAL_DIRECTIONS
        ),
        "per_complex_auprc": per_complex,
        "history": history,
        "authority": authority,
    }


def _condition_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(members, key=lambda item: item["seed"])
    return {
        "seed_count": len(members),
        "selected_epoch": _summary([float(item["selected_epoch"]) for item in members]),
        "metrics": {
            group: {
                metric: _summary([item["metrics"][group][metric] for item in members])
                for metric in METRICS
            }
            for group in GROUPS
        },
        "retrieval": {
            direction: {
                field: _summary([item["retrieval"][direction][field] for item in members])
                for field in RETRIEVAL_FIELDS
            }
            for direction in RETRIEVAL_DIRECTIONS
        },
        "minimum_bidirectional_margin": {
            **_summary([item["minimum_bidirectional_margin"] for item in members]),
            "positive_seed_fraction": sum(
                item["minimum_bidirectional_margin"] > 0 for item in members
            )
            / len(members),
        },
        "members": [
            {
                key: item[key]
                for key in ("seed", "selected_epoch", "result_sha256", "selected_checkpoint_sha256")
            }
            for item in members
        ],
    }


def _comparisons(
    by_condition: dict[str, list[dict[str, Any]]],
    conditions: tuple[str, ...],
    *,
    resamples: int,
) -> dict[str, Any]:
    endpoints = {
        "contact_auprc": lambda item: item["metrics"]["contact"]["auprc"],
        "contact_mean_per_complex_auprc": lambda item: item["metrics"]["contact"]["mean_per_complex_auprc"],
        "epitope_auprc": lambda item: item["metrics"]["epitope"]["auprc"],
        "paratope_auprc": lambda item: item["metrics"]["paratope"]["auprc"],
        "minimum_bidirectional_margin": lambda item: item["minimum_bidirectional_margin"],
    }
    output: dict[str, Any] = {}
    for index, (left_name, right_name) in enumerate(zip(conditions, conditions[1:])):
        left = sorted(by_condition[left_name], key=lambda item: item["seed"])
        right = sorted(by_condition[right_name], key=lambda item: item["seed"])
        if [item["seed"] for item in left] != [item["seed"] for item in right]:
            raise ValueError("Paired attribution seed axes differ")
        output[f"{left_name}__to__{right_name}"] = {
            endpoint: _paired_bootstrap(
                [selector(item) for item in left],
                [selector(item) for item in right],
                resamples=resamples,
                seed=20260816 + 100 * index + endpoint_index,
            )
            for endpoint_index, (endpoint, selector) in enumerate(endpoints.items())
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite an attribution aggregate")
    protocol, _ = _protocol(args.protocol, args.protocol_receipt)
    conditions = tuple(protocol.get("development_matrix", ()))
    seeds = tuple(protocol.get("model_selection", {}).get("seeds", ()))
    if len(conditions) != 3 or len(seeds) != 5:
        raise ValueError("Attribution matrix must contain three conditions and five seeds")
    protocol_sha = _sha256(args.protocol)
    receipt_sha = _sha256(args.protocol_receipt)
    by_condition = {
        condition: [
            _member(
                args.runs_root / condition / f"seed-{seed}",
                condition=condition,
                seed=seed,
                protocol_sha=protocol_sha,
                receipt_sha=receipt_sha,
            )
            for seed in seeds
        ]
        for condition in conditions
    }
    authorities = {
        json.dumps(
            {
                key: member["authority"].get(key)
                for key in (
                    "dataset_backend",
                    "asep_cache_manifest_sha256",
                    "training_partner_panel_sha256",
                    "validation_partner_panel_sha256",
                    "protocol_file_sha256",
                    "protocol_receipt_sha256",
                    "test_loaded",
                    "external_test_loaded",
                )
            },
            sort_keys=True,
        )
        for members in by_condition.values()
        for member in members
    }
    if len(authorities) != 1:
        raise ValueError("Run authorities differ across attribution matrix")
    payload = {
        "format_version": FORMAT,
        "protocol_file_sha256": protocol_sha,
        "protocol_receipt_sha256": receipt_sha,
        "conditions": {
            condition: _condition_summary(members)
            for condition, members in by_condition.items()
        },
        "paired_seed_comparisons": _comparisons(
            by_condition, conditions, resamples=args.bootstrap_resamples
        ),
        "authority": json.loads(next(iter(authorities))),
        "checkpoint_selection_rule": protocol["model_selection"]["checkpoint_rule"],
        "claim_boundary": "AsEP epitope-group development-validation pair-to-node attribution only; no checkpoint, architecture, or test-role selection is performed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(payload))
    receipt = {
        "format_version": "framecontact_asep_joint_attribution_dev_v1_aggregate_receipt",
        "aggregate_sha256": _sha256(args.output),
        "protocol_file_sha256": protocol_sha,
        "protocol_receipt_sha256": receipt_sha,
        "run_count": len(conditions) * len(seeds),
        "test_loaded": False,
        "frozen_test_loaded": False,
        "official_test_loaded": False,
        "external_test_loaded": False,
    }
    receipt_path = args.output.with_name(args.output.name + ".receipt.json")
    receipt_path.write_bytes(_canonical(receipt))
    print(json.dumps({"output": str(args.output), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
