#!/usr/bin/env python3
"""Aggregate the frozen six-condition AsEP joint-coupling v2 matrix.

The script authenticates every run receipt and the frozen development-only
protocol, reads already-selected validation checkpoints without reselecting
them, and reports five-seed summaries plus explicitly seed-paired bootstrap
intervals.  It never opens a frozen, official, or external test artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


FORMAT = "framecontact_asep_joint_coupling_dev_v2_aggregate_v1"
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
COMPARISONS = (
    ("framecontact_joint_a0_pair_only", "framecontact_joint_a1_noisy_or"),
    ("framecontact_joint_a1_noisy_or", "framecontact_joint_a2_independent_heads"),
    ("framecontact_joint_a2_independent_heads", "framecontact_joint_a3_coupled_heads"),
    ("framecontact_joint_a3_coupled_heads", "framecontact_joint_a4_coupled_soft_crop"),
    ("framecontact_joint_a4_coupled_soft_crop", "framecontact_joint_a5_graph_spatial"),
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}")
    return result


def _summary(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("Every joint-coupling condition requires exactly five seeds")
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_seed_bootstrap(
    left: list[float],
    right: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != 5 or len(right) != 5:
        raise ValueError("Paired seed bootstrap requires five aligned seeds")
    if isinstance(resamples, bool) or resamples < 1000:
        raise ValueError("Paired seed bootstrap requires at least 1000 resamples")
    deltas = [right[index] - left[index] for index in range(5)]
    generator = random.Random(seed)
    sampled = sorted(
        statistics.mean(deltas[generator.randrange(5)] for _ in range(5))
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
        "caution": "Only five seed pairs are available; this interval is not a per-complex bootstrap.",
    }


def _load_protocol(protocol_path: Path, receipt_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if any(path.is_symlink() or not path.is_file() for path in (protocol_path, receipt_path)):
        raise ValueError("Protocol and receipt must be regular files")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("protocol_file_sha256") != _sha256(protocol_path)
        or protocol.get("schema_version")
        != "framecontact_asep_joint_coupling_development_protocol_v2"
        or protocol.get("data", {}).get("test_role_forbidden") is not True
        or protocol.get("model_selection", {}).get("test_role_forbidden") is not True
        or receipt.get("official_test_used_for_selection") is not False
        or receipt.get("official_test_scores_read_for_this_protocol") is not False
    ):
        raise ValueError("Joint-coupling v2 protocol authority is ineligible")
    return protocol, receipt


def _history(result: dict[str, Any], path: Path) -> dict[int, dict[str, Any]]:
    rows = result.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Run lacks validation history: {path}")
    history: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("epoch"), bool):
            raise ValueError(f"Malformed validation history: {path}")
        epoch = int(row.get("epoch", 0))
        validation = row.get("validation")
        if epoch < 1 or epoch in history or not isinstance(validation, dict):
            raise ValueError(f"Malformed or duplicate validation epoch: {path}")
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
        margins = [
            _number(
                directions[direction].get(
                    "cluster_weighted_cognate_compatibility_margin"
                ),
                f"{direction}.margin",
            )
            for direction in RETRIEVAL_DIRECTIONS
        ]
        history[epoch] = {
            "metrics": metrics,
            "minimum_bidirectional_margin": min(margins),
        }
    return history


def _member(
    run_root: Path,
    *,
    condition: str,
    seed: int,
    protocol_sha256: str,
    protocol_receipt_sha256: str,
) -> dict[str, Any]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = run_root / "RUN_RECEIPT.json"
    if any(path.is_symlink() or not path.is_file() for path in (result_path, receipt_path)):
        raise ValueError(f"Incomplete development run: {run_root}")
    run_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
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
        or authority.get("protocol_file_sha256") != protocol_sha256
        or authority.get("protocol_receipt_sha256") != protocol_receipt_sha256
        or result.get("experiment") != condition
        or result.get("configuration", {}).get("seed") != seed
    ):
        raise ValueError(f"Run falls outside the frozen development authority: {run_root}")
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
    directions = selected.get("partner_retrieval_by_direction")
    if not isinstance(directions, dict) or set(directions) != set(RETRIEVAL_DIRECTIONS):
        raise ValueError(f"Selected retrieval directions differ: {run_root}")
    retrieval = {
        direction: {
            field: _number(directions[direction].get(field), f"{direction}.{field}")
            for field in RETRIEVAL_FIELDS
        }
        for direction in RETRIEVAL_DIRECTIONS
    }
    history = _history(result, run_root)
    selected_epoch = int(result.get("selection", {}).get("best_epoch", 0))
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
        "history": history,
        "authority": authority,
    }


def _condition_summary(members: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(members, key=lambda item: item["seed"])
    return {
        "seed_count": 5,
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
            / 5,
        },
        "members": [
            {
                key: item[key]
                for key in (
                    "seed",
                    "selected_epoch",
                    "result_sha256",
                    "selected_checkpoint_sha256",
                )
            }
            for item in members
        ],
    }


def _paired_comparisons(
    by_condition: dict[str, list[dict[str, Any]]], *, resamples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    endpoints = (
        ("contact_auprc", lambda item: item["metrics"]["contact"]["auprc"]),
        ("contact_mean_per_complex_auprc", lambda item: item["metrics"]["contact"]["mean_per_complex_auprc"]),
        ("epitope_auprc", lambda item: item["metrics"]["epitope"]["auprc"]),
        ("paratope_auprc", lambda item: item["metrics"]["paratope"]["auprc"]),
        ("minimum_bidirectional_margin", lambda item: item["minimum_bidirectional_margin"]),
    )
    for comparison_index, (left_name, right_name) in enumerate(COMPARISONS):
        left = sorted(by_condition[left_name], key=lambda item: item["seed"])
        right = sorted(by_condition[right_name], key=lambda item: item["seed"])
        if [item["seed"] for item in left] != [item["seed"] for item in right]:
            raise ValueError("Paired comparison seed axes differ")
        output[f"{left_name}__to__{right_name}"] = {
            endpoint: _paired_seed_bootstrap(
                [selector(item) for item in left],
                [selector(item) for item in right],
                resamples=resamples,
                seed=20260815 + 100 * comparison_index + endpoint_index,
            )
            for endpoint_index, (endpoint, selector) in enumerate(endpoints)
        }
    return output


def _matched_trajectory(by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    epoch_sets = [
        set(member["history"])
        for members in by_condition.values()
        for member in members
    ]
    matched_epochs = sorted(set.intersection(*epoch_sets))
    if not matched_epochs:
        raise ValueError("Conditions have no common validation epoch")
    trajectory: dict[str, Any] = {}
    for epoch in matched_epochs:
        trajectory[str(epoch)] = {
            condition: {
                "contact_auprc": _summary(
                    [member["history"][epoch]["metrics"]["contact"]["auprc"] for member in members]
                ),
                "epitope_auprc": _summary(
                    [member["history"][epoch]["metrics"]["epitope"]["auprc"] for member in members]
                ),
                "paratope_auprc": _summary(
                    [member["history"][epoch]["metrics"]["paratope"]["auprc"] for member in members]
                ),
                "minimum_bidirectional_margin": _summary(
                    [member["history"][epoch]["minimum_bidirectional_margin"] for member in members]
                ),
            }
            for condition, members in by_condition.items()
        }
    return {"matched_epochs": matched_epochs, "by_epoch": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=50000)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError("Refusing to overwrite a joint-coupling aggregate")
    protocol, receipt = _load_protocol(arguments.protocol, arguments.protocol_receipt)
    conditions = tuple(protocol.get("development_matrix", ()))
    seeds = tuple(protocol.get("model_selection", {}).get("seeds", ()))
    if conditions != tuple(condition for comparison in COMPARISONS for condition in comparison[:1]) + (COMPARISONS[-1][1],):
        raise ValueError("Protocol condition order differs from the frozen v2 matrix")
    if seeds != (17, 42, 73, 101, 137):
        raise ValueError("Protocol seed axis differs from the frozen v2 matrix")
    protocol_sha = _sha256(arguments.protocol)
    protocol_receipt_sha = _sha256(arguments.protocol_receipt)
    by_condition = {
        condition: [
            _member(
                arguments.runs_root / condition / f"seed-{seed}",
                condition=condition,
                seed=seed,
                protocol_sha256=protocol_sha,
                protocol_receipt_sha256=protocol_receipt_sha,
            )
            for seed in seeds
        ]
        for condition in conditions
    }
    authority_fingerprints = {
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
    if len(authority_fingerprints) != 1:
        raise ValueError("Run authorities differ across the frozen matrix")
    payload = {
        "format_version": FORMAT,
        "protocol_file_sha256": protocol_sha,
        "protocol_receipt_sha256": protocol_receipt_sha,
        "conditions": {
            condition: _condition_summary(members)
            for condition, members in by_condition.items()
        },
        "paired_seed_comparisons": _paired_comparisons(
            by_condition, resamples=arguments.bootstrap_resamples
        ),
        "matched_validation_trajectory": _matched_trajectory(by_condition),
        "authority": json.loads(next(iter(authority_fingerprints))),
        "checkpoint_selection_rule": protocol["model_selection"]["checkpoint_rule"],
        "claim_boundary": "AsEP epitope_group development-validation aggregation only. This script does not select checkpoints or architectures and reads no frozen, official, or external test result.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical(payload))
    aggregate_receipt = {
        "format_version": "framecontact_asep_joint_coupling_dev_v2_aggregate_receipt_v1",
        "aggregate_sha256": _sha256(arguments.output),
        "protocol_file_sha256": protocol_sha,
        "protocol_receipt_sha256": protocol_receipt_sha,
        "run_count": 30,
        "test_loaded": False,
        "frozen_test_loaded": False,
        "official_test_loaded": False,
        "external_test_loaded": False,
    }
    receipt_path = arguments.output.with_name(arguments.output.name + ".receipt.json")
    receipt_path.write_bytes(_canonical(aggregate_receipt))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "aggregate_sha256": aggregate_receipt["aggregate_sha256"],
                "run_count": 30,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
