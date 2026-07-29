#!/usr/bin/env python3
"""Independently verify a SAbDab2 geometry checkpoint-audit archive."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


FORMAT = "sabdab2_geometry_independent_verification_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--expected-audit-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _authenticate(path: Path, expected: Any, name: str) -> str:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{name} expected SHA-256 is invalid")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return observed


def _validated_side(raw: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(raw, Mapping) or set(raw) != {"labels", "probabilities"}:
        raise ValueError("Prediction sides contain unexpected fields")
    labels = np.asarray(raw["labels"], dtype=np.uint8)
    probabilities = np.asarray(raw["probabilities"], dtype=np.float64)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or labels.shape != probabilities.shape
        or labels.size == 0
        or not np.isin(labels, (0, 1)).all()
        or not np.isfinite(probabilities).all()
        or np.logical_or(probabilities < 0.0, probabilities > 1.0).any()
        or int(labels.sum()) in {0, int(labels.size)}
    ):
        raise ValueError("Prediction sides are not valid binary probability streams")
    return labels, probabilities


def _components(instances: list[Mapping[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    endpoint_by_instance: dict[str, tuple[str, str]] = {}
    for instance in instances:
        identifier = str(instance["instance_id"])
        if identifier in endpoint_by_instance or identifier != identifier.strip():
            raise ValueError("Prediction instances are duplicated or noncanonical")
        antibody = f"ab:{instance['antibody_sequence_sha256']}"
        antigen = f"ag:{instance['antigen_sequence_sha256']}"
        endpoint_by_instance[identifier] = (antibody, antigen)
        union(antibody, antigen)
    members: dict[str, list[str]] = defaultdict(list)
    for identifier, (antibody, _) in endpoint_by_instance.items():
        members[find(antibody)].append(identifier)
    digest_by_root = {
        root: hashlib.sha256(_canonical_json_bytes(sorted(identifiers))).hexdigest()
        for root, identifiers in members.items()
    }
    return {
        identifier: digest_by_root[find(antibody)]
        for identifier, (antibody, _) in endpoint_by_instance.items()
    }


def _bootstrap(
    differences: Mapping[str, float],
    components: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for identifier, value in differences.items():
        grouped[components[identifier]].append(float(value))
    arrays = [
        np.asarray(grouped[component], dtype=np.float64)
        for component in sorted(grouped)
    ]
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, len(arrays), size=len(arrays))
        estimates[iteration] = np.concatenate(
            [arrays[index] for index in sampled]
        ).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(list(differences.values()))),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
    }


def _sign_flip(
    differences: Mapping[str, float],
    components: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for identifier, value in differences.items():
        grouped[components[identifier]].append(float(value))
    arrays = [
        np.asarray(grouped[component], dtype=np.float64)
        for component in sorted(grouped)
    ]
    observed = float(np.mean(list(differences.values())))
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    null = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(arrays))
        null[iteration] = np.concatenate(
            [values * signs[index] for index, values in enumerate(arrays)]
        ).mean()
    return {
        "one_sided_p_value": float(
            (1 + int((null >= observed).sum())) / (iterations + 1)
        ),
        "two_sided_p_value": float(
            (1 + int((np.abs(null) >= abs(observed)).sum())) / (iterations + 1)
        ),
    }


def _close(observed: float, expected: float, tolerance: float, name: str) -> None:
    if (
        not math.isfinite(observed)
        or not math.isfinite(expected)
        or abs(observed - expected) > tolerance
    ):
        raise ValueError(
            f"{name} differs: observed {observed}, expected {expected}, "
            f"tolerance {tolerance}"
        )


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("output already exists")
    if not math.isfinite(arguments.tolerance) or arguments.tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    receipt_path = arguments.audit_root / "AUDIT_RECEIPT.json"
    predictions_path = arguments.audit_root / "PER_COMPLEX_PREDICTIONS.json"
    statistics_path = arguments.audit_root / "GEOMETRY_ABLATION_STATISTICS.json"
    receipt_sha256 = _authenticate(
        receipt_path,
        arguments.expected_audit_receipt_sha256,
        "audit receipt",
    )
    receipt = _load_json(receipt_path, "audit receipt")
    predictions_sha256 = _authenticate(
        predictions_path,
        receipt.get("predictions_sha256"),
        "predictions",
    )
    statistics_sha256 = _authenticate(
        statistics_path,
        receipt.get("statistics_sha256"),
        "statistics",
    )
    predictions = _load_json(predictions_path, "predictions")
    statistics = _load_json(statistics_path, "statistics")
    if receipt.get("status") != "passed":
        raise ValueError("Checkpoint audit did not pass")
    if predictions.get("format_version") != "sabdab2_geometry_predictions_v1":
        raise ValueError("Unexpected predictions format")
    if statistics.get("format_version") != "sabdab2_geometry_statistics_v1":
        raise ValueError("Unexpected statistics format")
    if statistics.get("prediction_payload_sha256") != hashlib.sha256(
        _canonical_json_bytes(predictions)
    ).hexdigest():
        raise ValueError("Statistics do not authenticate predictions")

    modes: dict[str, list[Mapping[str, Any]]] = {}
    labels_by_instance: dict[str, dict[str, np.ndarray]] = {}
    endpoints_by_instance: dict[str, tuple[str, str]] = {}
    metrics_by_mode: dict[str, dict[str, dict[str, float]]] = {}
    for raw_mode in predictions["modes"]:
        raw_name = str(raw_mode["mode"])
        mode = "topology_only" if raw_name == "none" else raw_name
        if mode in modes:
            raise ValueError("Prediction modes are duplicated after normalization")
        instances = raw_mode["instances"]
        if not isinstance(instances, list) or not instances:
            raise ValueError("Prediction modes require nonempty instance lists")
        modes[mode] = instances
        metrics_by_mode[mode] = {}
        for instance in instances:
            identifier = str(instance["instance_id"])
            if identifier in metrics_by_mode[mode]:
                raise ValueError("Prediction instances are duplicated")
            paratope_labels, paratope_probabilities = _validated_side(
                instance["paratope"]
            )
            epitope_labels, epitope_probabilities = _validated_side(
                instance["epitope"]
            )
            if identifier not in labels_by_instance:
                labels_by_instance[identifier] = {
                    "paratope": paratope_labels,
                    "epitope": epitope_labels,
                }
                endpoints_by_instance[identifier] = (
                    str(instance["antibody_sequence_sha256"]),
                    str(instance["antigen_sequence_sha256"]),
                )
            else:
                for side, labels in (
                    ("paratope", paratope_labels),
                    ("epitope", epitope_labels),
                ):
                    if not np.array_equal(labels_by_instance[identifier][side], labels):
                        raise ValueError("Labels differ across prediction modes")
                if endpoints_by_instance[identifier] != (
                    str(instance["antibody_sequence_sha256"]),
                    str(instance["antigen_sequence_sha256"]),
                ):
                    raise ValueError("Endpoints differ across prediction modes")
            paratope_auprc = float(
                average_precision_score(paratope_labels, paratope_probabilities)
            )
            epitope_auprc = float(
                average_precision_score(epitope_labels, epitope_probabilities)
            )
            metrics_by_mode[mode][identifier] = {
                "paratope_auprc": paratope_auprc,
                "epitope_auprc": epitope_auprc,
                "joint_macro_auprc": 0.5 * (paratope_auprc + epitope_auprc),
                "paratope_auroc": float(
                    roc_auc_score(paratope_labels, paratope_probabilities)
                ),
                "epitope_auroc": float(
                    roc_auc_score(epitope_labels, epitope_probabilities)
                ),
            }

    reference_ids = [str(item["instance_id"]) for item in modes["topology_only"]]
    for mode, instances in modes.items():
        if [str(item["instance_id"]) for item in instances] != reference_ids:
            raise ValueError(f"Mode {mode} changes the ordered validation universe")
    components = _components(modes["topology_only"])
    if components != statistics.get("component_by_instance"):
        raise ValueError("Independent endpoint components differ")

    checks = 0
    for mode, by_instance in metrics_by_mode.items():
        expected = statistics["modes"][mode]
        for identifier, metrics in by_instance.items():
            stored = expected["per_instance"][identifier]
            for name in ("paratope_auprc", "epitope_auprc"):
                side = name.removesuffix("_auprc")
                _close(
                    metrics[name],
                    float(stored[side]["auprc"]),
                    arguments.tolerance,
                    f"{mode}/{identifier}/{name}",
                )
                checks += 1
            _close(
                metrics["joint_macro_auprc"],
                float(stored["joint_macro_auprc"]),
                arguments.tolerance,
                f"{mode}/{identifier}/joint_macro_auprc",
            )
            checks += 1
        observed_macro = float(
            np.mean(
                [value["joint_macro_auprc"] for value in by_instance.values()]
            )
        )
        _close(
            observed_macro,
            float(expected["complex_macro"]["joint_macro_auprc"]),
            arguments.tolerance,
            f"{mode}/complex_macro",
        )
        checks += 1

    resampling = statistics["resampling"]
    candidate = metrics_by_mode["distance_direction"]
    reference = metrics_by_mode["topology_only"]
    differences = {
        identifier: candidate[identifier]["joint_macro_auprc"]
        - reference[identifier]["joint_macro_auprc"]
        for identifier in reference_ids
    }
    bootstrap = _bootstrap(
        differences,
        components,
        iterations=int(resampling["bootstrap_iterations"]),
        seed=int(resampling["bootstrap_seed"]),
        confidence=float(resampling["confidence"]),
    )
    sign_flip = _sign_flip(
        differences,
        components,
        iterations=int(resampling["permutation_iterations"]),
        seed=int(resampling["permutation_seed"]),
    )
    stored_primary = statistics["comparisons"][
        "distance_direction_minus_topology_only"
    ]["joint_macro_auprc"]
    for name in ("mean", "lower", "upper"):
        stored_name = "mean_difference" if name == "mean" else name
        _close(
            bootstrap[name],
            float(stored_primary["paired_component_bootstrap"][stored_name]),
            arguments.tolerance,
            f"primary_bootstrap/{name}",
        )
        checks += 1
    for name in ("one_sided_p_value", "two_sided_p_value"):
        _close(
            sign_flip[name],
            float(stored_primary["paired_component_sign_flip"][name]),
            arguments.tolerance,
            f"primary_sign_flip/{name}",
        )
        checks += 1

    if statistics["primary_gate"]["confirmatory_superiority_authorized"] is not False:
        raise ValueError("Single-seed audit improperly authorizes superiority")
    if receipt.get("confirmatory_superiority_authorized") is not False:
        raise ValueError("Audit receipt improperly authorizes superiority")

    result = {
        "format_version": FORMAT,
        "status": "passed",
        "audit_receipt_sha256": receipt_sha256,
        "predictions_sha256": predictions_sha256,
        "statistics_sha256": statistics_sha256,
        "independent_check_count": checks,
        "validation_instance_count": len(reference_ids),
        "independent_endpoint_component_count": len(set(components.values())),
        "primary_mean_difference": bootstrap["mean"],
        "primary_bootstrap_ci95": [bootstrap["lower"], bootstrap["upper"]],
        "primary_one_sided_permutation_p_value": sign_flip[
            "one_sided_p_value"
        ],
        "confirmatory_superiority_authorized": False,
        "claim_boundary": (
            "Independent verification of development-only marginal-site "
            "statistics. It is not partner-specific, frozen-test, or SOTA evidence."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_json_bytes(result))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Independent verification failed: {error}", file=sys.stderr)
        raise

