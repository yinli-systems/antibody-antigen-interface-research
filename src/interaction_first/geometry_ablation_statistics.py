"""Paired complex-level statistics for SAbDab2 geometry ablations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from .metrics import (
    binary_metrics,
    calibration_metrics,
    paired_cluster_bootstrap_mean_difference,
)


PREDICTION_FORMAT = "sabdab2_geometry_predictions_v1"
STATISTICS_FORMAT = "sabdab2_geometry_statistics_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODE_ALIASES = {"none": "topology_only"}
_PRIMARY_MODE = "distance_direction"
_REFERENCE_MODE = "topology_only"
_SUPERIORITY_MODES = ("distance", "distance_direction", "full_pose")
_COMPARISONS = (
    ("distance", "topology_only"),
    ("distance_direction", "topology_only"),
    ("full_pose", "topology_only"),
    ("full_pose", "distance_direction"),
)


class GeometryStatisticsError(ValueError):
    """Raised when a geometry-ablation statistics input fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a receipt payload without platform-dependent whitespace."""
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


def _mode_name(value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise GeometryStatisticsError("Mode names must be canonical nonempty strings")
    return _MODE_ALIASES.get(value, value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GeometryStatisticsError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GeometryStatisticsError(f"{name} must be a canonical nonempty string")
    return value


def _validated_side(side: Any, name: str) -> tuple[Any, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for geometry statistics") from error
    if not isinstance(side, Mapping):
        raise GeometryStatisticsError(f"{name} must be an object")
    if set(side) != {"labels", "probabilities"}:
        raise GeometryStatisticsError(
            f"{name} must contain only labels and probabilities"
        )
    labels = np.asarray(side["labels"], dtype=np.uint8)
    probabilities = np.asarray(side["probabilities"], dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise GeometryStatisticsError(f"{name} arrays must be aligned one-dimensional data")
    if labels.size == 0:
        raise GeometryStatisticsError(f"{name} arrays must be nonempty")
    if not np.isin(labels, (0, 1)).all():
        raise GeometryStatisticsError(f"{name} labels must be binary")
    if not np.isfinite(probabilities).all() or np.logical_or(
        probabilities < 0.0, probabilities > 1.0
    ).any():
        raise GeometryStatisticsError(f"{name} probabilities must be finite in [0, 1]")
    if int(labels.sum()) == 0 or int(labels.sum()) == int(labels.size):
        raise GeometryStatisticsError(
            f"{name} must contain both classes for per-complex ranking metrics"
        )
    return labels, probabilities


def _validated_instance(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise GeometryStatisticsError("Prediction instances must be objects")
    expected = {
        "instance_id",
        "antibody_sequence_sha256",
        "antigen_sequence_sha256",
        "paratope",
        "epitope",
    }
    if set(raw) != expected:
        raise GeometryStatisticsError(
            "Prediction instances contain missing or unexpected fields"
        )
    paratope_labels, paratope_probabilities = _validated_side(
        raw["paratope"], "paratope"
    )
    epitope_labels, epitope_probabilities = _validated_side(
        raw["epitope"], "epitope"
    )
    return {
        "instance_id": _identifier(raw["instance_id"], "instance_id"),
        "antibody_sequence_sha256": _sha256(
            raw["antibody_sequence_sha256"], "antibody_sequence_sha256"
        ),
        "antigen_sequence_sha256": _sha256(
            raw["antigen_sequence_sha256"], "antigen_sequence_sha256"
        ),
        "paratope_labels": paratope_labels,
        "paratope_probabilities": paratope_probabilities,
        "epitope_labels": epitope_labels,
        "epitope_probabilities": epitope_probabilities,
    }


def endpoint_connected_components(
    instances: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Return exact endpoint-connected components for paired resampling."""
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
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    endpoints: dict[str, tuple[str, str]] = {}
    for raw in instances:
        instance = _validated_instance(raw)
        identifier = instance["instance_id"]
        if identifier in endpoints:
            raise GeometryStatisticsError(f"Duplicate prediction instance: {identifier}")
        antibody = f"ab:{instance['antibody_sequence_sha256']}"
        antigen = f"ag:{instance['antigen_sequence_sha256']}"
        endpoints[identifier] = (antibody, antigen)
        union(antibody, antigen)
    if not endpoints:
        raise GeometryStatisticsError("At least one prediction instance is required")

    members_by_root: dict[str, list[str]] = {}
    for identifier, (antibody, _) in endpoints.items():
        members_by_root.setdefault(find(antibody), []).append(identifier)
    component_digest_by_root = {
        root: hashlib.sha256(canonical_json_bytes(sorted(members))).hexdigest()
        for root, members in members_by_root.items()
    }
    return {
        identifier: component_digest_by_root[find(antibody)]
        for identifier, (antibody, _) in endpoints.items()
    }


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((float(value), name) for name, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (value, name) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _component_sign_flip(
    differences: Mapping[str, float],
    components: Mapping[str, str],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for geometry statistics") from error
    if set(differences) != set(components) or not differences:
        raise GeometryStatisticsError("Permutation inputs must be nonempty and aligned")
    grouped: dict[str, list[float]] = {}
    for identifier, raw_value in differences.items():
        value = float(raw_value)
        if not math.isfinite(value):
            raise GeometryStatisticsError("Permutation differences must be finite")
        grouped.setdefault(components[identifier], []).append(value)
    component_values = [
        np.asarray(grouped[name], dtype=np.float64) for name in sorted(grouped)
    ]
    observed = float(np.mean(list(differences.values())))
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    null = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(component_values))
        signed = [
            values * signs[index] for index, values in enumerate(component_values)
        ]
        null[iteration] = float(np.concatenate(signed).mean())
    one_sided = (1 + int((null >= observed).sum())) / (iterations + 1)
    two_sided = (1 + int((np.abs(null) >= abs(observed)).sum())) / (
        iterations + 1
    )
    return {
        "observed_mean_difference": observed,
        "independent_component_count": len(component_values),
        "iterations": iterations,
        "seed": seed,
        "generator": "PCG64DXSM",
        "one_sided_p_value": float(one_sided),
        "two_sided_p_value": float(two_sided),
        "plus_one_correction": True,
    }


def _instance_metrics(instance: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for side in ("paratope", "epitope"):
        labels = instance[f"{side}_labels"]
        probabilities = instance[f"{side}_probabilities"]
        metrics = binary_metrics(labels, probabilities, threshold=0.5)
        if metrics["auprc"] is None or metrics["auroc"] is None:
            raise GeometryStatisticsError(
                f"{instance['instance_id']} {side} lacks both classes"
            )
        metrics["calibration"] = calibration_metrics(
            labels, probabilities, bins=15, strategy="adaptive"
        )
        result[side] = metrics
    result["joint_macro_auprc"] = 0.5 * (
        float(result["paratope"]["auprc"]) + float(result["epitope"]["auprc"])
    )
    result["joint_macro_auroc"] = 0.5 * (
        float(result["paratope"]["auroc"]) + float(result["epitope"]["auroc"])
    )
    return result


def analyze_geometry_predictions(
    payload: Mapping[str, Any],
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20_260_729,
    permutation_iterations: int = 10_000,
    permutation_seed: int = 20_260_730,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Analyze immutable per-complex predictions without selecting a model."""
    if payload.get("format_version") != PREDICTION_FORMAT:
        raise GeometryStatisticsError("Unexpected prediction format")
    raw_modes = payload.get("modes")
    if not isinstance(raw_modes, Sequence) or isinstance(raw_modes, (str, bytes)):
        raise GeometryStatisticsError("Prediction modes must be a sequence")

    mode_instances: dict[str, list[dict[str, Any]]] = {}
    raw_name_by_mode: dict[str, str] = {}
    for raw_mode in raw_modes:
        if not isinstance(raw_mode, Mapping) or set(raw_mode) != {"mode", "instances"}:
            raise GeometryStatisticsError("Mode records contain unexpected fields")
        raw_name = _identifier(raw_mode["mode"], "mode")
        mode = _mode_name(raw_name)
        if mode in mode_instances:
            raise GeometryStatisticsError(f"Duplicate normalized mode: {mode}")
        raw_instances = raw_mode["instances"]
        if not isinstance(raw_instances, Sequence) or isinstance(
            raw_instances, (str, bytes)
        ):
            raise GeometryStatisticsError("Mode instances must be a sequence")
        instances = [_validated_instance(item) for item in raw_instances]
        identifiers = [item["instance_id"] for item in instances]
        if len(set(identifiers)) != len(identifiers):
            raise GeometryStatisticsError(f"Mode {mode} contains duplicate instances")
        mode_instances[mode] = instances
        raw_name_by_mode[mode] = raw_name

    required = {_REFERENCE_MODE, _PRIMARY_MODE}
    if not required.issubset(mode_instances):
        raise GeometryStatisticsError(
            f"Prediction payload is missing required modes: {sorted(required - set(mode_instances))}"
        )

    reference = mode_instances[_REFERENCE_MODE]
    reference_ids = [item["instance_id"] for item in reference]
    reference_by_id = {item["instance_id"]: item for item in reference}
    for mode, instances in mode_instances.items():
        identifiers = [item["instance_id"] for item in instances]
        if identifiers != reference_ids:
            raise GeometryStatisticsError(
                f"Mode {mode} does not use the identical ordered validation universe"
            )
        for instance in instances:
            reference_instance = reference_by_id[instance["instance_id"]]
            for field in (
                "antibody_sequence_sha256",
                "antigen_sequence_sha256",
            ):
                if instance[field] != reference_instance[field]:
                    raise GeometryStatisticsError(
                        f"Mode {mode} changes endpoint identity for {instance['instance_id']}"
                    )
            for field in ("paratope_labels", "epitope_labels"):
                if not (instance[field] == reference_instance[field]).all():
                    raise GeometryStatisticsError(
                        f"Mode {mode} changes labels for {instance['instance_id']}"
                    )

    serializable_reference = [
        {
            "instance_id": item["instance_id"],
            "antibody_sequence_sha256": item["antibody_sequence_sha256"],
            "antigen_sequence_sha256": item["antigen_sequence_sha256"],
            "paratope": {
                "labels": item["paratope_labels"].tolist(),
                "probabilities": item["paratope_probabilities"].tolist(),
            },
            "epitope": {
                "labels": item["epitope_labels"].tolist(),
                "probabilities": item["epitope_probabilities"].tolist(),
            },
        }
        for item in reference
    ]
    components = endpoint_connected_components(serializable_reference)

    per_mode: dict[str, Any] = {}
    values: dict[str, dict[str, dict[str, float]]] = {}
    for mode, instances in mode_instances.items():
        per_instance: dict[str, Any] = {}
        values[mode] = {
            "joint_macro_auprc": {},
            "paratope_auprc": {},
            "epitope_auprc": {},
        }
        pooled_labels: dict[str, list[Any]] = {"paratope": [], "epitope": []}
        pooled_probabilities: dict[str, list[Any]] = {"paratope": [], "epitope": []}
        for instance in instances:
            metrics = _instance_metrics(instance)
            identifier = instance["instance_id"]
            per_instance[identifier] = metrics
            values[mode]["joint_macro_auprc"][identifier] = metrics[
                "joint_macro_auprc"
            ]
            values[mode]["paratope_auprc"][identifier] = float(
                metrics["paratope"]["auprc"]
            )
            values[mode]["epitope_auprc"][identifier] = float(
                metrics["epitope"]["auprc"]
            )
            for side in ("paratope", "epitope"):
                pooled_labels[side].append(instance[f"{side}_labels"])
                pooled_probabilities[side].append(instance[f"{side}_probabilities"])
        try:
            import numpy as np
        except ModuleNotFoundError as error:
            raise RuntimeError("NumPy is required for geometry statistics") from error
        pooled = {
            side: binary_metrics(
                np.concatenate(pooled_labels[side]),
                np.concatenate(pooled_probabilities[side]),
                threshold=0.5,
            )
            for side in ("paratope", "epitope")
        }
        pooled["macro_auprc"] = 0.5 * (
            float(pooled["paratope"]["auprc"])
            + float(pooled["epitope"]["auprc"])
        )
        per_mode[mode] = {
            "source_mode_name": raw_name_by_mode[mode],
            "per_instance": per_instance,
            "complex_macro": {
                metric: float(sum(by_id.values()) / len(by_id))
                for metric, by_id in values[mode].items()
            },
            "pooled_residue_descriptive": pooled,
        }

    comparisons: dict[str, Any] = {}
    for candidate, reference_mode in _COMPARISONS:
        if candidate not in values or reference_mode not in values:
            continue
        comparison_name = f"{candidate}_minus_{reference_mode}"
        comparison: dict[str, Any] = {}
        for metric in (
            "joint_macro_auprc",
            "paratope_auprc",
            "epitope_auprc",
        ):
            candidate_values = values[candidate][metric]
            reference_values = values[reference_mode][metric]
            differences = {
                identifier: candidate_values[identifier] - reference_values[identifier]
                for identifier in reference_ids
            }
            bootstrap = paired_cluster_bootstrap_mean_difference(
                candidate_values,
                reference_values,
                components,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
                confidence=confidence,
                insufficient_cluster_policy="mark_non_authorizing",
            )
            permutation = _component_sign_flip(
                differences,
                components,
                iterations=permutation_iterations,
                seed=permutation_seed,
            )
            comparison[metric] = {
                "paired_component_bootstrap": bootstrap,
                "paired_component_sign_flip": permutation,
                "positive_complex_fraction": float(
                    sum(value > 0.0 for value in differences.values())
                    / len(differences)
                ),
                "zero_complex_fraction": float(
                    sum(value == 0.0 for value in differences.values())
                    / len(differences)
                ),
            }
        comparisons[comparison_name] = comparison

    raw_primary_p_values = {
        mode: comparisons[f"{mode}_minus_{_REFERENCE_MODE}"][
            "joint_macro_auprc"
        ]["paired_component_sign_flip"]["one_sided_p_value"]
        for mode in _SUPERIORITY_MODES
        if f"{mode}_minus_{_REFERENCE_MODE}" in comparisons
    }
    holm = _holm_adjust(raw_primary_p_values)
    for mode, adjusted in holm.items():
        comparisons[f"{mode}_minus_{_REFERENCE_MODE}"]["joint_macro_auprc"][
            "holm_adjusted_one_sided_p_value"
        ] = adjusted

    primary = comparisons[f"{_PRIMARY_MODE}_minus_{_REFERENCE_MODE}"]
    primary_joint = primary["joint_macro_auprc"]
    primary_paratope = primary["paratope_auprc"]
    primary_epitope = primary["epitope_auprc"]
    bootstrap = primary_joint["paired_component_bootstrap"]
    statistical_conditions = {
        "mean_difference_positive": float(bootstrap["mean_difference"]) > 0.0,
        "bootstrap_lower_bound_above_zero": (
            bool(bootstrap["inferential_authorized"])
            and float(bootstrap["lower"]) > 0.0
        ),
        "holm_adjusted_one_sided_p_below_0_05": (
            float(primary_joint["holm_adjusted_one_sided_p_value"]) < 0.05
        ),
        "paratope_direction_nonnegative": (
            float(
                primary_paratope["paired_component_bootstrap"]["mean_difference"]
            )
            >= 0.0
        ),
        "epitope_direction_nonnegative": (
            float(primary_epitope["paired_component_bootstrap"]["mean_difference"])
            >= 0.0
        ),
    }
    statistical_pass = all(statistical_conditions.values())
    gate = {
        "primary_comparison": f"{_PRIMARY_MODE}_minus_{_REFERENCE_MODE}",
        "statistical_conditions": statistical_conditions,
        "single_seed_statistical_gate_passed": statistical_pass,
        "multiseed_requirement_satisfied": False,
        "confirmatory_superiority_authorized": False,
        "status": (
            "pending_preregistered_multiseed_extension"
            if statistical_pass
            else "failed_single_seed_statistical_gate"
        ),
        "claim_boundary": (
            "This development analysis evaluates intra-chain marginal site "
            "prediction only. It cannot establish partner specificity or SOTA."
        ),
    }

    return {
        "format_version": STATISTICS_FORMAT,
        "prediction_payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest(),
        "validation_instance_count": len(reference_ids),
        "independent_endpoint_component_count": len(set(components.values())),
        "component_by_instance": components,
        "resampling": {
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_generator": "NumPy default_rng for existing validated bootstrap",
            "permutation_iterations": permutation_iterations,
            "permutation_seed": permutation_seed,
            "permutation_generator": "PCG64DXSM",
            "confidence": confidence,
            "unit": "exact antibody-antigen endpoint-connected component",
        },
        "modes": per_mode,
        "comparisons": comparisons,
        "primary_gate": gate,
    }

