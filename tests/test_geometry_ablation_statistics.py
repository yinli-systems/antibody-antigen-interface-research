"""Tests for paired complex-level geometry-ablation statistics."""

from __future__ import annotations

import hashlib

import pytest

from interaction_first.geometry_ablation_statistics import (
    GeometryStatisticsError,
    analyze_geometry_predictions,
    endpoint_connected_components,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _instance(index: int, quality: str) -> dict:
    labels = [1, 0, 1, 0]
    probabilities = {
        "reference": [0.1, 0.9, 0.2, 0.8],
        "distance": [0.6, 0.4, 0.55, 0.45],
        "direction": [0.9, 0.1, 0.8, 0.2],
        "pose": [0.8, 0.2, 0.7, 0.3],
    }[quality]
    return {
        "instance_id": f"complex_{index:02d}",
        "antibody_sequence_sha256": _digest(f"antibody_{index}"),
        "antigen_sequence_sha256": _digest(f"antigen_{index}"),
        "paratope": {
            "labels": labels,
            "probabilities": probabilities,
        },
        "epitope": {
            "labels": labels,
            "probabilities": probabilities,
        },
    }


def _payload(count: int = 12) -> dict:
    return {
        "format_version": "sabdab2_geometry_predictions_v1",
        "modes": [
            {
                "mode": "none",
                "instances": [_instance(index, "reference") for index in range(count)],
            },
            {
                "mode": "distance",
                "instances": [_instance(index, "distance") for index in range(count)],
            },
            {
                "mode": "distance_direction",
                "instances": [_instance(index, "direction") for index in range(count)],
            },
            {
                "mode": "full_pose",
                "instances": [_instance(index, "pose") for index in range(count)],
            },
        ],
    }


def test_endpoint_components_include_transitive_shared_endpoints() -> None:
    instances = [_instance(index, "reference") for index in range(3)]
    instances[1]["antibody_sequence_sha256"] = instances[0][
        "antibody_sequence_sha256"
    ]
    instances[2]["antigen_sequence_sha256"] = instances[1][
        "antigen_sequence_sha256"
    ]
    components = endpoint_connected_components(instances)
    assert len(set(components.values())) == 1


def test_analysis_is_deterministic_and_normalizes_legacy_none() -> None:
    payload = _payload()
    arguments = {
        "bootstrap_iterations": 200,
        "permutation_iterations": 200,
        "bootstrap_seed": 7,
        "permutation_seed": 11,
    }
    first = analyze_geometry_predictions(payload, **arguments)
    second = analyze_geometry_predictions(payload, **arguments)
    assert first == second
    assert "topology_only" in first["modes"]
    assert first["modes"]["topology_only"]["source_mode_name"] == "none"
    primary = first["comparisons"][
        "distance_direction_minus_topology_only"
    ]["joint_macro_auprc"]
    assert primary["paired_component_bootstrap"]["mean_difference"] > 0.0
    assert first["primary_gate"]["confirmatory_superiority_authorized"] is False
    assert first["primary_gate"]["multiseed_requirement_satisfied"] is False


def test_analysis_fails_when_one_mode_changes_labels() -> None:
    payload = _payload()
    payload["modes"][2]["instances"][0]["paratope"]["labels"] = [0, 1, 0, 1]
    with pytest.raises(GeometryStatisticsError, match="changes labels"):
        analyze_geometry_predictions(
            payload,
            bootstrap_iterations=20,
            permutation_iterations=20,
        )


def test_analysis_marks_small_component_universe_non_authorizing() -> None:
    result = analyze_geometry_predictions(
        _payload(count=4),
        bootstrap_iterations=20,
        permutation_iterations=20,
    )
    primary = result["comparisons"][
        "distance_direction_minus_topology_only"
    ]["joint_macro_auprc"]["paired_component_bootstrap"]
    assert primary["inferential_authorized"] is False
    assert (
        result["primary_gate"]["statistical_conditions"][
            "bootstrap_lower_bound_above_zero"
        ]
        is False
    )


def test_analysis_rejects_probability_outside_unit_interval() -> None:
    payload = _payload()
    payload["modes"][0]["instances"][0]["epitope"]["probabilities"][0] = 1.1
    with pytest.raises(GeometryStatisticsError, match=r"finite in \[0, 1\]"):
        analyze_geometry_predictions(
            payload,
            bootstrap_iterations=20,
            permutation_iterations=20,
        )
