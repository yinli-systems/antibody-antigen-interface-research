import numpy as np

from interaction_first.chimera_case_study import (
    connected_component_sizes,
    geometry_descriptors,
    mean_delta_group_bootstrap,
    prediction_metrics,
    quantile_strata,
)


def test_connected_components_respect_distance_threshold() -> None:
    coords = np.asarray([[0, 0, 0], [0, 0, 7], [30, 0, 0]], dtype=np.float64)
    assert connected_component_sizes(coords, 8.0) == [2, 1]
    assert connected_component_sizes(coords, 6.0) == [1, 1, 1]


def test_geometry_descriptors_capture_sequence_and_spatial_dispersion() -> None:
    ca = np.asarray(
        [[0, 0, 0], [0, 0, 4], [0, 0, 8], [30, 0, 0], [30, 0, 4]],
        dtype=np.float32,
    )
    backbone = np.repeat(ca[:, None, :], 3, axis=1)
    labels = np.asarray([1, 1, 0, 1, 0], dtype=np.float32)
    result = geometry_descriptors(
        backbone,
        np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        np.ones(5, dtype=np.bool_),
        labels,
    )
    assert result["antigen_length"] == 5
    assert result["epitope_count"] == 3
    assert result["epitope_sequence_run_count"] == 2
    assert result["epitope_components_8a"] == 2
    assert result["epitope_largest_component_fraction_8a"] == 2 / 3


def test_prediction_metrics_include_rank_and_fixed_threshold_metrics() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.float32)
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
    result = prediction_metrics(labels, probabilities)
    assert result["precision_at_label_count"] == 0.5
    assert result["precision_at_0_5"] == 2 / 3
    assert result["recall_at_0_5"] == 1.0


def test_quantile_strata_preserve_all_rows() -> None:
    rows = [
        {
            "antigen_length": value,
            "delta_auprc_spatial_minus_dense": value / 100.0,
            "sequence_dense_auprc": 0.2,
            "spatial_topology_auprc": 0.2 + value / 100.0,
        }
        for value in range(1, 9)
    ]
    result = quantile_strata(rows, "antigen_length")
    assert sum(record["n"] for record in result["bins"]) == len(rows)


def test_group_bootstrap_reports_candidate_means() -> None:
    rows = [
        {
            "antigen_sequence_sha256": f"group-{index}",
            "sequence_dense_auprc": 0.2,
            "spatial_topology_auprc": 0.3,
            "delta_auprc_spatial_minus_dense": 0.1,
        }
        for index in range(4)
    ]
    result = mean_delta_group_bootstrap(rows, replicates=100, seed=7)
    assert result["n"] == 4
    assert result["group_count"] == 4
    assert np.isclose(result["mean_delta_auprc"], 0.1)
    assert np.isclose(result["ci_low"], 0.1)
    mean_delta_group_bootstrap,
