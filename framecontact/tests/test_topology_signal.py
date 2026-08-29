import numpy as np

from interaction_first.topology_signal import (
    paired_relation_delta,
    positive_component_sizes,
    relation_label_signal,
    sequence_relation_mask,
    spatial_relation_mask,
)


def _backbone(ca: np.ndarray) -> np.ndarray:
    backbone = np.zeros((len(ca), 3, 3), dtype=np.float32)
    backbone[:, 0] = ca + np.asarray([-1.0, 0.0, 0.0])
    backbone[:, 1] = ca
    backbone[:, 2] = ca + np.asarray([1.0, 0.0, 0.0])
    return backbone


def test_spatial_relation_respects_radius_chain_and_missing_frame() -> None:
    ca = np.asarray(
        [[0, 0, 0], [0, 0, 5], [0, 0, 7], [0, 0, 2]], dtype=np.float32
    )
    mask = spatial_relation_mask(
        _backbone(ca),
        np.asarray([0, 0, 1, 0]),
        np.asarray([True, True, True, False]),
        radius=6.0,
    )
    expected = np.zeros((4, 4), dtype=np.bool_)
    expected[0, 1] = expected[1, 0] = True
    assert np.array_equal(mask, expected)


def test_sequence_relation_does_not_cross_chain_or_keep_self_edges() -> None:
    mask = sequence_relation_mask(
        np.asarray([0, 0, 1, 1]),
        np.ones(4, dtype=np.bool_),
        window=2,
    )
    assert mask[0, 1] and mask[2, 3]
    assert not mask[1, 2]
    assert not np.diag(mask).any()


def test_label_signal_detects_positive_spatial_cluster() -> None:
    relation = np.asarray(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.bool_,
    )
    labels = np.asarray([1, 1, 0, 0], dtype=np.float32)
    result = relation_label_signal(relation, labels)
    assert result["query_count"] == 4
    assert result["positive_component_count"] == 1
    assert result["positive_largest_component_fraction"] == 1.0
    assert result["positive_neighbor_rate"] > result["negative_neighbor_rate"]
    assert result["positive_neighbor_lift"] > 1.0


def test_components_do_not_merge_without_positive_path() -> None:
    relation = np.asarray(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.bool_,
    )
    assert positive_component_sizes(relation, np.asarray([1, 0, 0, 1])) == [1, 1]


def test_paired_delta_is_within_complex() -> None:
    rows = [
        {
            "relations": {
                "left": {"metric": 0.5},
                "right": {"metric": 0.2},
            }
        },
        {
            "relations": {
                "left": {"metric": 0.4},
                "right": {"metric": 0.3},
            }
        },
    ]
    result = paired_relation_delta(
        rows,
        left="left",
        right="right",
        metric="metric",
        bootstrap_replicates=100,
        seed=7,
    )
    assert np.isclose(result["mean"], 0.2)
    assert result["count"] == 2
