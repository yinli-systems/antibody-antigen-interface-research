from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from extensions.frontier_multimodel_v1.reparank_same_protocol import (
    checkpoint_hyperparameters,
    equal_component_bootstrap,
    model_state,
    parse_backbone_pdb,
    per_complex_metrics,
    tie_aware_first_positive,
)


def test_checkpoint_hyperparameters_fill_only_declared_defaults() -> None:
    payload = {
        "hyper_parameters": {
            "dims": 512,
            "num_heads": 16,
            "hidden_dims": 1024,
            "num_sequence_layers": 8,
            "num_complex_layers": 8,
            "skip_position_encoding": True,
            "feature_dims": None,
        }
    }
    result = checkpoint_hyperparameters(payload)
    assert result["dims"] == 512
    assert result["ax_dims"] == 32
    assert result["pos_num"] == 1
    assert result["skip_position_encoding"] is True
    assert result["feature_dims"] is None


def test_checkpoint_hyperparameters_reject_missing_core_parameter() -> None:
    with pytest.raises(ValueError, match="could not locate"):
        checkpoint_hyperparameters({"hyper_parameters": {"dims": 512}})


def test_model_state_strips_only_lightning_model_prefix() -> None:
    payload = {
        "state_dict": {
            "_model.layer.weight": np.array([1.0]),
            "_model.layer.bias": np.array([0.0]),
            "_quantiles": np.array([0.5]),
        }
    }
    prefix, state = model_state(payload)
    assert prefix == "_model."
    assert set(state) == {"layer.weight", "layer.bias"}


def test_model_state_rejects_only_wrapper_buffers() -> None:
    with pytest.raises(ValueError, match="isolate"):
        model_state({"state_dict": {"_quantiles": np.array([0.5])}})


def test_parse_backbone_pdb_requires_exact_sequence_and_four_atoms(tmp_path: Path) -> None:
    pdb = tmp_path / "unit.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 20.00           C\n"
        "ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00 20.00           O\n"
        "ATOM      5  N   GLY A   2       5.000   6.000   7.000  1.00 20.00           N\n"
        "ATOM      6  CA  GLY A   2       6.000   7.000   8.000  1.00 20.00           C\n"
        "ATOM      7  C   GLY A   2       7.000   8.000   9.000  1.00 20.00           C\n"
        "ATOM      8  O   GLY A   2       8.000   9.000  10.000  1.00 20.00           O\nEND\n",
        encoding="utf-8",
    )
    array = parse_backbone_pdb(pdb, "AG")
    assert array.shape == (2, 4, 3)
    assert np.allclose(array[0, 1], [2, 3, 4])
    with pytest.raises(ValueError, match="sequence mismatch"):
        parse_backbone_pdb(pdb, "AA")


def test_parse_backbone_pdb_rejects_missing_oxygen(tmp_path: Path) -> None:
    pdb = tmp_path / "bad.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 20.00           C\nEND\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete backbone"):
        parse_backbone_pdb(pdb, "A")


def test_tie_aware_first_positive_uses_expected_rank() -> None:
    result = tie_aware_first_positive(
        np.array([1.0, 1.0, 1.0, 0.0]),
        np.array([True, False, False, False]),
    )
    assert result["recall_at_1"] == pytest.approx(1 / 3)
    assert result["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)


def test_per_complex_metrics_use_higher_score_as_more_contact_like() -> None:
    good = per_complex_metrics(
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([4.0, 3.0, 2.0, 1.0]),
    )
    bad = per_complex_metrics(
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert good["average_precision"] == 1.0
    assert good["mrr"] == 1.0
    assert bad["average_precision"] < good["average_precision"]
    assert bad["mrr"] == pytest.approx(0.25)


def test_per_complex_metrics_reject_undefined_binary_problem() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        per_complex_metrics(np.ones(4, dtype=np.uint8), np.arange(4, dtype=float))


def test_equal_component_bootstrap_weights_components_not_rows() -> None:
    values = {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 1.0}
    components = {"a1": "a", "a2": "a", "a3": "a", "b1": "b"}
    result = equal_component_bootstrap(values, components, repeats=1000)
    assert result["estimate"] == pytest.approx(0.5)
    assert result["complex_weighted_estimate"] == pytest.approx(0.25)
    assert result["component_count"] == 2


def test_protocol_names_overlap_and_threshold_boundaries() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "extensions/frontier_multimodel_v1/REPARANK_SAME_PROTOCOL_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "training accession list" in text
    assert "Threshold metrics never" in text
    assert "Experimental bound antibody-antigen pose is forbidden" in text
    assert "binary BCE would be a different model" in text
