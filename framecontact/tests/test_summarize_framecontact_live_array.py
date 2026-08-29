from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_framecontact_live_array.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_framecontact_live_array", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(epoch: int, value: float) -> dict:
    metric = {
        "auprc": value,
        "brier": value + 0.1,
        "ece_15": value + 0.2,
        "mean_per_complex_auprc": value + 0.3,
        "precision": value + 0.4,
        "recall": value + 0.5,
        "f1": value + 0.45,
        "mcc": value + 0.35,
        "predicted_positive_ratio": value + 0.25,
        "threshold_selected_on_validation": value + 0.6,
        "fixed_raw_logit_threshold_zero": {
            "precision": value + 0.7,
            "recall": value + 0.8,
            "f1": value + 0.75,
            "mcc": value + 0.65,
            "predicted_positive_ratio": value + 0.9,
        },
        "ranked_pair_macro": {
            "10": {
                "mean_precision": value + 0.11,
                "mean_recall": value + 0.12,
                "mean_effective_k": 10.0,
            },
            "lmin_over_5": {
                "mean_precision": value + 0.13,
                "mean_recall": value + 0.14,
                "mean_effective_k": 20.0,
            },
        },
    }
    residue = {
        "auprc": value + 1.0,
        "mean_per_complex_auprc": value + 1.1,
        "precision": value + 1.2,
        "recall": value + 1.3,
        "f1": value + 1.4,
        "mcc": value + 1.5,
        "brier": value + 1.6,
        "ece_15": value + 1.7,
        "predicted_positive_ratio": value + 1.8,
    }
    return {
        "epoch": epoch,
        "elapsed_seconds": 60.0 * epoch,
        "train": {
            "loss": value,
            "contact_loss": value + 0.1,
            "distogram_loss": value + 0.2,
            "triangle_loss": value + 0.3,
        },
        "validation": {
            "contact": metric,
            "paratope": residue,
            "epitope": residue,
            "selection_metric": value + 1.1,
            "partner_retrieval_by_direction": {
                "fixed_antibody_wrong_antigen": {
                    "cluster_weighted_cognate_compatibility_margin": value + 2.0,
                    "top1_retrieval_accuracy": value + 2.1,
                    "mean_reciprocal_rank": value + 2.2,
                },
                "fixed_antigen_wrong_antibody": {
                    "cluster_weighted_cognate_compatibility_margin": value + 3.0,
                    "top1_retrieval_accuracy": value + 3.1,
                    "mean_reciprocal_rank": value + 3.2,
                },
            },
        },
    }


def test_live_summary_keeps_only_epochs_with_required_seed_count(tmp_path: Path) -> None:
    paths = [tmp_path / f"fc-123_{index}.out" for index in range(2)]
    paths[0].write_text(
        "\n".join(json.dumps(_record(epoch, float(epoch))) for epoch in (1, 2))
        + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(json.dumps(_record(1, 3.0)) + "\n", encoding="utf-8")

    summary = MODULE.summarize_array(
        log_root=tmp_path,
        log_prefix="fc",
        job_id="123",
        conditions=["condition"],
        seeds=[17, 42],
        minimum_seeds=2,
    )

    condition = summary["conditions"]["condition"]
    assert condition["epochs_logged_by_seed"] == {"17": [1, 2], "42": [1]}
    trajectory = condition["complete_epoch_trajectory"]
    assert [entry["epoch"] for entry in trajectory] == [1]
    pooled = trajectory[0]["metrics"]["contact_pooled_auprc"]
    assert pooled["n"] == 2
    assert pooled["mean"] == pytest.approx(2.0)
    assert trajectory[0]["per_seed"]["17"]["contact_selected_precision"] == 1.4
    assert trajectory[0]["per_seed"]["17"]["contact_fixed_raw_precision"] == 1.7
    assert trajectory[0]["per_seed"]["17"]["contact_selected_f1"] == 1.45
    assert trajectory[0]["per_seed"]["17"]["contact_ranked_10_precision"] == 1.11
    assert trajectory[0]["per_seed"]["17"]["epitope_recall"] == 2.3
    assert (
        trajectory[0]["per_seed"]["17"]["retrieval_worst_directional_margin"]
        == 3.0
    )


def test_live_summary_reports_descriptive_epoch_trends(tmp_path: Path) -> None:
    path = tmp_path / "fc-123_0.out"
    path.write_text(
        "\n".join(
            json.dumps(_record(epoch, value))
            for epoch, value in ((1, 0.1), (2, 0.2), (3, 0.4))
        )
        + "\n",
        encoding="utf-8",
    )

    summary = MODULE.summarize_array(
        log_root=tmp_path,
        log_prefix="fc",
        job_id="123",
        conditions=["condition"],
        seeds=[17],
        minimum_seeds=1,
    )
    diagnostics = summary["conditions"]["condition"]["trajectory_diagnostics"]
    precision = diagnostics["contact_selected_precision"]
    assert precision["optimization_direction"] == "higher_is_better"
    assert precision["last_minus_first"] == pytest.approx(0.3)
    assert precision["least_squares_slope_per_epoch"] == pytest.approx(0.15)
    assert precision["improving_transition_fraction"] == 1.0
    assert precision["best_epoch_descriptive_only"] == 3

    loss = diagnostics["train_loss"]
    assert loss["optimization_direction"] == "lower_is_better"
    assert loss["improving_transition_fraction"] == 0.0
    assert loss["best_epoch_descriptive_only"] == 1

    compact = MODULE.compact_summary(summary)
    compact_condition = compact["conditions"]["condition"]
    assert "complete_epoch_trajectory" not in compact_condition
    assert compact_condition["latest_synchronized_epoch"]["epoch"] == 3
    assert compact_condition["latest_synchronized_epoch"]["seed_count"] == 1
    assert (
        compact_condition["latest_synchronized_epoch"]["metrics"]
        ["contact_selected_precision"]["mean"]
        == pytest.approx(0.8)
    )


def test_live_summary_ignores_source_verification_lines(tmp_path: Path) -> None:
    path = tmp_path / "fc-123_0.out"
    path.write_text(
        json.dumps({"valid": True, "tree_sha256": "abc"})
        + "\n"
        + json.dumps(_record(1, 1.0))
        + "\n",
        encoding="utf-8",
    )

    summary = MODULE.summarize_array(
        log_root=tmp_path,
        log_prefix="fc",
        job_id="123",
        conditions=["condition"],
        seeds=[17],
        minimum_seeds=1,
    )
    trajectory = summary["conditions"]["condition"]["complete_epoch_trajectory"]
    assert [entry["epoch"] for entry in trajectory] == [1]


def test_live_summary_rejects_duplicate_epochs(tmp_path: Path) -> None:
    path = tmp_path / "fc-123_0.out"
    line = json.dumps(_record(1, 1.0))
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate epoch 1"):
        MODULE.summarize_array(
            log_root=tmp_path,
            log_prefix="fc",
            job_id="123",
            conditions=["condition"],
            seeds=[17],
            minimum_seeds=1,
        )
