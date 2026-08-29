from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_framecontact_live_conditions.py"
)
SPEC = importlib.util.spec_from_file_location("compare_framecontact_live_conditions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(epoch: int, value: float) -> dict:
    binary = {
        "auprc": value,
        "brier": value,
        "ece_15": value,
        "mean_per_complex_auprc": value,
        "precision": value,
        "recall": value,
        "f1": value,
        "mcc": value,
        "predicted_positive_ratio": value,
        "threshold_selected_on_validation": 0.5,
        "fixed_raw_logit_threshold_zero": {
            "precision": value,
            "recall": value,
            "f1": value,
            "mcc": value,
            "predicted_positive_ratio": value,
        },
    }
    return {
        "epoch": epoch,
        "elapsed_seconds": epoch * 60,
        "train": {"loss": value, "contact_loss": value},
        "validation": {
            "selection_metric": value,
            "contact": binary,
            "paratope": binary,
            "epitope": binary,
        },
    }


def _write(path: Path, values: list[tuple[int, float]]) -> None:
    path.write_text(
        "\n".join(json.dumps(_record(epoch, value)) for epoch, value in values)
        + "\n",
        encoding="utf-8",
    )


def test_compare_conditions_uses_latest_epoch_shared_by_every_seed(tmp_path: Path) -> None:
    # Left condition index 0 uses tasks 0/1; right condition index 1 uses 2/3.
    _write(tmp_path / "left-10_0.out", [(1, 1.0), (2, 2.0), (3, 9.0)])
    _write(tmp_path / "left-10_1.out", [(1, 3.0), (2, 4.0)])
    _write(tmp_path / "right-20_2.out", [(1, 2.0), (2, 5.0), (3, 10.0)])
    _write(tmp_path / "right-20_3.out", [(1, 5.0), (2, 8.0), (3, 11.0)])

    result = MODULE.compare_conditions(
        log_root=tmp_path,
        left_label="left",
        left_log_prefix="left",
        left_job_id="10",
        left_condition_index=0,
        right_label="right",
        right_log_prefix="right",
        right_job_id="20",
        right_condition_index=1,
        seeds=[17, 42],
        metrics=["contact_selected_precision", "selection_margin"],
    )

    assert result["matched_epoch"] == 2
    precision = result["metrics"]["contact_selected_precision"]
    assert precision["left"]["mean"] == pytest.approx(3.0)
    assert precision["right"]["mean"] == pytest.approx(6.5)
    assert precision["right_minus_left_paired"]["mean"] == pytest.approx(3.5)
    assert precision["per_seed_right_minus_left"] == {"17": 3.0, "42": 4.0}
    diagnostics = precision["paired_small_sample_diagnostics"]
    assert diagnostics["median_delta"] == pytest.approx(3.5)
    assert diagnostics["positive_seed_fraction"] == 1.0
    assert diagnostics["exact_two_sided_sign_flip_p"] == pytest.approx(0.5)
    assert diagnostics["minimum_attainable_two_sided_p_at_n"] == pytest.approx(0.5)
    assert diagnostics["leave_one_seed_out_mean_delta_range"] == [3.0, 4.0]


def test_paired_uncertainty_reports_mixed_seed_directions() -> None:
    diagnostics = MODULE._paired_uncertainty([1.0, -2.0, 3.0])
    assert diagnostics["median_delta"] == 1.0
    assert diagnostics["positive_seed_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["negative_seed_fraction"] == pytest.approx(1 / 3)
    assert diagnostics["exact_sign_flip_permutation_count"] == 8
    assert diagnostics["minimum_attainable_two_sided_p_at_n"] == pytest.approx(0.25)


def test_compare_conditions_rejects_missing_synchronized_epoch(tmp_path: Path) -> None:
    _write(tmp_path / "left-10_0.out", [(1, 1.0)])
    _write(tmp_path / "right-20_0.out", [(2, 2.0)])
    with pytest.raises(ValueError, match="no epoch"):
        MODULE.compare_conditions(
            log_root=tmp_path,
            left_label="left",
            left_log_prefix="left",
            left_job_id="10",
            left_condition_index=0,
            right_label="right",
            right_log_prefix="right",
            right_job_id="20",
            right_condition_index=0,
            seeds=[17],
        )
