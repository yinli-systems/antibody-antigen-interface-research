"""End-to-end test of the one-time frozen external array scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn is unavailable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_scorer_enforces_axes_and_reports_primary_interval(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    scorer = repository / "scripts" / "score_framecontact_external.py"
    instance_ids = np.asarray(["case_a", "case_b"], dtype=np.str_)
    offsets = np.asarray([0, 4, 8], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray(
        [0.9, 0.1, 0.2, 0.8, 0.85, 0.05, 0.75, 0.15], dtype=np.float32
    )
    panel_sha = "a" * 64
    prediction_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        format_version=np.asarray("framecontact_external_predictions_v1"),
        instance_ids=instance_ids,
        test_partner_panel_sha256=np.asarray(panel_sha),
        contact_offsets=offsets,
        contact_probabilities=probabilities,
        paratope_offsets=offsets,
        paratope_probabilities=probabilities,
        epitope_offsets=offsets,
        epitope_probabilities=probabilities,
        correct_partner_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        wrong_partner_scores=np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
    )
    truth_path = tmp_path / "truth.npz"
    np.savez_compressed(
        truth_path,
        format_version=np.asarray("framecontact_external_truth_v1"),
        instance_ids=instance_ids,
        antigen_cluster_ids=np.asarray(["cluster_a", "cluster_b"], dtype=np.str_),
        contact_offsets=offsets,
        contact_labels=labels,
        paratope_offsets=offsets,
        paratope_labels=labels,
        epitope_offsets=offsets,
        epitope_labels=labels,
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_validation_thresholds_v1",
                "selected_on_validation_only": True,
                "thresholds": {"contact": 0.5, "paratope": 0.5, "epitope": 0.5},
            }
        ),
        encoding="utf-8",
    )
    opening_path = tmp_path / "opening.json"
    opening_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_external_opening_receipt_v1",
                "eligible_to_open": True,
                "test_labels_read_at_freeze": False,
                "test_open_count_before": 0,
                "scorer_source_sha256": _sha256(scorer),
                "test_truth_authority_sha256": _sha256(truth_path),
                "thresholds_sha256": _sha256(threshold_path),
                "test_partner_panel_sha256": panel_sha,
                "test_partner_panel_size": 2,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable,
            str(scorer),
            "--opening-receipt",
            str(opening_path),
            "--opening-receipt-sha256",
            _sha256(opening_path),
            "--predictions",
            str(prediction_path),
            "--predictions-sha256",
            _sha256(prediction_path),
            "--truth",
            str(truth_path),
            "--truth-sha256",
            _sha256(truth_path),
            "--thresholds",
            str(threshold_path),
            "--thresholds-sha256",
            _sha256(threshold_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "primary_success" in completed.stdout
    result = json.loads(
        (output_root / "FRAMECONTACT_FROZEN_EXTERNAL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["test_open_count"] == 1
    assert result["primary_endpoint"]["success"] is True
    assert result["partner_retrieval"]["top1_retrieval_accuracy"] == 1.0
    assert result["contact"]["auprc"] == 1.0
    assert result["contact"]["calibration_slope"] is not None
    assert result["contact"]["eligible_per_complex_auprc_count"] == 2
    assert result["contact"]["ineligible_per_complex_auprc_count"] == 0
    assert result["contact"]["cluster_macro_per_complex_auprc_ci95"][
        "group_count"
    ] == 2


def test_external_scorer_v2_reports_fixed_antigen_and_reciprocal_retrieval(
    tmp_path: Path,
):
    repository = Path(__file__).resolve().parents[1]
    scorer = repository / "scripts" / "score_framecontact_external.py"
    instance_ids = np.asarray(["case_a", "case_b"], dtype=np.str_)
    offsets = np.asarray([0, 4, 8], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    probabilities = np.asarray(
        [0.9, 0.1, 0.2, 0.8, 0.85, 0.05, 0.75, 0.15], dtype=np.float32
    )
    panel_sha = "b" * 64
    prediction_path = tmp_path / "predictions_v2.npz"
    np.savez_compressed(
        prediction_path,
        format_version=np.asarray("framecontact_external_predictions_v2"),
        instance_ids=instance_ids,
        test_partner_panel_sha256=np.asarray(panel_sha),
        contact_offsets=offsets,
        contact_probabilities=probabilities,
        paratope_offsets=offsets,
        paratope_probabilities=probabilities,
        epitope_offsets=offsets,
        epitope_probabilities=probabilities,
        correct_partner_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        wrong_antigen_scores=np.asarray(
            [[0.2, 0.3], [0.4, 0.5]], dtype=np.float32
        ),
        wrong_antibody_scores=np.asarray(
            [[0.1, 0.4], [0.2, 0.6]], dtype=np.float32
        ),
    )
    truth_path = tmp_path / "truth_v2.npz"
    np.savez_compressed(
        truth_path,
        format_version=np.asarray("framecontact_external_truth_v2"),
        instance_ids=instance_ids,
        antigen_cluster_ids=np.asarray(["ag_a", "ag_b"], dtype=np.str_),
        antibody_cluster_ids=np.asarray(["ab_a", "ab_b"], dtype=np.str_),
        contact_offsets=offsets,
        contact_labels=labels,
        paratope_offsets=offsets,
        paratope_labels=labels,
        epitope_offsets=offsets,
        epitope_labels=labels,
    )
    threshold_path = tmp_path / "thresholds_v2.json"
    threshold_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_validation_thresholds_v1",
                "selected_on_validation_only": True,
                "thresholds": {"contact": 0.5, "paratope": 0.5, "epitope": 0.5},
            }
        ),
        encoding="utf-8",
    )
    opening_path = tmp_path / "opening_v2.json"
    opening_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_external_opening_receipt_v2",
                "eligible_to_open": True,
                "test_labels_read_at_freeze": False,
                "test_open_count_before": 0,
                "scorer_source_sha256": _sha256(scorer),
                "test_truth_authority_sha256": _sha256(truth_path),
                "thresholds_sha256": _sha256(threshold_path),
                "test_partner_panel_sha256": panel_sha,
                "test_partner_panel_size": 2,
                "test_partner_panel_directions": 2,
                "specificity_score_source": "interaction_only",
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "result_v2"
    subprocess.run(
        [
            sys.executable,
            str(scorer),
            "--opening-receipt",
            str(opening_path),
            "--opening-receipt-sha256",
            _sha256(opening_path),
            "--predictions",
            str(prediction_path),
            "--predictions-sha256",
            _sha256(prediction_path),
            "--truth",
            str(truth_path),
            "--truth-sha256",
            _sha256(truth_path),
            "--thresholds",
            str(threshold_path),
            "--thresholds-sha256",
            _sha256(threshold_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(
        (output_root / "FRAMECONTACT_FROZEN_EXTERNAL_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["format_version"] == "framecontact_frozen_external_result_v2"
    assert result["primary_endpoint"]["name"] == (
        "fixed_antigen_cognate_antibody_interaction_margin"
    )
    assert result["primary_endpoint"]["success"] is True
    assert result["required_reciprocal_success"] is True
    assert set(result["partner_retrieval_by_direction"]) == {
        "fixed_antigen_wrong_antibody",
        "fixed_antibody_wrong_antigen",
    }


def test_external_scorer_keeps_zero_positive_complex_in_pooled_metrics_only(
    tmp_path: Path,
):
    repository = Path(__file__).resolve().parents[1]
    scorer = repository / "scripts" / "score_framecontact_external.py"
    instance_ids = np.asarray(["positive", "zero_positive"], dtype=np.str_)
    offsets = np.asarray([0, 4, 8], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    probabilities = np.asarray(
        [0.9, 0.1, 0.2, 0.8, 0.3, 0.2, 0.1, 0.05], dtype=np.float32
    )
    panel_sha = "c" * 64
    prediction_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        format_version=np.asarray("framecontact_external_predictions_v2"),
        instance_ids=instance_ids,
        test_partner_panel_sha256=np.asarray(panel_sha),
        contact_offsets=offsets,
        contact_probabilities=probabilities,
        paratope_offsets=offsets,
        paratope_probabilities=probabilities,
        epitope_offsets=offsets,
        epitope_probabilities=probabilities,
        correct_partner_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        wrong_antigen_scores=np.asarray([[0.1], [0.2]], dtype=np.float32),
        wrong_antibody_scores=np.asarray([[0.1], [0.2]], dtype=np.float32),
    )
    truth_path = tmp_path / "truth.npz"
    np.savez_compressed(
        truth_path,
        format_version=np.asarray("framecontact_external_truth_v2"),
        instance_ids=instance_ids,
        antigen_cluster_ids=np.asarray(["ag_a", "ag_b"], dtype=np.str_),
        antibody_cluster_ids=np.asarray(["ab_a", "ab_b"], dtype=np.str_),
        contact_offsets=offsets,
        contact_labels=labels,
        paratope_offsets=offsets,
        paratope_labels=labels,
        epitope_offsets=offsets,
        epitope_labels=labels,
    )
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_validation_thresholds_v1",
                "selected_on_validation_only": True,
                "thresholds": {"contact": 0.5, "paratope": 0.5, "epitope": 0.5},
            }
        ),
        encoding="utf-8",
    )
    opening_path = tmp_path / "opening.json"
    opening_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_external_opening_receipt_v2",
                "eligible_to_open": True,
                "test_labels_read_at_freeze": False,
                "test_open_count_before": 0,
                "scorer_source_sha256": _sha256(scorer),
                "test_truth_authority_sha256": _sha256(truth_path),
                "thresholds_sha256": _sha256(threshold_path),
                "test_partner_panel_sha256": panel_sha,
                "test_partner_panel_size": 1,
                "test_partner_panel_directions": 2,
                "specificity_score_source": "interaction_only",
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "result"
    subprocess.run(
        [
            sys.executable,
            str(scorer),
            "--opening-receipt",
            str(opening_path),
            "--opening-receipt-sha256",
            _sha256(opening_path),
            "--predictions",
            str(prediction_path),
            "--predictions-sha256",
            _sha256(prediction_path),
            "--truth",
            str(truth_path),
            "--truth-sha256",
            _sha256(truth_path),
            "--thresholds",
            str(threshold_path),
            "--thresholds-sha256",
            _sha256(threshold_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(
        (output_root / "FRAMECONTACT_FROZEN_EXTERNAL_RESULT.json").read_text()
    )
    assert result["contact"]["candidate_count"] == 8
    assert result["contact"]["eligible_per_complex_auprc_count"] == 1
    assert result["contact"]["ineligible_per_complex_auprc_count"] == 1
    assert result["contact"]["ineligible_per_complex_auprc"][0]["instance_id"] == (
        "zero_positive"
    )
    assert result["contact"]["cluster_macro_per_complex_auprc_ci95"]["reason"] == (
        "fewer_than_two_eligible_groups"
    )
