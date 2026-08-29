"""One-shot source and consumption controls for SAbDab2 frozen testing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opening_authenticates_all_evaluation_sources(tmp_path: Path) -> None:
    from interaction_first.sabdab2_frozen_test import (
        FROZEN_BOOTSTRAP,
        FROZEN_COORDINATE_NOISE_ANGSTROM,
        FROZEN_REQUIRED_INTERVENTIONS,
        OPENING_FORMAT,
        authenticate_sabdab2_test_opening,
        sabdab2_evaluation_source_hashes,
    )

    state = tmp_path / "state"
    state.mkdir()
    sources = sabdab2_evaluation_source_hashes()
    opening = tmp_path / "opening.json"
    opening.write_text(
        json.dumps(
            {
                "format_version": OPENING_FORMAT,
                "eligible_to_open": True,
                "test_score_session_count_before": 0,
                "test_predictions_or_scores_read_at_freeze": False,
                "evaluation_source_dependencies": sources,
                "test_evaluator_source_sha256": sources[
                    "scripts/evaluate_framecontact_sabdab2_ensemble_test.py"
                ],
                "selection_receipt_sha256": "1" * 64,
                "selection_result_sha256": "2" * 64,
                "selected_ensemble_result_sha256": "3" * 64,
                "selected_ensemble_receipt_sha256": "4" * 64,
                "selected_experiment": "framecontact_interleaved",
                "members": [
                    {
                        "seed": seed,
                        "development_result_sha256": "5" * 64,
                        "selected_checkpoint_sha256": "6" * 64,
                    }
                    for seed in (17, 42, 73, 101, 137)
                ],
                "structure_manifest_sha256": "7" * 64,
                "contact_manifest_sha256": "8" * 64,
                "embedding_manifest_sha256": "9" * 64,
                "split_manifest_sha256": "a" * 64,
                "test_partner_panel_sha256": "b" * 64,
                "same_antigen_challenge_sha256": "c" * 64,
                "test_instance_count": 33,
                "test_partner_panel_size": 31,
                "same_antigen_maximum_epitope_jaccard": 0.5,
                "coordinate_noise_angstrom": list(
                    FROZEN_COORDINATE_NOISE_ANGSTROM
                ),
                "bootstrap": FROZEN_BOOTSTRAP,
                "required_interventions": list(FROZEN_REQUIRED_INTERVENTIONS),
                "test_state_root": str(state),
                "score_consumption_token_name": "SABDAB2_TEST_SCORE_CONSUMED.json",
            }
        ),
        encoding="utf-8",
    )
    authenticate_sabdab2_test_opening(opening, _sha(opening))
    payload = json.loads(opening.read_text())
    payload["evaluation_source_dependencies"]["scripts/train_framecontact.py"] = (
        "0" * 64
    )
    opening.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ineligible"):
        authenticate_sabdab2_test_opening(opening, _sha(opening))


def test_opening_rejects_confirmatory_parameter_drift(tmp_path: Path) -> None:
    from interaction_first.sabdab2_frozen_test import (
        OPENING_FORMAT,
        authenticate_sabdab2_test_opening,
        sabdab2_evaluation_source_hashes,
    )

    state = tmp_path / "state"
    state.mkdir()
    sources = sabdab2_evaluation_source_hashes()
    opening = tmp_path / "opening.json"
    opening.write_text(
        json.dumps(
            {
                "format_version": OPENING_FORMAT,
                "eligible_to_open": True,
                "test_score_session_count_before": 0,
                "test_predictions_or_scores_read_at_freeze": False,
                "evaluation_source_dependencies": sources,
                "test_evaluator_source_sha256": sources[
                    "scripts/evaluate_framecontact_sabdab2_ensemble_test.py"
                ],
                "selection_receipt_sha256": "1" * 64,
                "selection_result_sha256": "2" * 64,
                "selected_ensemble_result_sha256": "3" * 64,
                "selected_ensemble_receipt_sha256": "4" * 64,
                "selected_experiment": "framecontact_interleaved",
                "members": [
                    {
                        "seed": seed,
                        "development_result_sha256": "5" * 64,
                        "selected_checkpoint_sha256": "6" * 64,
                    }
                    for seed in (17, 42, 73, 101, 137)
                ],
                "structure_manifest_sha256": "7" * 64,
                "contact_manifest_sha256": "8" * 64,
                "embedding_manifest_sha256": "9" * 64,
                "split_manifest_sha256": "a" * 64,
                "test_partner_panel_sha256": "b" * 64,
                "same_antigen_challenge_sha256": "c" * 64,
                "test_instance_count": 33,
                "test_partner_panel_size": 7,
                "same_antigen_maximum_epitope_jaccard": 0.5,
                "coordinate_noise_angstrom": [0.1, 0.25, 0.5],
                "bootstrap": {"replicates": 10_000, "seed": 20_260_730},
                "required_interventions": [
                    "geometry_shuffle",
                    "local_frame_corruption",
                    "degree_matched_graph_rewiring",
                    "coordinate_noise_0.1_angstrom",
                    "coordinate_noise_0.25_angstrom",
                    "coordinate_noise_0.5_angstrom",
                    "independent_monomer_se3_prediction_invariance",
                ],
                "test_state_root": str(state),
                "score_consumption_token_name": (
                    "SABDAB2_TEST_SCORE_CONSUMED.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ineligible"):
        authenticate_sabdab2_test_opening(opening, _sha(opening))


def test_score_consumption_is_create_only(tmp_path: Path) -> None:
    from interaction_first.sabdab2_frozen_test import consume_sabdab2_score_once

    opening = {
        "test_state_root": str(tmp_path),
        "score_consumption_token_name": "SABDAB2_TEST_SCORE_CONSUMED.json",
        "selected_ensemble_result_sha256": "a" * 64,
    }
    token = consume_sabdab2_score_once(
        opening,
        opening_sha256="b" * 64,
        output_root=tmp_path / "result",
    )
    assert json.loads(token.read_text())["state"] == (
        "frozen_test_scoring_started_fail_closed"
    )
    with pytest.raises(FileExistsError):
        consume_sabdab2_score_once(
            opening,
            opening_sha256="b" * 64,
            output_root=tmp_path / "other-result",
        )
