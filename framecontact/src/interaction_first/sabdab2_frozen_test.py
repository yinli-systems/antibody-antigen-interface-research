"""Hash-bound selection and one-shot opening controls for SAbDab2 test scoring."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


SELECTION_FORMAT = "framecontact_sabdab2_development_selection_v1"
ENSEMBLE_FORMAT = "framecontact_development_ensemble_v1"
OPENING_FORMAT = "framecontact_sabdab2_frozen_test_opening_v1"
SCORE_CONSUMPTION_FORMAT = "framecontact_sabdab2_test_score_consumption_v1"
FROZEN_TEST_PARTNER_PANEL_SIZE = 31
FROZEN_SAME_ANTIGEN_MAXIMUM_EPITOPE_JACCARD = 0.50
FROZEN_COORDINATE_NOISE_ANGSTROM = (0.10, 0.25, 0.50)
FROZEN_BOOTSTRAP = {"replicates": 10_000, "seed": 20_260_730}
FROZEN_REQUIRED_INTERVENTIONS = (
    "geometry_shuffle",
    "local_frame_corruption",
    "degree_matched_graph_rewiring",
    "coordinate_noise_0.1_angstrom",
    "coordinate_noise_0.25_angstrom",
    "coordinate_noise_0.5_angstrom",
    "independent_monomer_se3_prediction_invariance",
)
FROZEN_SEEDS = (17, 42, 73, 101, 137)
ARCHITECTURE_MATRIX = (
    "sequence_dense",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
    "frozen_frame_ipa",
    "framecontact",
    "framecontact_interleaved",
)
EVALUATION_DEPENDENCY_PATHS = (
    "scripts/evaluate_framecontact_sabdab2_ensemble_test.py",
    "scripts/freeze_framecontact_development_ensemble.py",
    "scripts/freeze_framecontact_development_selection.py",
    "scripts/train_framecontact.py",
    "src/interaction_first/sabdab2_frozen_test.py",
    "src/interaction_first/framecontact_ensemble.py",
    "src/interaction_first/framecontact_interventions.py",
    "src/interaction_first/framecontact_metrics.py",
    "src/interaction_first/same_antigen_challenge.py",
    "src/interaction_first/sabdab2_contacts.py",
    "src/interaction_first/sabdab2_structure.py",
)


class Sabdab2FrozenTestError(ValueError):
    """Raised when a frozen-test authority is incomplete or has drifted."""


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Sabdab2FrozenTestError(f"Cannot parse {name}: {path}") from error
    if not isinstance(value, dict):
        raise Sabdab2FrozenTestError(f"{name} must be a JSON object")
    return value


def sabdab2_evaluation_source_hashes(
    repository_root: Path | None = None,
) -> dict[str, str]:
    root = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else repository_root.resolve()
    )
    result = {}
    for relative in EVALUATION_DEPENDENCY_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise Sabdab2FrozenTestError(
                f"Missing regular SAbDab2 evaluation dependency: {relative}"
            )
        result[relative] = sha256_file(path)
    return result


def authenticate_sabdab2_development_selection(
    selection_root: Path,
    *,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    result_path = selection_root / "SABDAB2_DEVELOPMENT_SELECTION.json"
    receipt_path = selection_root / "SELECTION_RECEIPT.json"
    authenticate_file(
        receipt_path, selection_receipt_sha256, "SAbDab2 selection receipt"
    )
    receipt = _json(receipt_path, "SAbDab2 selection receipt")
    authenticate_file(
        result_path, str(receipt.get("result_sha256", "")), "SAbDab2 selection result"
    )
    result = _json(result_path, "SAbDab2 selection result")
    selected = result.get("selected_ensemble", {})
    candidates = result.get("all_candidates", [])
    rule = result.get("selection_rule", {})
    experiments = [
        candidate.get("experiment")
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    architecture_index = {
        experiment: index for index, experiment in enumerate(ARCHITECTURE_MATRIX)
    }
    scores_valid = all(
        isinstance(candidate.get("selection_metric"), (int, float))
        and math.isfinite(float(candidate["selection_metric"]))
        and isinstance(candidate.get("contact_mean_per_complex_auprc"), (int, float))
        and math.isfinite(float(candidate["contact_mean_per_complex_auprc"]))
        for candidate in candidates
        if isinstance(candidate, dict)
    )
    order_valid = all(
        (
            (
                float(left["selection_metric"]),
                float(left["contact_mean_per_complex_auprc"]),
            )
            > (
                float(right["selection_metric"]),
                float(right["contact_mean_per_complex_auprc"]),
            )
            or (
                (
                    float(left["selection_metric"]),
                    float(left["contact_mean_per_complex_auprc"]),
                )
                == (
                    float(right["selection_metric"]),
                    float(right["contact_mean_per_complex_auprc"]),
                )
                and architecture_index.get(left.get("experiment"), 99)
                < architecture_index.get(right.get("experiment"), 99)
            )
        )
        for left, right in zip(candidates, candidates[1:], strict=False)
        if isinstance(left, dict) and isinstance(right, dict)
    )
    if (
        receipt.get("format_version") != SELECTION_FORMAT
        or result.get("format_version") != SELECTION_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or receipt.get("selected_experiment") != result.get("selected_experiment")
        or receipt.get("selected_ensemble_result_sha256")
        != selected.get("result_sha256")
        or rule.get("primary")
        != "maximize minimum_direction_cluster_weighted_margin"
        or rule.get("first_tie_break")
        != "maximize mean_per_complex_contact_auprc"
        or rule.get("test_role_forbidden") is not True
        or len(candidates) != 8
        or set(experiments) != set(ARCHITECTURE_MATRIX)
        or not scores_valid
        or not order_valid
        or candidates[0].get("experiment") != result.get("selected_experiment")
        or any(
            candidates[0].get(key) != selected.get(key)
            for key in (
                "root",
                "result_sha256",
                "receipt_sha256",
                "selection_metric",
                "contact_mean_per_complex_auprc",
            )
        )
    ):
        raise Sabdab2FrozenTestError(
            "SAbDab2 development selection is not pre-opening eligible"
        )
    ensemble_root = Path(str(selected.get("root", "")))
    ensemble_result_path = ensemble_root / "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
    ensemble_receipt_path = ensemble_root / "ENSEMBLE_RECEIPT.json"
    authenticate_file(
        ensemble_result_path,
        str(selected.get("result_sha256", "")),
        "selected SAbDab2 ensemble result",
    )
    authenticate_file(
        ensemble_receipt_path,
        str(selected.get("receipt_sha256", "")),
        "selected SAbDab2 ensemble receipt",
    )
    ensemble = _json(ensemble_result_path, "selected SAbDab2 ensemble")
    ensemble_receipt = _json(ensemble_receipt_path, "selected SAbDab2 ensemble receipt")
    if (
        ensemble.get("format_version") != ENSEMBLE_FORMAT
        or ensemble_receipt.get("format_version") != ENSEMBLE_FORMAT
        or ensemble_receipt.get("result_sha256") != selected.get("result_sha256")
        or ensemble.get("experiment") != result.get("selected_experiment")
        or ensemble.get("seeds") != list(FROZEN_SEEDS)
        or ensemble.get("ensemble_rule", {}).get("member_weighting") != "uniform 1/5"
        or len(ensemble.get("authority", {}).get("members", [])) != 5
        or ensemble.get("authority", {}).get("test_loaded") is not False
        or ensemble.get("authority", {}).get("external_test_loaded") is not False
        or ensemble_receipt.get("test_loaded") is not False
        or ensemble_receipt.get("external_test_loaded") is not False
        or ensemble.get("validation", {}).get("selection_metric_rule")
        != "minimum_direction_cluster_weighted_margin"
        or ensemble.get("validation", {}).get("specificity_score_source")
        != "interaction_only"
        or ensemble.get("validation", {}).get("selection_metric")
        != selected.get("selection_metric")
        or ensemble.get("validation", {})
        .get("contact", {})
        .get("mean_per_complex_auprc")
        != selected.get("contact_mean_per_complex_auprc")
    ):
        raise Sabdab2FrozenTestError(
            "Selected SAbDab2 ensemble is not pre-opening eligible"
        )
    return {
        "selection_result": result,
        "selection_result_sha256": receipt["result_sha256"],
        "selection_receipt_sha256": selection_receipt_sha256,
        "selected_ensemble_root": str(ensemble_root),
        "selected_ensemble_result_sha256": selected["result_sha256"],
        "selected_ensemble_receipt_sha256": selected["receipt_sha256"],
    }


def authenticate_sabdab2_test_opening(
    opening_path: Path,
    opening_sha256: str,
) -> dict[str, Any]:
    authenticate_file(opening_path, opening_sha256, "SAbDab2 test opening")
    opening = _json(opening_path, "SAbDab2 test opening")
    observed_sources = sabdab2_evaluation_source_hashes()
    state_root = Path(str(opening.get("test_state_root", "")))
    members = opening.get("members", [])
    member_seeds = tuple(
        int(member.get("seed", -1))
        for member in members
        if isinstance(member, dict)
    )
    fixed_digest_fields = (
        "selection_receipt_sha256",
        "selection_result_sha256",
        "selected_ensemble_result_sha256",
        "selected_ensemble_receipt_sha256",
        "structure_manifest_sha256",
        "contact_manifest_sha256",
        "embedding_manifest_sha256",
        "split_manifest_sha256",
        "test_partner_panel_sha256",
        "same_antigen_challenge_sha256",
    )
    if (
        opening.get("format_version") != OPENING_FORMAT
        or opening.get("eligible_to_open") is not True
        or opening.get("test_score_session_count_before") != 0
        or opening.get("test_predictions_or_scores_read_at_freeze") is not False
        or opening.get("evaluation_source_dependencies") != observed_sources
        or opening.get("test_evaluator_source_sha256")
        != observed_sources["scripts/evaluate_framecontact_sabdab2_ensemble_test.py"]
        or not state_root.is_absolute()
        or state_root.is_symlink()
        or not state_root.is_dir()
        or opening.get("score_consumption_token_name")
        != "SABDAB2_TEST_SCORE_CONSUMED.json"
        or any(
            not isinstance(opening.get(field), str)
            or len(str(opening.get(field))) != 64
            for field in fixed_digest_fields
        )
        or not isinstance(opening.get("selected_experiment"), str)
        or not opening.get("selected_experiment")
        or member_seeds != FROZEN_SEEDS
        or any(
            not isinstance(member.get("development_result_sha256"), str)
            or len(member["development_result_sha256"]) != 64
            or not isinstance(member.get("selected_checkpoint_sha256"), str)
            or len(member["selected_checkpoint_sha256"]) != 64
            for member in members
        )
        or not isinstance(opening.get("test_instance_count"), int)
        or opening.get("test_instance_count", 0) <= 0
        or opening.get("test_partner_panel_size")
        != FROZEN_TEST_PARTNER_PANEL_SIZE
        or not math.isclose(
            float(opening.get("same_antigen_maximum_epitope_jaccard", math.nan)),
            FROZEN_SAME_ANTIGEN_MAXIMUM_EPITOPE_JACCARD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or tuple(opening.get("coordinate_noise_angstrom", ()))
        != FROZEN_COORDINATE_NOISE_ANGSTROM
        or opening.get("bootstrap") != FROZEN_BOOTSTRAP
        or tuple(opening.get("required_interventions", ()))
        != FROZEN_REQUIRED_INTERVENTIONS
    ):
        raise Sabdab2FrozenTestError("SAbDab2 test opening is ineligible")
    return opening


def consume_sabdab2_score_once(
    opening: dict[str, Any],
    *,
    opening_sha256: str,
    output_root: Path,
) -> Path:
    token = Path(str(opening["test_state_root"])) / str(
        opening["score_consumption_token_name"]
    )
    payload = canonical_json_bytes(
        {
            "format_version": SCORE_CONSUMPTION_FORMAT,
            "opening_receipt_sha256": opening_sha256,
            "selected_ensemble_result_sha256": opening[
                "selected_ensemble_result_sha256"
            ],
            "requested_output_root": str(output_root.resolve()),
            "state": "frozen_test_scoring_started_fail_closed",
            "recovery_boundary": (
                "If scoring crashes, preserve this token and freeze a recovery "
                "receipt; never delete it and rerun silently."
            ),
        }
    )
    descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return token


__all__ = [
    "ARCHITECTURE_MATRIX",
    "EVALUATION_DEPENDENCY_PATHS",
    "FROZEN_BOOTSTRAP",
    "FROZEN_COORDINATE_NOISE_ANGSTROM",
    "FROZEN_REQUIRED_INTERVENTIONS",
    "FROZEN_SAME_ANTIGEN_MAXIMUM_EPITOPE_JACCARD",
    "FROZEN_TEST_PARTNER_PANEL_SIZE",
    "OPENING_FORMAT",
    "Sabdab2FrozenTestError",
    "authenticate_sabdab2_development_selection",
    "authenticate_sabdab2_test_opening",
    "consume_sabdab2_score_once",
    "sabdab2_evaluation_source_hashes",
]
