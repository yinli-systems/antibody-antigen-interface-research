#!/usr/bin/env python3
"""One-time receipt-gated scoring of frozen FrameContact prediction arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.framecontact_metrics import (  # noqa: E402
    binary_classification_metrics,
    bootstrap_cluster_weighted_margin,
    bootstrap_group_weighted_values,
    partner_retrieval_metrics,
    positive_required_average_precision,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


OPENING_FORMAT_V1 = "framecontact_external_opening_receipt_v1"
OPENING_FORMAT_V2 = "framecontact_external_opening_receipt_v2"
RESULT_FORMAT_V1 = "framecontact_frozen_external_result_v1"
RESULT_FORMAT_V2 = "framecontact_frozen_external_result_v2"
BASE_PREDICTION_KEYS = {
    "format_version",
    "instance_ids",
    "test_partner_panel_sha256",
    "contact_offsets",
    "contact_probabilities",
    "paratope_offsets",
    "paratope_probabilities",
    "epitope_offsets",
    "epitope_probabilities",
    "correct_partner_scores",
}
BASE_TRUTH_KEYS = {
    "format_version",
    "instance_ids",
    "antigen_cluster_ids",
    "contact_offsets",
    "contact_labels",
    "paratope_offsets",
    "paratope_labels",
    "epitope_offsets",
    "epitope_labels",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--opening-receipt-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--predictions-sha256", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_npz(path: Path, required: set[str], format_version: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"{path.name} schema differs")
        values = {key: archive[key].copy() for key in archive.files}
    if str(values["format_version"].item()) != format_version:
        raise ValueError(f"{path.name} format version differs")
    return values


def _validate_offsets(offsets: np.ndarray, value_count: int, instance_count: int, name: str) -> None:
    if (
        offsets.dtype.kind not in {"i", "u"}
        or offsets.shape != (instance_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != value_count
        or bool((np.diff(offsets) <= 0).any())
    ):
        raise ValueError(f"{name} offsets are invalid")


def _task_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
    *,
    instance_ids: np.ndarray,
    group_ids: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    absolute = binary_classification_metrics(
        labels,
        probabilities,
        threshold=threshold,
        include_calibration_regression=True,
    )
    if instance_ids.shape != group_ids.shape or instance_ids.shape != (
        len(offsets) - 1,
    ):
        raise ValueError("Task instance/group axes differ from offsets")
    eligible: dict[str, float] = {}
    ineligible: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    for identifier, group, start, stop in zip(
        instance_ids.astype(str),
        group_ids.astype(str),
        offsets[:-1],
        offsets[1:],
    ):
        if not identifier or not group:
            raise ValueError("Task instance and group identifiers must be nonempty")
        record = positive_required_average_precision(
            labels[int(start) : int(stop)],
            probabilities[int(start) : int(stop)],
        )
        groups[identifier] = group
        if record["eligible"]:
            eligible[identifier] = float(record["value"])
        else:
            ineligible.append(
                {
                    "instance_id": identifier,
                    "reason": record["ineligible_reason"],
                    "positive_count": record["positive_count"],
                    "candidate_count": record["candidate_count"],
                }
            )
    per_complex = np.asarray(list(eligible.values()), dtype=np.float64)
    if len(set(groups[identifier] for identifier in eligible)) >= 2:
        interval = bootstrap_group_weighted_values(
            eligible,
            groups,
            replicates=10_000,
            seed=bootstrap_seed,
        )
    else:
        interval = {
            "estimate": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "replicates": 0,
            "seed": bootstrap_seed,
            "eligible_complex_count": len(eligible),
            "group_count": len(set(groups[identifier] for identifier in eligible)),
            "reason": "fewer_than_two_eligible_groups",
        }
    return {
        **absolute,
        "mean_per_complex_auprc": (
            float(np.mean(per_complex)) if per_complex.size else None
        ),
        "median_per_complex_auprc": (
            float(np.median(per_complex)) if per_complex.size else None
        ),
        "per_complex_auprc_eligibility_policy": (
            "Complexes with zero positive labels remain in pooled metrics but are "
            "ineligible for per-complex AP and its group-macro bootstrap."
        ),
        "eligible_per_complex_auprc_count": len(eligible),
        "ineligible_per_complex_auprc_count": len(ineligible),
        "ineligible_per_complex_auprc": ineligible,
        "cluster_macro_per_complex_auprc_ci95": interval,
        "threshold_frozen_from_validation": threshold,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite external result: {arguments.output_root}")
    authenticate_file(
        arguments.opening_receipt,
        arguments.opening_receipt_sha256,
        "external opening receipt",
    )
    opening = json.loads(arguments.opening_receipt.read_text(encoding="utf-8"))
    opening_format = opening.get("format_version")
    reciprocal = opening_format == OPENING_FORMAT_V2
    if (
        opening_format not in {OPENING_FORMAT_V1, OPENING_FORMAT_V2}
        or opening.get("eligible_to_open") is not True
        or opening.get("test_labels_read_at_freeze") is not False
        or opening.get("test_open_count_before") != 0
    ):
        raise ValueError("External opening receipt is not eligible")
    if sha256_file(Path(__file__)) != opening.get("scorer_source_sha256"):
        raise ValueError("External scorer source differs from the frozen hash")
    authenticate_file(arguments.predictions, arguments.predictions_sha256, "predictions")
    observed_truth = authenticate_file(arguments.truth, arguments.truth_sha256, "truth")
    if observed_truth != opening.get("test_truth_authority_sha256"):
        raise ValueError("Truth file differs from the pre-committed external authority")
    observed_thresholds = authenticate_file(
        arguments.thresholds, arguments.thresholds_sha256, "validation thresholds"
    )
    if observed_thresholds != opening.get("thresholds_sha256"):
        raise ValueError("Threshold authority differs from the opening receipt")
    thresholds = json.loads(arguments.thresholds.read_text(encoding="utf-8"))
    if (
        thresholds.get("format_version") != "framecontact_validation_thresholds_v1"
        or thresholds.get("selected_on_validation_only") is not True
        or set(thresholds.get("thresholds", {})) != {"contact", "paratope", "epitope"}
    ):
        raise ValueError("Threshold authority is ineligible")
    prediction_keys = set(BASE_PREDICTION_KEYS)
    truth_keys = set(BASE_TRUTH_KEYS)
    if reciprocal:
        prediction_keys.update({"wrong_antigen_scores", "wrong_antibody_scores"})
        truth_keys.add("antibody_cluster_ids")
        prediction_format = "framecontact_external_predictions_v2"
        truth_format = "framecontact_external_truth_v2"
        result_format = RESULT_FORMAT_V2
        if (
            opening.get("specificity_score_source") != "interaction_only"
            or opening.get("test_partner_panel_directions") != 2
        ):
            raise ValueError("Reciprocal opening lacks interaction-only authority")
    else:
        prediction_keys.add("wrong_partner_scores")
        prediction_format = "framecontact_external_predictions_v1"
        truth_format = "framecontact_external_truth_v1"
        result_format = RESULT_FORMAT_V1
    predictions = _load_npz(arguments.predictions, prediction_keys, prediction_format)
    truth = _load_npz(arguments.truth, truth_keys, truth_format)
    instance_ids = predictions["instance_ids"].astype(str)
    truth_ids = truth["instance_ids"].astype(str)
    if (
        instance_ids.ndim != 1
        or len(set(instance_ids.tolist())) != len(instance_ids)
        or not np.array_equal(instance_ids, truth_ids)
    ):
        raise ValueError("Prediction and truth instance axes differ")
    instance_count = len(instance_ids)
    if str(predictions["test_partner_panel_sha256"].item()) != opening.get(
        "test_partner_panel_sha256"
    ):
        raise ValueError("Prediction partner panel differs from the frozen test panel")
    for task in ("contact", "paratope", "epitope"):
        probabilities = predictions[f"{task}_probabilities"].astype(np.float64)
        labels = truth[f"{task}_labels"].astype(np.int8)
        prediction_offsets = predictions[f"{task}_offsets"]
        truth_offsets = truth[f"{task}_offsets"]
        _validate_offsets(prediction_offsets, len(probabilities), instance_count, task)
        _validate_offsets(truth_offsets, len(labels), instance_count, task)
        if not np.array_equal(prediction_offsets, truth_offsets):
            raise ValueError(f"{task} prediction/truth offsets differ")
        if not np.isfinite(probabilities).all() or bool(
            ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError(f"{task} probabilities are invalid")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{task} labels are not binary")
    correct_scores = predictions["correct_partner_scores"].astype(np.float64)
    expected_wrong_shape = (
        instance_count,
        int(opening["test_partner_panel_size"]),
    )
    if correct_scores.shape != (instance_count,):
        raise ValueError("Correct-partner score axis differs from the opening receipt")
    if reciprocal:
        wrong_antigen_scores = predictions["wrong_antigen_scores"].astype(np.float64)
        wrong_antibody_scores = predictions["wrong_antibody_scores"].astype(np.float64)
        if (
            wrong_antigen_scores.shape != expected_wrong_shape
            or wrong_antibody_scores.shape != expected_wrong_shape
        ):
            raise ValueError("Reciprocal partner score axes differ from the opening receipt")
        antigen_clusters = truth["antigen_cluster_ids"].astype(str).tolist()
        antibody_clusters = truth["antibody_cluster_ids"].astype(str).tolist()
        retrieval_by_direction = {
            "fixed_antigen_wrong_antibody": partner_retrieval_metrics(
                correct_scores, wrong_antibody_scores, antigen_clusters
            ),
            "fixed_antibody_wrong_antigen": partner_retrieval_metrics(
                correct_scores, wrong_antigen_scores, antibody_clusters
            ),
        }
        intervals_by_direction = {
            direction: bootstrap_cluster_weighted_margin(
                retrieval["cluster_margins"],
                replicates=10_000,
                seed=20_260_730,
            )
            for direction, retrieval in retrieval_by_direction.items()
        }
        primary_interval = intervals_by_direction["fixed_antigen_wrong_antibody"]
        primary_name = "fixed_antigen_cognate_antibody_interaction_margin"
    else:
        wrong_scores = predictions["wrong_partner_scores"].astype(np.float64)
        if wrong_scores.shape != expected_wrong_shape:
            raise ValueError("Partner retrieval score axes differ from the opening receipt")
        cluster_ids = truth["antigen_cluster_ids"].astype(str).tolist()
        retrieval = partner_retrieval_metrics(correct_scores, wrong_scores, cluster_ids)
        primary_interval = bootstrap_cluster_weighted_margin(
            retrieval["cluster_margins"], replicates=10_000, seed=20_260_730
        )
        retrieval_by_direction = None
        intervals_by_direction = None
        primary_name = "cluster_weighted_cognate_compatibility_margin"
    result = {
        "format_version": result_format,
        "claim_boundary": (
            "One-time frozen external evaluation on the authenticated arrays. This is "
            "not clinical validation and does not imply canonical CHIMERA co-design performance."
        ),
        "test_open_count": 1,
        "authority": {
            "opening_receipt_sha256": arguments.opening_receipt_sha256,
            "predictions_sha256": arguments.predictions_sha256,
            "truth_sha256": observed_truth,
            "thresholds_sha256": observed_thresholds,
            "scorer_source_sha256": sha256_file(Path(__file__)),
        },
        "primary_endpoint": {
            "name": primary_name,
            **primary_interval,
            "success": primary_interval["ci95_lower"] > 0.0,
        },
        "partner_retrieval": None if reciprocal else retrieval,
        "partner_retrieval_by_direction": retrieval_by_direction,
        "directional_intervals": intervals_by_direction,
        "required_reciprocal_success": (
            None
            if not reciprocal
            else intervals_by_direction["fixed_antibody_wrong_antigen"][
                "ci95_lower"
            ]
            > 0.0
        ),
        "contact": _task_metrics(
            truth["contact_labels"],
            predictions["contact_probabilities"],
            truth["contact_offsets"],
            float(thresholds["thresholds"]["contact"]),
            instance_ids=instance_ids,
            group_ids=truth["antigen_cluster_ids"],
            bootstrap_seed=20_260_731,
        ),
        "paratope": _task_metrics(
            truth["paratope_labels"],
            predictions["paratope_probabilities"],
            truth["paratope_offsets"],
            float(thresholds["thresholds"]["paratope"]),
            instance_ids=instance_ids,
            group_ids=(
                truth["antibody_cluster_ids"]
                if reciprocal
                else truth["antigen_cluster_ids"]
            ),
            bootstrap_seed=20_260_732,
        ),
        "epitope": _task_metrics(
            truth["epitope_labels"],
            predictions["epitope_probabilities"],
            truth["epitope_offsets"],
            float(thresholds["thresholds"]["epitope"]),
            instance_ids=instance_ids,
            group_ids=truth["antigen_cluster_ids"],
            bootstrap_seed=20_260_733,
        ),
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary_root / "FRAMECONTACT_FROZEN_EXTERNAL_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(result))
        receipt = {
            "format_version": result_format,
            "result_sha256": sha256_file(result_path),
            "test_open_count": 1,
            "primary_success": result["primary_endpoint"]["success"],
        }
        (temporary_root / "RUN_RECEIPT.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary_root, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()
