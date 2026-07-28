"""Development-only contact and partner evaluation for PAIRScale E1/E2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .development_contact_dataset import DevelopmentContactExample
from .metrics import average_precision, contact_map_metrics, partner_ranking_metrics
from .pairformer_evaluator import evaluate_bidirectional_cognate_retrieval
from .pairscale_e1_e2_step import _forward_cognate, _forward_unknown_pair
from .pairscale_partner_specificity import (
    raw_specificity_partner_score,
    require_specificity_partner_score,
    select_score_free_bidirectional_decoys,
)


EVALUATION_FORMAT = "pairscale_e1_e2_development_evaluation_v2"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return float(sum(values) / len(values))


def _panel_summary(
    matched: Sequence[float],
    decoys: Sequence[Sequence[float]],
) -> Mapping[str, Any]:
    if len(matched) != len(decoys) or not matched:
        raise ValueError("Matched and decoy scores must be aligned and nonempty")
    if any(not row for row in decoys):
        raise ValueError("Every matched score must have at least one decoy")
    deltas = [
        float(matched_score - decoy_score)
        for matched_score, row in zip(matched, decoys, strict=True)
        for decoy_score in row
    ]
    return {
        "ranking": partner_ranking_metrics(matched, decoys),
        "pairwise_comparison_count": len(deltas),
        "pairwise_strict_win_rate": _mean(
            [float(value > 0.0) for value in deltas]
        ),
        "pairwise_half_tie_win_rate": _mean(
            [float(value > 0.0) + 0.5 * float(value == 0.0) for value in deltas]
        ),
        "matched_minus_decoy_mean": _mean(deltas),
        "matched_scores": list(matched),
        "decoy_scores": [list(row) for row in decoys],
    }


def _canonical_endpoint_representatives(
    examples: Sequence[DevelopmentContactExample],
) -> tuple[
    tuple[str, ...],
    tuple[DevelopmentContactExample, ...],
    tuple[str, ...],
    tuple[DevelopmentContactExample, ...],
]:
    antibody: dict[str, DevelopmentContactExample] = {}
    antigen: dict[str, DevelopmentContactExample] = {}
    for example in sorted(examples, key=lambda item: item.instance_id):
        antibody.setdefault(example.antibody_id, example)
        antigen.setdefault(example.antigen_id, example)
    antibody_ids = tuple(sorted(antibody))
    antigen_ids = tuple(sorted(antigen))
    if len(antibody_ids) < 2 or len(antigen_ids) < 2:
        raise ValueError("Exhaustive retrieval requires at least two unique endpoints")
    return (
        antibody_ids,
        tuple(antibody[item] for item in antibody_ids),
        antigen_ids,
        tuple(antigen[item] for item in antigen_ids),
    )


def _exhaustive_retrieval(
    *,
    model: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
) -> Mapping[str, Any]:
    import numpy as np

    (
        antibody_ids,
        antibody_examples,
        antigen_ids,
        antigen_examples,
    ) = _canonical_endpoint_representatives(examples)
    centered = np.empty((len(antibody_ids), len(antigen_ids)), dtype=np.float64)
    raw = np.empty_like(centered)
    for antibody_index, antibody_example in enumerate(antibody_examples):
        for antigen_index, antigen_example in enumerate(antigen_examples):
            output = _forward_unknown_pair(
                model,
                antibody_example,
                antigen_example,
                device,
            )
            centered[antibody_index, antigen_index] = float(
                require_specificity_partner_score(output)[0].detach().cpu()
            )
            raw[antibody_index, antigen_index] = float(
                raw_specificity_partner_score(output)[0].detach().cpu()
            )
    antibody_index = {identifier: index for index, identifier in enumerate(antibody_ids)}
    antigen_index = {identifier: index for index, identifier in enumerate(antigen_ids)}
    cognate = np.zeros_like(centered, dtype=np.bool_)
    for example in examples:
        cognate[
            antibody_index[example.antibody_id],
            antigen_index[example.antigen_id],
        ] = True
    candidate = np.ones_like(cognate, dtype=np.bool_)
    return {
        "antibody_ids": list(antibody_ids),
        "antigen_ids": list(antigen_ids),
        "observed_cognate_pair_count": int(cognate.sum()),
        "candidate_pair_count": int(candidate.sum()),
        "unknown_cross_pair_semantics": "unlabeled_retrieval_control_not_nonbinder",
        "centered_residual": evaluate_bidirectional_cognate_retrieval(
            centered,
            cognate,
            candidate,
        ),
        "raw_residual": evaluate_bidirectional_cognate_retrieval(
            raw,
            cognate,
            candidate,
        ),
    }


def evaluate_e1_e2(
    *,
    model: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    decoy_count: int,
    seed: int,
    include_partner_panel: bool,
    include_exhaustive_retrieval: bool,
) -> Mapping[str, Any]:
    """Evaluate fixed outputs without selecting a checkpoint or threshold."""
    import numpy as np
    import torch

    if not examples:
        raise ValueError("Evaluation requires at least one complex")
    if not isinstance(include_partner_panel, bool) or not isinstance(
        include_exhaustive_retrieval,
        bool,
    ):
        raise ValueError("Evaluation flags must be boolean")
    model.eval()
    per_complex: list[dict[str, Any]] = []
    pooled_labels: list[Any] = []
    pooled_probabilities: list[Any] = []
    matched_centered: list[float] = []
    matched_raw: list[float] = []
    with torch.inference_mode():
        for example in examples:
            outputs, targets, supervision_mask = _forward_cognate(
                model,
                example,
                device,
            )
            mask = supervision_mask[0].cpu().numpy()
            labels = targets[0].cpu().numpy()[mask]
            probabilities = (
                outputs["contact_probabilities"][0].float().cpu().numpy()[mask]
            )
            metrics = contact_map_metrics(labels, probabilities)
            prevalence = float(np.mean(labels))
            metrics["contact_prevalence"] = prevalence
            metrics["ap_lift"] = float(
                (float(metrics["auprc"]) - prevalence) / (1.0 - prevalence)
            )
            metrics["ap_over_prevalence"] = float(metrics["auprc"]) / prevalence
            unary_probabilities = torch.sigmoid(
                outputs["antibody_unary_logits"].unsqueeze(2)
                + outputs["antigen_unary_logits"].unsqueeze(1)
            )[0].float().cpu().numpy()[mask]
            residual_probabilities = torch.sigmoid(
                outputs["contact_interaction_residual_logits"]
            )[0].float().cpu().numpy()[mask]
            unary_metrics = contact_map_metrics(
                labels,
                unary_probabilities,
            )
            residual_metrics = contact_map_metrics(
                labels,
                residual_probabilities,
            )
            metrics["unary_additive_auprc"] = unary_metrics["auprc"]
            metrics["unary_additive_precision_at_true_contact_count"] = (
                unary_metrics["precision_at_true_contact_count"]
            )
            metrics[
                "unary_additive_precision_at_true_contact_count_tie_policy"
            ] = unary_metrics["precision_at_true_contact_count_tie_policy"]
            metrics["unary_additive_ap_lift"] = float(
                (float(unary_metrics["auprc"]) - prevalence) / (1.0 - prevalence)
            )
            metrics["interaction_residual_auprc"] = residual_metrics["auprc"]
            metrics["interaction_residual_precision_at_true_contact_count"] = (
                residual_metrics["precision_at_true_contact_count"]
            )
            metrics[
                "interaction_residual_precision_at_true_contact_count_tie_policy"
            ] = residual_metrics["precision_at_true_contact_count_tie_policy"]
            metrics["interaction_residual_ap_lift"] = float(
                (float(residual_metrics["auprc"]) - prevalence)
                / (1.0 - prevalence)
            )
            metrics["instance_id"] = example.instance_id
            metrics["antibody_id"] = example.antibody_id
            metrics["antigen_id"] = example.antigen_id
            per_complex.append(metrics)
            pooled_labels.append(labels.astype(np.uint8, copy=False))
            pooled_probabilities.append(
                probabilities.astype(np.float32, copy=False)
            )
            matched_centered.append(
                float(require_specificity_partner_score(outputs)[0].cpu())
            )
            matched_raw.append(
                float(raw_specificity_partner_score(outputs)[0].cpu())
            )

        partner_controls = None
        if include_partner_panel:
            panel = select_score_free_bidirectional_decoys(
                examples,
                count=decoy_count,
                seed=seed,
            )
            directional: dict[str, dict[str, list[list[float]]]] = {
                "antibody_to_antigen": {"centered": [], "raw": []},
                "antigen_to_antibody": {"centered": [], "raw": []},
            }
            for index, example in enumerate(examples):
                for direction, indices in (
                    (
                        "antibody_to_antigen",
                        panel.antibody_to_antigen[index],
                    ),
                    (
                        "antigen_to_antibody",
                        panel.antigen_to_antibody[index],
                    ),
                ):
                    centered_row: list[float] = []
                    raw_row: list[float] = []
                    for decoy_index in indices:
                        if direction == "antibody_to_antigen":
                            antibody_example = example
                            antigen_example = examples[decoy_index]
                        else:
                            antibody_example = examples[decoy_index]
                            antigen_example = example
                        output = _forward_unknown_pair(
                            model,
                            antibody_example,
                            antigen_example,
                            device,
                        )
                        centered_row.append(
                            float(require_specificity_partner_score(output)[0].cpu())
                        )
                        raw_row.append(
                            float(raw_specificity_partner_score(output)[0].cpu())
                        )
                    directional[direction]["centered"].append(centered_row)
                    directional[direction]["raw"].append(raw_row)
            partner_controls = {
                "decoy_count_per_direction": decoy_count,
                "unknown_cross_pair_semantics": (
                    "unlabeled_ranking_control_not_nonbinder"
                ),
            }
            for direction, values in directional.items():
                partner_controls[direction] = {
                    "centered_residual": _panel_summary(
                        matched_centered,
                        values["centered"],
                    ),
                    "raw_residual": _panel_summary(
                        matched_raw,
                        values["raw"],
                    ),
                }

        exhaustive = None
        if include_exhaustive_retrieval:
            exhaustive = _exhaustive_retrieval(
                model=model,
                examples=examples,
                device=device,
            )

    macro_auprc = _mean([float(item["auprc"]) for item in per_complex])
    return {
        "format_version": EVALUATION_FORMAT,
        "model_selection_authorized": False,
        "threshold_selection_performed": False,
        "unknown_cross_pairs_assigned_binary_labels": False,
        "complex_count": len(examples),
        "macro_pair_auprc": macro_auprc,
        "pooled_pair_auprc": average_precision(
            np.concatenate(pooled_labels),
            np.concatenate(pooled_probabilities),
        ),
        "macro_precision_at_true_contact_count": _mean(
            [float(item["precision_at_true_contact_count"]) for item in per_complex]
        ),
        "macro_ap_lift": _mean([float(item["ap_lift"]) for item in per_complex]),
        "macro_unary_additive_auprc": _mean(
            [float(item["unary_additive_auprc"]) for item in per_complex]
        ),
        "macro_unary_additive_precision_at_true_contact_count": _mean(
            [
                float(item["unary_additive_precision_at_true_contact_count"])
                for item in per_complex
            ]
        ),
        "macro_unary_additive_ap_lift": _mean(
            [float(item["unary_additive_ap_lift"]) for item in per_complex]
        ),
        "macro_interaction_residual_auprc": _mean(
            [float(item["interaction_residual_auprc"]) for item in per_complex]
        ),
        "macro_interaction_residual_precision_at_true_contact_count": _mean(
            [
                float(
                    item[
                        "interaction_residual_precision_at_true_contact_count"
                    ]
                )
                for item in per_complex
            ]
        ),
        "macro_interaction_residual_ap_lift": _mean(
            [float(item["interaction_residual_ap_lift"]) for item in per_complex]
        ),
        "partner_controls": partner_controls,
        "exhaustive_bidirectional_retrieval": exhaustive,
        "per_complex": per_complex,
    }


__all__ = ["EVALUATION_FORMAT", "evaluate_e1_e2"]
