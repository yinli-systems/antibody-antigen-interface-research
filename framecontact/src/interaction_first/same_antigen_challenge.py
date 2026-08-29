"""Exact-sequence same-antigen/different-antibody counterfactual evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT = (
    "framecontact_label_free_same_antigen_challenge_v1"
)


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def build_label_free_same_antigen_challenge_entries(
    records: Sequence[Mapping[str, Any]],
    *,
    max_alternates_per_focal: int = 31,
    seed: int = 20_260_730,
) -> list[dict[str, Any]]:
    """Freeze exact-antigen antibody swaps without reading task labels.

    Every candidate has the same exact antigen residue-axis sequence as the
    focal complex but a distinct antibody sequence and antibody cluster.  The
    deterministic cap is based only on identifiers, so target-epitope labels
    can remain sealed until scoring.
    """
    if max_alternates_per_focal <= 0:
        raise ValueError("max_alternates_per_focal must be positive")
    required = {
        "instance_id",
        "antigen_sequence_sha256",
        "antibody_sequence_sha256",
        "antibody_cluster_id",
    }
    normalized: dict[str, dict[str, str]] = {}
    for raw in records:
        if not required.issubset(raw):
            raise ValueError("Same-antigen records are missing required metadata")
        row = {name: str(raw[name]).strip() for name in required}
        if not all(row.values()):
            raise ValueError("Same-antigen metadata fields must be nonempty")
        identifier = row["instance_id"]
        if identifier in normalized:
            raise ValueError("Records contain duplicate instance identifiers")
        normalized[identifier] = row

    groups: dict[str, list[str]] = defaultdict(list)
    for identifier, row in normalized.items():
        groups[row["antigen_sequence_sha256"]].append(identifier)

    entries: list[dict[str, Any]] = []
    for antigen_sha, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for focal_id in sorted(members):
            focal = normalized[focal_id]
            candidates = []
            for alternate_id in members:
                if alternate_id == focal_id:
                    continue
                alternate = normalized[alternate_id]
                if (
                    alternate["antibody_sequence_sha256"]
                    == focal["antibody_sequence_sha256"]
                    or alternate["antibody_cluster_id"]
                    == focal["antibody_cluster_id"]
                ):
                    continue
                tie_break = hashlib.sha256(
                    f"{seed}\0{focal_id}\0{alternate_id}".encode("utf-8")
                ).hexdigest()
                candidates.append((tie_break, alternate_id))
            selected = sorted(candidates)[:max_alternates_per_focal]
            if selected:
                entries.append(
                    {
                        "antigen_sequence_sha256": antigen_sha,
                        "focal_instance_id": focal_id,
                        "focal_antibody_cluster_id": focal["antibody_cluster_id"],
                        "alternates": [
                            {
                                "instance_id": alternate_id,
                                "antibody_cluster_id": normalized[alternate_id][
                                    "antibody_cluster_id"
                                ],
                                "selection_sha256": tie_break,
                            }
                            for tie_break, alternate_id in selected
                        ],
                    }
                )
    return entries


def build_same_antigen_challenge_entries(
    records: Sequence[Mapping[str, Any]],
    *,
    epitope_indices_by_id: Mapping[str, frozenset[int]],
    antibody_cluster_by_id: Mapping[str, str],
    max_alternates_per_focal: int = 31,
    maximum_epitope_jaccard: float = 0.50,
    seed: int = 20_260_730,
) -> list[dict[str, Any]]:
    """Freeze ordered counterfactuals without balancing by family size.

    Aggregation is performed by exact-antigen group downstream so large antigen
    families cannot dominate. Candidate ordering is a deterministic hash, not
    a task metric.
    """
    if max_alternates_per_focal <= 0:
        raise ValueError("max_alternates_per_focal must be positive")
    if not math.isfinite(maximum_epitope_jaccard) or not 0 <= maximum_epitope_jaccard < 1:
        raise ValueError("maximum_epitope_jaccard must lie in [0, 1)")
    record_by_id = {str(row["instance_id"]): row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("Records contain duplicate instance identifiers")
    identifiers = set(record_by_id)
    if set(epitope_indices_by_id) != identifiers or set(antibody_cluster_by_id) != identifiers:
        raise ValueError("Labels, cluster metadata, and records must align exactly")
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier, row in record_by_id.items():
        groups[str(row["antigen_sequence_sha256"])].append(identifier)

    entries: list[dict[str, Any]] = []
    for antigen_sha, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        for focal_id in sorted(members):
            focal = record_by_id[focal_id]
            focal_labels = epitope_indices_by_id[focal_id]
            candidates = []
            for alternate_id in members:
                if alternate_id == focal_id:
                    continue
                alternate = record_by_id[alternate_id]
                if (
                    str(alternate["antibody_sequence_sha256"])
                    == str(focal["antibody_sequence_sha256"])
                    or antibody_cluster_by_id[alternate_id]
                    == antibody_cluster_by_id[focal_id]
                ):
                    continue
                overlap = _jaccard(focal_labels, epitope_indices_by_id[alternate_id])
                if overlap > maximum_epitope_jaccard:
                    continue
                tie_break = hashlib.sha256(
                    f"{seed}\0{focal_id}\0{alternate_id}".encode("utf-8")
                ).hexdigest()
                candidates.append((tie_break, alternate_id, overlap))
            selected = sorted(candidates)[:max_alternates_per_focal]
            if selected:
                entries.append(
                    {
                        "antigen_sequence_sha256": antigen_sha,
                        "focal_instance_id": focal_id,
                        "focal_antibody_cluster_id": antibody_cluster_by_id[focal_id],
                        "alternates": [
                            {
                                "instance_id": alternate_id,
                                "antibody_cluster_id": antibody_cluster_by_id[
                                    alternate_id
                                ],
                                "epitope_jaccard_to_focal": overlap,
                                "selection_sha256": tie_break,
                            }
                            for tie_break, alternate_id, overlap in selected
                        ],
                    }
                )
    return entries


def score_same_antigen_pair(
    cognate_epitope_probabilities: Any,
    swapped_epitope_probabilities: Any,
    focal_epitope_labels: Any,
    alternate_epitope_labels: Any,
) -> dict[str, Any]:
    """Score whether an antibody swap moves a fixed-antigen epitope map correctly."""
    from sklearn.metrics import average_precision_score

    cognate = np.asarray(cognate_epitope_probabilities, dtype=np.float64).reshape(-1)
    swapped = np.asarray(swapped_epitope_probabilities, dtype=np.float64).reshape(-1)
    focal = np.asarray(focal_epitope_labels).reshape(-1)
    alternate = np.asarray(alternate_epitope_labels).reshape(-1)
    if not (cognate.shape == swapped.shape == focal.shape == alternate.shape):
        raise ValueError("Same-antigen probabilities and labels must share one axis")
    if cognate.size == 0:
        raise ValueError("Same-antigen residue axis cannot be empty")
    if not np.isfinite(cognate).all() or not np.isfinite(swapped).all():
        raise ValueError("Same-antigen probabilities must be finite")
    if (
        ((cognate < 0) | (cognate > 1)).any()
        or ((swapped < 0) | (swapped > 1)).any()
    ):
        raise ValueError("Same-antigen probabilities must lie in [0, 1]")
    if not np.isin(focal, [0, 1]).all() or not np.isin(alternate, [0, 1]).all():
        raise ValueError("Same-antigen labels must be binary")
    focal = focal.astype(np.int8, copy=False)
    alternate = alternate.astype(np.int8, copy=False)
    focal_positive = int(focal.sum())
    alternate_positive = int(alternate.sum())
    overlap = _jaccard(
        frozenset(np.flatnonzero(focal).tolist()),
        frozenset(np.flatnonzero(alternate).tolist()),
    )
    base = {
        "epitope_jaccard": overlap,
        "focal_positive_count": focal_positive,
        "alternate_positive_count": alternate_positive,
        "residue_count": int(focal.size),
        "epitope_map_mean_absolute_change": float(np.mean(np.abs(swapped - cognate))),
    }
    if focal_positive == 0 or alternate_positive == 0:
        return {
            **base,
            "eligible": False,
            "ineligible_reason": "zero_positive_focal_or_alternate_epitope",
        }

    ap_cognate_focal = float(average_precision_score(focal, cognate))
    ap_cognate_alternate = float(average_precision_score(alternate, cognate))
    ap_swapped_focal = float(average_precision_score(focal, swapped))
    ap_swapped_alternate = float(average_precision_score(alternate, swapped))
    cognate_assignment_margin = ap_cognate_focal - ap_cognate_alternate
    swapped_assignment_margin = ap_swapped_alternate - ap_swapped_focal
    alternate_k = alternate_positive

    def topk_recall(probabilities: np.ndarray) -> float:
        order = np.argsort(-probabilities, kind="stable")[:alternate_k]
        return float(alternate[order].sum() / alternate_positive)

    return {
        **base,
        "eligible": True,
        "ineligible_reason": None,
        "cognate_focal_auprc": ap_cognate_focal,
        "cognate_alternate_auprc": ap_cognate_alternate,
        "swapped_focal_auprc": ap_swapped_focal,
        "swapped_alternate_auprc": ap_swapped_alternate,
        "alternate_target_auprc_gain": ap_swapped_alternate - ap_cognate_alternate,
        "focal_target_auprc_change": ap_swapped_focal - ap_cognate_focal,
        "cognate_assignment_margin": cognate_assignment_margin,
        "swapped_assignment_margin": swapped_assignment_margin,
        "directed_assignment_margin": (
            cognate_assignment_margin + swapped_assignment_margin
        )
        / 2.0,
        "target_preference_shift": (
            cognate_assignment_margin + swapped_assignment_margin
        ),
        "alternate_target_topk_definition": "K=alternate positive epitope count",
        "alternate_target_topk_recall_gain": (
            topk_recall(swapped) - topk_recall(cognate)
        ),
    }


def aggregate_same_antigen_pair_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate ordered pair metrics with one vote per exact-antigen group."""
    if not rows:
        raise ValueError("Same-antigen challenge rows are empty")
    required_metric_names = (
        "alternate_target_auprc_gain",
        "target_preference_shift",
        "alternate_target_topk_recall_gain",
        "epitope_map_mean_absolute_change",
    )
    optional_metric_names = (
        "cognate_assignment_margin",
        "swapped_assignment_margin",
        "directed_assignment_margin",
        "focal_target_auprc_change",
    )
    optional_presence = {
        name: [name in row for row in rows] for name in optional_metric_names
    }
    if any(any(values) and not all(values) for values in optional_presence.values()):
        raise ValueError("Optional same-antigen metrics must be present in every row")
    metric_names = required_metric_names + tuple(
        name for name, values in optional_presence.items() if all(values)
    )
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["antigen_sequence_sha256"])].append(row)
        for name in metric_names:
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Non-finite same-antigen metric: {name}")
    group_means = {
        group: {
            name: sum(float(row[name]) for row in local) / len(local)
            for name in metric_names
        }
        for group, local in by_group.items()
    }
    result = {
        "exact_antigen_group_count": len(group_means),
        "ordered_pair_count": len(rows),
        "group_weighted": {
            name: sum(values[name] for values in group_means.values()) / len(group_means)
            for name in metric_names
        },
        "ordered_pair_success_rate": sum(
            float(row["alternate_target_auprc_gain"]) > 0 for row in rows
        )
        / len(rows),
        "group_means": group_means,
    }
    if "cognate_assignment_margin" in metric_names:
        result["ordered_pair_directed_assignment_success_rate"] = sum(
            float(row["cognate_assignment_margin"]) > 0
            and float(row["swapped_assignment_margin"]) > 0
            for row in rows
        ) / len(rows)
    return result


def bootstrap_same_antigen_group_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_730,
) -> dict[str, Any]:
    """Bootstrap exact-antigen groups, retaining all ordered pairs per draw."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be positive")
    aggregate = aggregate_same_antigen_pair_metrics(rows)
    group_means = aggregate["group_means"]
    if len(group_means) < 2:
        raise ValueError("At least two exact-antigen groups are required")
    metric_names = tuple(next(iter(group_means.values())))
    matrix = np.asarray(
        [[group_means[group][name] for name in metric_names] for group in sorted(group_means)],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(matrix), size=(replicates, len(matrix)))
    samples = matrix[indices].mean(axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "exact_antigen_group_count": len(matrix),
        "metrics": {
            name: {
                "estimate": float(matrix[:, index].mean()),
                "ci95_lower": float(np.quantile(samples[:, index], 0.025)),
                "ci95_upper": float(np.quantile(samples[:, index], 0.975)),
            }
            for index, name in enumerate(metric_names)
        },
    }


__all__ = [
    "LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT",
    "aggregate_same_antigen_pair_metrics",
    "bootstrap_same_antigen_group_metrics",
    "build_label_free_same_antigen_challenge_entries",
    "build_same_antigen_challenge_entries",
    "score_same_antigen_pair",
]
