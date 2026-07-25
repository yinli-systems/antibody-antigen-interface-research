"""Paired diagnostics for hard-negative and partner-permutation controls."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _numpy() -> Any:
    """Import NumPy only when a control diagnostic is requested."""
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError("NumPy is required for partner-control diagnostics") from error
    return np


def _matched_array(scores: Sequence[float]) -> Any:
    """Validate one matched-partner score per evaluated instance."""
    np = _numpy()
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("matched_scores must be a non-empty one-dimensional sequence")
    if not bool(np.isfinite(values).all()):
        raise ValueError("matched_scores must contain only finite values")
    return values


def _control_array(scores: Any, instance_count: int, name: str) -> Any:
    """Normalize one control family to shape [instances, candidates]."""
    np = _numpy()
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or values.shape[0] != instance_count
        or values.shape[1] == 0
    ):
        raise ValueError(
            f"Control {name!r} must have shape [instances] or "
            "[instances, candidates]"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"Control {name!r} must contain only finite values")
    return values


def _single_control_metrics(matched: Any, controls: Any) -> dict[str, Any]:
    """Summarize conservative retrieval ranks and paired score drops."""
    np = _numpy()
    hardest = controls.max(axis=1)
    drops = matched - hardest
    ranks = 1 + (controls >= matched[:, None]).sum(axis=1)
    standard_deviation = (
        float(drops.std(ddof=1)) if int(drops.size) > 1 else 0.0
    )
    standardized_drop = (
        float(drops.mean() / standard_deviation)
        if standard_deviation > 0.0
        else None
    )
    result: dict[str, Any] = {
        "instance_count": int(matched.size),
        "control_candidates_per_instance": int(controls.shape[1]),
        "recall_at_1": float((ranks == 1).mean()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "matched_strict_win_rate": float((drops > 0.0).mean()),
        "matched_tie_rate": float((drops == 0.0).mean()),
        "mean_matched_score": float(matched.mean()),
        "mean_hardest_control_score": float(hardest.mean()),
        "mean_matched_minus_hardest_control": float(drops.mean()),
        "median_matched_minus_hardest_control": float(np.median(drops)),
        "paired_standardized_mean_drop": standardized_drop,
    }
    for cutoff in (5, 10):
        if cutoff <= int(controls.shape[1]) + 1:
            result[f"recall_at_{cutoff}"] = float((ranks <= cutoff).mean())
    return result


def partner_specificity_control_metrics(
    matched_scores: Sequence[float],
    controls_by_name: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate matched partners against named hard-negative control families.

    Scores must use the convention that larger values indicate greater partner
    compatibility. Each control family may provide one score per instance or a
    fixed-width candidate matrix. A tie is conservatively counted as outranking
    the matched partner. The ``all_controls`` result pools every named family
    and therefore measures the strictest available control set.
    """
    np = _numpy()
    matched = _matched_array(matched_scores)
    if not controls_by_name:
        raise ValueError("At least one named partner-control family is required")

    normalized: dict[str, Any] = {}
    for raw_name, scores in controls_by_name.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Partner-control names cannot be empty")
        if name in normalized:
            raise ValueError(f"Duplicate normalized partner-control name: {name!r}")
        normalized[name] = _control_array(scores, int(matched.size), name)

    by_control = {
        name: _single_control_metrics(matched, normalized[name])
        for name in sorted(normalized)
    }
    pooled = np.concatenate(
        [normalized[name] for name in sorted(normalized)],
        axis=1,
    )
    return {
        "instance_count": int(matched.size),
        "control_family_count": len(normalized),
        "control_names": sorted(normalized),
        "by_control": by_control,
        "all_controls": _single_control_metrics(matched, pooled),
    }


def validate_partner_score_support(
    matched_candidate_counts: Sequence[int],
    control_candidate_counts: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Detect incomparable score supports before interpreting control deltas.

    Length-normalized pooling reduces support-size bias but does not prove that
    scores from materially different candidate masks are exchangeable. This
    diagnostic reports exact support parity and relative count gaps so callers
    can fail closed or stratify their analysis.
    """
    np = _numpy()
    matched = np.asarray(matched_candidate_counts)
    if (
        matched.ndim != 1
        or matched.size == 0
        or not bool(np.issubdtype(matched.dtype, np.integer))
        or bool((matched <= 0).any())
    ):
        raise ValueError("matched_candidate_counts must contain positive integers")
    if not control_candidate_counts:
        raise ValueError("At least one control candidate-count family is required")

    by_control: dict[str, Any] = {}
    all_equal = True
    for raw_name, raw_counts in control_candidate_counts.items():
        name = str(raw_name).strip()
        if not name or name in by_control:
            raise ValueError("Control candidate-count names must be unique and non-empty")
        counts = np.asarray(raw_counts)
        if (
            counts.ndim != 1
            or counts.shape != matched.shape
            or not bool(np.issubdtype(counts.dtype, np.integer))
            or bool((counts <= 0).any())
        ):
            raise ValueError(
                f"Control {name!r} candidate counts must be aligned positive integers"
            )
        equal = counts == matched
        relative_gap = np.abs(counts - matched) / np.maximum(counts, matched)
        family_equal = bool(equal.all())
        all_equal = all_equal and family_equal
        by_control[name] = {
            "exact_support_match": family_equal,
            "matched_instance_fraction": float(equal.mean()),
            "mean_relative_count_gap": float(relative_gap.mean()),
            "maximum_relative_count_gap": float(relative_gap.max()),
        }
    return {
        "instance_count": int(matched.size),
        "all_controls_exact_support_match": all_equal,
        "by_control": {name: by_control[name] for name in sorted(by_control)},
    }


__all__ = [
    "partner_specificity_control_metrics",
    "validate_partner_score_support",
]
