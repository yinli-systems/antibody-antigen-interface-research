#!/usr/bin/env python3
"""Replay FrameContact checkpoint selection and early stopping from live logs.

This is a read-only development monitor.  It deliberately does not inspect run
directories, checkpoints, test predictions, or external-test artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


SUMMARY_SCRIPT = Path(__file__).with_name("summarize_framecontact_live_array.py")
SPEC = importlib.util.spec_from_file_location("framecontact_live_summary", SUMMARY_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--log-prefix", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--condition-index", type=int, required=True)
    parser.add_argument("--condition-label", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SUMMARY.SEEDS))
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--recent-epoch-window", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_selection(
    records: Mapping[int, Mapping[str, Any]], *, minimum_improvement: float
) -> dict[str, Any]:
    """Exactly replay the checkpoint rule in ``train_framecontact.py``."""

    if not records:
        raise ValueError("Cannot replay selection on an empty history")
    best_margin = -math.inf
    best_contact_macro_auprc = -math.inf
    best_epoch = 0
    stale_epochs = 0
    improvement_epochs: list[int] = []
    for epoch in sorted(records):
        record = records[epoch]
        validation = record["validation"]
        margin = float(validation["selection_metric"])
        contact_macro_auprc = float(
            validation["contact"]["mean_per_complex_auprc"]
        )
        improved = margin > best_margin + minimum_improvement or (
            abs(margin - best_margin) <= minimum_improvement
            and contact_macro_auprc > best_contact_macro_auprc
        )
        if improved:
            best_margin = margin
            best_contact_macro_auprc = contact_macro_auprc
            best_epoch = epoch
            stale_epochs = 0
            improvement_epochs.append(epoch)
        else:
            stale_epochs += 1
    return {
        "best_epoch": best_epoch,
        "best_validation_worst_directional_margin": best_margin,
        "best_contact_macro_auprc_tie_break": best_contact_macro_auprc,
        "stale_epochs": stale_epochs,
        "improvement_epochs": improvement_epochs,
    }


def _recent_seconds_per_epoch(
    records: Mapping[int, Mapping[str, Any]], *, window: int
) -> float | None:
    epochs = sorted(records)
    if len(epochs) < 2:
        return None
    deltas = [
        float(records[right]["elapsed_seconds"])
        - float(records[left]["elapsed_seconds"])
        for left, right in zip(epochs, epochs[1:])
    ]
    positive = [value for value in deltas[-window:] if value > 0]
    return statistics.fmean(positive) if positive else None


def audit_live_patience(
    *,
    log_root: Path,
    log_prefix: str,
    job_id: str,
    condition_index: int,
    condition_label: str,
    seeds: list[int],
    patience: int,
    minimum_improvement: float = 1e-4,
    recent_epoch_window: int = 5,
) -> dict[str, Any]:
    if condition_index < 0:
        raise ValueError("condition_index must be nonnegative")
    if not condition_label:
        raise ValueError("condition_label must be nonempty")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if patience < 1:
        raise ValueError("patience must be positive")
    if minimum_improvement < 0:
        raise ValueError("minimum_improvement must be nonnegative")
    if recent_epoch_window < 1:
        raise ValueError("recent_epoch_window must be positive")

    states: dict[str, Any] = {}
    for seed_index, seed in enumerate(seeds):
        task_index = condition_index * len(seeds) + seed_index
        path = log_root / f"{log_prefix}-{job_id}_{task_index}.out"
        records = SUMMARY.read_epoch_records(path)
        if not records:
            states[str(seed)] = {
                "status": "not_started_or_no_epoch_record",
                "task_index": task_index,
                "log_path": str(path),
            }
            continue
        selection = replay_selection(
            records, minimum_improvement=minimum_improvement
        )
        latest_epoch = max(records)
        remaining = max(0, patience - int(selection["stale_epochs"]))
        seconds_per_epoch = _recent_seconds_per_epoch(
            records, window=recent_epoch_window
        )
        conditional_hours = (
            None
            if seconds_per_epoch is None
            else remaining * seconds_per_epoch / 3600.0
        )
        states[str(seed)] = {
            "status": (
                "early_stopping_condition_reached_in_log"
                if selection["stale_epochs"] >= patience
                else "running_or_between_epochs"
            ),
            "task_index": task_index,
            "log_path": str(path),
            "log_sha256_at_audit": _sha256(path),
            "epoch_record_count": len(records),
            "latest_epoch": latest_epoch,
            "latest_elapsed_hours": float(records[latest_epoch]["elapsed_seconds"])
            / 3600.0,
            **selection,
            "patience": patience,
            "patience_remaining_if_no_new_best": remaining,
            "recent_mean_seconds_per_epoch": seconds_per_epoch,
            "conditional_hours_to_early_stop_if_no_new_best": conditional_hours,
        }

    return {
        "format_version": "framecontact_live_patience_audit_v1",
        "claim_boundary": (
            "Read-only replay of in-flight development train/validation logs. "
            "The conditional timing assumes no future validation improvement; "
            "it is not a completion guarantee. No checkpoint, test prediction, "
            "or external-test artifact is loaded."
        ),
        "selection_rule": {
            "primary": "maximize minimum_direction_cluster_weighted_margin",
            "tie_break": "maximize contact_mean_per_complex_auprc",
            "minimum_improvement": minimum_improvement,
        },
        "job_id": str(job_id),
        "log_prefix": log_prefix,
        "condition_index": condition_index,
        "condition_label": condition_label,
        "seeds": seeds,
        "states_by_seed": states,
    }


def main() -> None:
    arguments = _arguments()
    result = audit_live_patience(
        log_root=arguments.log_root,
        log_prefix=arguments.log_prefix,
        job_id=arguments.job_id,
        condition_index=arguments.condition_index,
        condition_label=arguments.condition_label,
        seeds=arguments.seeds,
        patience=arguments.patience,
        minimum_improvement=arguments.minimum_improvement,
        recent_epoch_window=arguments.recent_epoch_window,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
