#!/usr/bin/env python3
"""Fail closed on missing raw-Brier instrumentation while v2 runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEEDS = (17, 42, 73, 101, 137)
RAW_METRIC = "fixed_raw_logit_threshold_zero"


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def latest_epoch(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"epoch"' not in line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("epoch"), int):
                latest = candidate
    return latest


def validate_raw_metric(epoch: dict[str, Any], label: str) -> dict[str, float]:
    train = epoch["train"]
    contact = epoch["validation"]["contact"]
    raw = contact[RAW_METRIC]
    temperature = finite(raw["probability_temperature"], f"{label} temperature")
    threshold = finite(raw["probability_threshold"], f"{label} threshold")
    brier = finite(raw["brier"], f"{label} raw Brier")
    if temperature != 1.0 or threshold != 0.5:
        raise ValueError(f"{label} raw metric is not fixed at T=1, p=0.5")
    if raw.get("calibration_source") != "none":
        raise ValueError(f"{label} raw metric used calibration")
    if not 0.0 <= brier <= 1.0:
        raise ValueError(f"{label} raw Brier is outside [0, 1]")
    shared = (
        finite(train["contact_loss"], f"{label} contact loss")
        + 0.25 * finite(train["epitope_loss"], f"{label} epitope loss")
        + 0.25 * finite(train["paratope_loss"], f"{label} paratope loss")
    )
    return {
        "train_contact_bce": finite(train["contact_loss"], f"{label} contact loss"),
        "train_shared_objective": shared,
        "validation_contact_auprc": finite(contact["auprc"], f"{label} AUPRC"),
        "validation_contact_raw_brier": brier,
    }


def task_record(
    *, data_root: Path, dataset: str, job_id: str, task: int
) -> dict[str, Any]:
    seed = SEEDS[task % 5]
    arm = "with_negative" if (task % 10) < 5 else "no_negative"
    logs = data_root / "slurm-logs"
    stdout = logs / f"fc-neg-v2-{job_id}_{task}.out"
    stderr = logs / f"fc-neg-v2-{job_id}_{task}.err"
    epoch = latest_epoch(stdout)
    record: dict[str, Any] = {
        "arm": arm,
        "dataset": dataset,
        "job_id": job_id,
        "latest_epoch": 0,
        "raw_brier_instrumented": False,
        "seed": seed,
        "stderr_bytes": stderr.stat().st_size if stderr.is_file() else 0,
        "stdout_bytes": stdout.stat().st_size if stdout.is_file() else 0,
        "task": task,
    }
    if epoch is not None:
        label = f"{job_id}_{task} epoch {epoch['epoch']}"
        record.update(validate_raw_metric(epoch, label))
        record["latest_epoch"] = int(epoch["epoch"])
        record["raw_brier_instrumented"] = True
    output_root = (
        data_root
        / dataset
        / "development-runs/negative-sample-ablation-v2"
        / arm
        / "framecontact"
        / f"seed-{seed}"
    )
    record["final_result_present"] = (
        output_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    ).is_file()
    return record


def audit(*, data_root: Path, asep_job_id: str, sabdab2_job_id: str) -> dict[str, Any]:
    tasks = [
        task_record(
            data_root=data_root,
            dataset="asep",
            job_id=asep_job_id,
            task=task,
        )
        for task in range(10)
    ]
    tasks.extend(
        task_record(
            data_root=data_root,
            dataset="sabdab2",
            job_id=sabdab2_job_id,
            task=task,
        )
        for task in range(10, 20)
    )
    datasets = {}
    for dataset in ("asep", "sabdab2"):
        selected = [item for item in tasks if item["dataset"] == dataset]
        datasets[dataset] = {
            "completed_task_count": sum(item["final_result_present"] for item in selected),
            "instrumented_task_count": sum(
                item["raw_brier_instrumented"] for item in selected
            ),
            "latest_epoch_min": min(item["latest_epoch"] for item in selected),
            "latest_epoch_max": max(item["latest_epoch"] for item in selected),
            "nonempty_stderr_task_count": sum(item["stderr_bytes"] > 0 for item in selected),
            "task_count": len(selected),
        }
    all_instrumented = all(item["raw_brier_instrumented"] for item in tasks)
    all_completed = all(item["final_result_present"] for item in tasks)
    any_stderr = any(item["stderr_bytes"] > 0 for item in tasks)
    if any_stderr:
        status = "ATTENTION_STDERR"
    elif all_completed:
        status = "READY_FOR_FROZEN_ANALYZER"
    elif all_instrumented:
        status = "RUNNING_INSTRUMENTED"
    else:
        status = "RUNNING_AWAITING_FIRST_EPOCH"
    return {
        "all_tasks_instrumented": all_instrumented,
        "claim_boundary": (
            "Development/validation progress only. Latest epochs can differ across "
            "tasks and are not a matched arm comparison. Test remains sealed."
        ),
        "datasets": datasets,
        "format_version": "negative_sample_ablation_v2_progress_audit_v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "tasks": tasks,
        "test_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asep-job-id", required=True)
    parser.add_argument("--sabdab2-job-id", required=True)
    parser.add_argument("--require-all-instrumented", action="store_true")
    args = parser.parse_args()
    payload = audit(
        data_root=args.data_root,
        asep_job_id=args.asep_job_id,
        sabdab2_job_id=args.sabdab2_job_id,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_all_instrumented and (
        not payload["all_tasks_instrumented"]
        or payload["status"] == "ATTENTION_STDERR"
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
