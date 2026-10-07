#!/usr/bin/env python3
"""Freeze all matched reviewer-ablation training outputs before any test access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-protocol", required=True, type=Path)
    parser.add_argument("--campaign-protocol-sha256", required=True)
    parser.add_argument("--analysis-protocol", required=True, type=Path)
    parser.add_argument("--analysis-protocol-sha256", required=True)
    parser.add_argument("--authority-receipt", required=True, type=Path)
    parser.add_argument("--authority-receipt-sha256", required=True)
    parser.add_argument("--smoke-manifest", required=True, type=Path)
    parser.add_argument("--smoke-manifest-sha256", required=True)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--train-job-id-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    started = time.monotonic()

    campaign_path = arguments.campaign_protocol.resolve()
    analysis_path = arguments.analysis_protocol.resolve()
    require(sha256_file(campaign_path) == arguments.campaign_protocol_sha256,
            "Campaign protocol hash differs")
    require(sha256_file(analysis_path) == arguments.analysis_protocol_sha256,
            "Analysis protocol hash differs")
    campaign = read_json(campaign_path)
    analysis = read_json(analysis_path)
    require(analysis["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
            "Analysis protocol belongs to another campaign")
    require(sha256_file(Path(__file__).resolve()) == analysis["implementation"]["freeze_training.py"],
            "Training freezer differs from preregistered implementation")
    require(analysis["status"] == "PREREGISTERED_BEFORE_FIRST_FULL_TRAINING_RESULT",
            "Analysis was not preregistered before results")

    authority_path = arguments.authority_receipt.resolve()
    smoke_path = arguments.smoke_manifest.resolve()
    require(sha256_file(authority_path) == arguments.authority_receipt_sha256,
            "Authority receipt hash differs")
    require(sha256_file(smoke_path) == arguments.smoke_manifest_sha256,
            "Smoke manifest hash differs")
    authority = read_json(authority_path)
    smoke = read_json(smoke_path)
    require(authority["status"] == "PASS_FULL_PARENT_BUNDLE_VERIFICATION",
            "Parent authority was not fully verified")
    require(smoke["status"] == "PASS_ALL_CONDITION_SMOKES_FROZEN",
            "Condition smokes were not fully frozen")
    require(authority["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
            "Authority receipt campaign differs")
    require(smoke["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
            "Smoke manifest campaign differs")

    runs_root = arguments.runs_root.resolve()
    logs_root = arguments.logs_root.resolve()
    require(runs_root.is_dir() and not runs_root.is_symlink(), "Runs root is invalid")
    require(logs_root.is_dir() and not logs_root.is_symlink(), "Logs root is invalid")
    train_job_id = arguments.train_job_id_file.read_text().strip()
    require(train_job_id.isdigit(), "Training job id is invalid")

    conditions = [entry["name"] for entry in campaign["conditions"]]
    seeds = [int(seed) for seed in campaign["seeds"]]
    expected_count = len(conditions) * len(seeds)
    records: list[dict[str, Any]] = []
    checkpoint_hashes: set[str] = set()
    result_hashes: set[str] = set()
    validation_ids_reference: set[str] | None = None

    for condition_index, condition in enumerate(conditions):
        require(condition in smoke["receipts"], f"Missing smoke receipt for {condition}")
        for seed_index, seed in enumerate(seeds):
            task_index = condition_index * len(seeds) + seed_index
            run_root = runs_root / condition / f"seed-{seed}"
            require(run_root.is_dir() and not run_root.is_symlink(),
                    f"Missing run directory: {condition}/{seed}")
            result_path = run_root / "RESULT.json"
            started_path = run_root / "STARTED.json"
            require(result_path.is_file() and started_path.is_file(),
                    f"Missing result/start receipt: {condition}/{seed}")
            result = read_json(result_path)
            start_receipt = read_json(started_path)
            require(result["status"] == "PASS_TRAINING_COMPLETE_CHECKPOINT_FROZEN",
                    f"Training did not complete cleanly: {condition}/{seed}")
            for payload, label in ((result, "result"), (start_receipt, "start")):
                require(payload["campaign_protocol_sha256"] == arguments.campaign_protocol_sha256,
                        f"{label} campaign differs: {condition}/{seed}")
                require(payload["condition"] == condition and int(payload["seed"]) == seed,
                        f"{label} condition/seed differs: {condition}/{seed}")
                require(payload.get("test_label_artifacts_opened") is False,
                        f"{label} reports test-label access: {condition}/{seed}")
            require(result.get("test_selection_performed") is False,
                    f"Test selection reported: {condition}/{seed}")
            require(result.get("old_checkpoint_loaded") is False,
                    f"Old checkpoint loaded: {condition}/{seed}")
            require(result["authority_receipt_sha256"] == arguments.authority_receipt_sha256,
                    f"Authority receipt differs: {condition}/{seed}")
            require(str(result["slurm_array_task_id"]) == str(task_index),
                    f"Array task mapping differs: {condition}/{seed}")
            require(str(start_receipt["slurm_array_task_id"]) == str(task_index),
                    f"Start array task mapping differs: {condition}/{seed}")

            checkpoint_path = run_root / result["checkpoint"]
            validation_path = run_root / result["validation_evidence"]
            require(checkpoint_path.is_file() and not checkpoint_path.is_symlink(),
                    f"Missing checkpoint: {condition}/{seed}")
            require(validation_path.is_file() and not validation_path.is_symlink(),
                    f"Missing validation evidence: {condition}/{seed}")
            checkpoint_sha = sha256_file(checkpoint_path)
            validation_sha = sha256_file(validation_path)
            result_sha = sha256_file(result_path)
            started_sha = sha256_file(started_path)
            require(checkpoint_sha == result["checkpoint_sha256"],
                    f"Checkpoint hash differs: {condition}/{seed}")
            require(validation_sha == result["validation_evidence_sha256"],
                    f"Validation hash differs: {condition}/{seed}")
            require(checkpoint_sha not in checkpoint_hashes,
                    f"Duplicate checkpoint bytes: {condition}/{seed}")
            require(result_sha not in result_hashes,
                    f"Duplicate result bytes: {condition}/{seed}")
            checkpoint_hashes.add(checkpoint_sha)
            result_hashes.add(result_sha)

            validation = read_json(validation_path)
            require(validation["condition"] == condition and int(validation["seed"]) == seed,
                    f"Validation condition/seed differs: {condition}/{seed}")
            require(validation["checkpoint_sha256"] == checkpoint_sha,
                    f"Validation checkpoint binding differs: {condition}/{seed}")
            require(validation["frozen_state_sha256"] == result["frozen_state_sha256"],
                    f"Frozen-state binding differs: {condition}/{seed}")
            require(validation.get("test_label_artifacts_opened") is False,
                    f"Validation reports test-label access: {condition}/{seed}")
            require(int(validation["best_epoch"]) == int(result["best_epoch"]),
                    f"Best epoch differs: {condition}/{seed}")
            require(finite(result["wall_seconds"]) and float(result["wall_seconds"]) > 0,
                    f"Invalid wall time: {condition}/{seed}")
            require(int(result["total_parameters"]) > 0 and int(result["trainable_parameters"]) > 0,
                    f"Invalid parameter counts: {condition}/{seed}")
            require(result["initial_state_sha256"] != result["frozen_state_sha256"],
                    f"Model state did not change: {condition}/{seed}")

            metrics = validation["metrics"]
            task_id_sets: list[set[str]] = []
            for task in ("contact", "epitope", "paratope"):
                task_metrics = metrics[task]
                per_complex = task_metrics.get("per_complex")
                require(isinstance(per_complex, list) and len(per_complex) == campaign["data_counts"]["validation"],
                        f"Incomplete per-complex validation: {condition}/{seed}/{task}")
                ids = [str(row["instance_id"]) for row in per_complex]
                require(len(ids) == len(set(ids)),
                        f"Duplicate validation instance: {condition}/{seed}/{task}")
                task_id_sets.append(set(ids))
                ap_values = [row["average_precision"] for row in per_complex
                             if row["average_precision"] is not None]
                auroc_values = [row["auroc"] for row in per_complex if row["auroc"] is not None]
                require(len(ap_values) == int(task_metrics["mpc_ap_complex_count"]),
                        f"AP count differs: {condition}/{seed}/{task}")
                require(len(auroc_values) == int(task_metrics["mpc_auroc_complex_count"]),
                        f"AUROC count differs: {condition}/{seed}/{task}")
                require(abs(sum(ap_values) / len(ap_values) - float(task_metrics["mpc_ap"])) < 1e-12,
                        f"AP recomputation differs: {condition}/{seed}/{task}")
                require(abs(sum(auroc_values) / len(auroc_values) - float(task_metrics["mpc_auroc"])) < 1e-12,
                        f"AUROC recomputation differs: {condition}/{seed}/{task}")
                for scalar in ("fixed_05_precision", "fixed_05_recall", "brier", "ece_20bin"):
                    require(finite(task_metrics[scalar]),
                            f"Nonfinite metric {scalar}: {condition}/{seed}/{task}")
            require(task_id_sets[0] == task_id_sets[1] == task_id_sets[2],
                    f"Task validation membership differs: {condition}/{seed}")
            if validation_ids_reference is None:
                validation_ids_reference = task_id_sets[0]
            else:
                require(task_id_sets[0] == validation_ids_reference,
                        f"Validation membership differs across runs: {condition}/{seed}")
            require("contact_ranked_cutoffs" in metrics,
                    f"Ranked contact metrics missing: {condition}/{seed}")

            stderr_path = logs_root / f"train-{train_job_id}_{task_index}.err"
            stdout_path = logs_root / f"train-{train_job_id}_{task_index}.out"
            require(stderr_path.is_file() and stdout_path.is_file(),
                    f"Missing Slurm log: task {task_index}")
            require(stderr_path.stat().st_size == 0,
                    f"Nonempty stderr requires audit: task {task_index}")
            records.append({
                "condition": condition,
                "seed": seed,
                "array_task_id": task_index,
                "slurm_child_job_id": str(result["slurm_job_id"]),
                "run_root": str(run_root),
                "started_path": str(started_path),
                "started_sha256": started_sha,
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "validation_path": str(validation_path),
                "validation_sha256": validation_sha,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "stdout_path": str(stdout_path),
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_path": str(stderr_path),
                "stderr_sha256": sha256_file(stderr_path),
                "best_epoch": int(result["best_epoch"]),
                "executed_epochs": len(result["trace"]),
                "wall_seconds": float(result["wall_seconds"]),
                "total_parameters": int(result["total_parameters"]),
                "trainable_parameters": int(result["trainable_parameters"]),
                "initial_state_sha256": result["initial_state_sha256"],
                "frozen_state_sha256": result["frozen_state_sha256"],
            })
            print(json.dumps({"frozen": len(records), "expected": expected_count,
                              "condition": condition, "seed": seed}, sort_keys=True), flush=True)

    require(len(records) == expected_count, "Training record count differs")
    require(validation_ids_reference is not None and
            len(validation_ids_reference) == campaign["data_counts"]["validation"],
            "Validation authority count differs")
    payload: dict[str, Any] = {
        "status": "PASS_ALL_TRAINING_CHECKPOINTS_FROZEN_BEFORE_TEST",
        "format_version": "epic_reviewer_closure_training_manifest_v1",
        "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
        "analysis_protocol_sha256": arguments.analysis_protocol_sha256,
        "authority_receipt_sha256": arguments.authority_receipt_sha256,
        "smoke_manifest_sha256": arguments.smoke_manifest_sha256,
        "train_array_job_id": train_job_id,
        "record_count": len(records),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "validation_instance_count": len(validation_ids_reference),
        "validation_instance_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(validation_ids_reference)) + "\n").encode()
        ).hexdigest(),
        "records": records,
        "test_label_artifacts_opened": False,
        "test_selection_performed": False,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "wall_seconds": time.monotonic() - started,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("COMPLETE", arguments.output, sha256_file(arguments.output), flush=True)


if __name__ == "__main__":
    main()
