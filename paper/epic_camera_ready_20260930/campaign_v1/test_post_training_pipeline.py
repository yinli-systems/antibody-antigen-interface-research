#!/usr/bin/env python3
"""Synthetic integration test for freeze_training.py and aggregate_validation.py."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
FREEZER = HERE / "freeze_training.py"
AGGREGATOR = HERE / "aggregate_validation.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def task_metrics(offset: float, ids: list[str]) -> dict:
    per_complex = []
    for index, instance_id in enumerate(ids):
        ap = 0.10 + offset + index * 0.01
        auroc = 0.70 + offset + index * 0.01
        per_complex.append({
            "instance_id": instance_id,
            "candidate_count": 100 + index,
            "positive_count": 5 + index,
            "prevalence": (5 + index) / (100 + index),
            "average_precision": ap,
            "auroc": auroc,
            "precision_at_05": 0.20 + offset + index * 0.01,
            "recall_at_05": 0.30 + offset + index * 0.01,
        })
    ap_values = [row["average_precision"] for row in per_complex]
    auroc_values = [row["auroc"] for row in per_complex]
    return {
        "mpc_ap": sum(ap_values) / len(ap_values),
        "mpc_auroc": sum(auroc_values) / len(auroc_values),
        "mpc_ap_complex_count": len(ap_values),
        "mpc_auroc_complex_count": len(auroc_values),
        "fixed_05_counts": {"tp": 10, "fp": 20, "tn": 60, "fn": 10},
        "fixed_05_precision": 1 / 3 + offset,
        "fixed_05_recall": 0.5 + offset,
        "brier": 0.20 - offset,
        "ece_20bin": 0.10 - offset,
        "per_complex": per_complex,
    }


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="epic-reviewer-pipeline-test-"))
    try:
        conditions = [
            {"name": "full_learned_gate", "description": "synthetic reference"},
            {"name": "no_geometry_branch", "description": "synthetic ablation"},
        ]
        seeds = [17, 29, 43]
        ids = [f"complex-{index}" for index in range(4)]
        campaign = {
            "format_version": "synthetic_campaign_v1",
            "status": "PREREGISTERED_BEFORE_GPU_EXECUTION",
            "conditions": conditions,
            "seeds": seeds,
            "data_counts": {"train": 8, "validation": len(ids), "test": 2},
            "evidence_class": "synthetic pipeline test only",
        }
        campaign_path = root / "protocol/CAMPAIGN_PROTOCOL.json"
        write(campaign_path, campaign)
        campaign_sha = sha(campaign_path)
        analysis = {
            "format_version": "synthetic_analysis_v1",
            "status": "PREREGISTERED_BEFORE_FIRST_FULL_TRAINING_RESULT",
            "campaign_protocol_sha256": campaign_sha,
            "reference_condition": "full_learned_gate",
            "bootstrap": {"replicates": 200, "random_seed": 20260930},
            "interpretation_policy": ["synthetic only"],
            "claim_boundaries": ["synthetic only"],
            "implementation": {
                "freeze_training.py": sha(FREEZER),
                "aggregate_validation.py": sha(AGGREGATOR),
            },
        }
        analysis_path = root / "protocol/ANALYSIS_PROTOCOL.json"
        write(analysis_path, analysis)
        analysis_sha = sha(analysis_path)
        authority_path = root / "receipts/AUTHORITY_RECEIPT.json"
        write(authority_path, {
            "status": "PASS_FULL_PARENT_BUNDLE_VERIFICATION",
            "campaign_protocol_sha256": campaign_sha,
        })
        authority_sha = sha(authority_path)
        smoke_path = root / "receipts/SMOKE_MANIFEST.json"
        write(smoke_path, {
            "status": "PASS_ALL_CONDITION_SMOKES_FROZEN",
            "campaign_protocol_sha256": campaign_sha,
            "receipts": {entry["name"]: {} for entry in conditions},
        })
        smoke_sha = sha(smoke_path)
        (root / "receipts/TRAIN_JOB_ID.txt").write_text("999\n")

        for condition_index, entry in enumerate(conditions):
            condition = entry["name"]
            for seed_index, seed in enumerate(seeds):
                task_index = condition_index * len(seeds) + seed_index
                run = root / "runs" / condition / f"seed-{seed}"
                run.mkdir(parents=True)
                offset = 0.01 if condition == "full_learned_gate" else -0.01
                offset += (seed_index - 1) * 0.001
                metrics = {
                    "contact": task_metrics(offset, ids),
                    "epitope": task_metrics(offset / 2, ids),
                    "paratope": task_metrics(offset / 3, ids),
                    "contact_ranked_cutoffs": {},
                }
                for cutoff in ("10", "20", "50", "100", "lmin", "top_1pct", "true_count"):
                    metrics["contact_ranked_cutoffs"][cutoff] = {
                        "mean_precision": 0.05 + offset,
                        "mean_recall": 0.10 + offset,
                        "mean_enrichment_over_prevalence": 5.0 + offset,
                        "positive_complex_count": len(ids),
                    }
                checkpoint = run / "checkpoint-epoch-001.pt"
                checkpoint.write_bytes(f"checkpoint|{condition}|{seed}".encode())
                checkpoint_sha = sha(checkpoint)
                frozen_state = hashlib.sha256(f"frozen|{condition}|{seed}".encode()).hexdigest()
                validation_path = run / "VALIDATION_EVIDENCE.json"
                write(validation_path, {
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": 1,
                    "checkpoint_sha256": checkpoint_sha,
                    "frozen_state_sha256": frozen_state,
                    "metrics": metrics,
                    "test_label_artifacts_opened": False,
                })
                result_path = run / "RESULT.json"
                write(result_path, {
                    "status": "PASS_TRAINING_COMPLETE_CHECKPOINT_FROZEN",
                    "format_version": "synthetic_campaign_v1",
                    "evidence_class": "synthetic",
                    "campaign_protocol_sha256": campaign_sha,
                    "authority_receipt_sha256": authority_sha,
                    "condition": condition,
                    "seed": seed,
                    "best_epoch": 1,
                    "checkpoint": checkpoint.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "frozen_state_sha256": frozen_state,
                    "initial_state_sha256": hashlib.sha256(f"initial|{condition}|{seed}".encode()).hexdigest(),
                    "validation_evidence": validation_path.name,
                    "validation_evidence_sha256": sha(validation_path),
                    "trace": [{"epoch": 1}],
                    "total_parameters": 1000 + condition_index,
                    "trainable_parameters": 900 + condition_index,
                    "wall_seconds": 10.0 + seed_index,
                    "slurm_job_id": str(10000 + task_index),
                    "slurm_array_task_id": str(task_index),
                    "old_checkpoint_loaded": False,
                    "test_label_artifacts_opened": False,
                    "test_selection_performed": False,
                })
                write(run / "STARTED.json", {
                    "campaign_protocol_sha256": campaign_sha,
                    "condition": condition,
                    "seed": seed,
                    "slurm_array_task_id": str(task_index),
                    "test_label_artifacts_opened": False,
                })
                logs = root / "logs"
                logs.mkdir(exist_ok=True)
                (logs / f"train-999_{task_index}.out").write_text("synthetic complete\n")
                (logs / f"train-999_{task_index}.err").write_text("")

        manifest = root / "receipts/TRAINING_MANIFEST.json"
        subprocess.run([
            sys.executable, str(FREEZER),
            "--campaign-protocol", str(campaign_path),
            "--campaign-protocol-sha256", campaign_sha,
            "--analysis-protocol", str(analysis_path),
            "--analysis-protocol-sha256", analysis_sha,
            "--authority-receipt", str(authority_path),
            "--authority-receipt-sha256", authority_sha,
            "--smoke-manifest", str(smoke_path),
            "--smoke-manifest-sha256", smoke_sha,
            "--runs-root", str(root / "runs"),
            "--logs-root", str(root / "logs"),
            "--train-job-id-file", str(root / "receipts/TRAIN_JOB_ID.txt"),
            "--output", str(manifest),
        ], check=True)
        manifest_payload = json.loads(manifest.read_text())
        assert manifest_payload["status"] == "PASS_ALL_TRAINING_CHECKPOINTS_FROZEN_BEFORE_TEST"
        assert manifest_payload["record_count"] == 6

        analysis_json = root / "analysis/VALIDATION_ANALYSIS.json"
        analysis_md = root / "analysis/VALIDATION_ANALYSIS.md"
        analysis_receipt = root / "receipts/VALIDATION_ANALYSIS_RECEIPT.json"
        subprocess.run([
            sys.executable, str(AGGREGATOR),
            "--campaign-protocol", str(campaign_path),
            "--campaign-protocol-sha256", campaign_sha,
            "--analysis-protocol", str(analysis_path),
            "--analysis-protocol-sha256", analysis_sha,
            "--training-manifest", str(manifest),
            "--training-manifest-sha256", sha(manifest),
            "--output-json", str(analysis_json),
            "--output-markdown", str(analysis_md),
            "--receipt", str(analysis_receipt),
        ], check=True)
        output = json.loads(analysis_json.read_text())
        assert output["status"] == "PASS_MATCHED_MULTI_SEED_VALIDATION_ANALYSIS_NO_TEST_ACCESS"
        delta = output["paired_comparisons"]["no_geometry_branch"]["tasks"]["contact"]["mpc_ap"]
        assert delta["condition_minus_reference_mean"] < 0
        assert delta["paired_seed_and_complex_bootstrap"]["robust_raw_direction"] == "condition_lower"
        receipt = {
            "status": "PASS_SYNTHETIC_POST_TRAINING_PIPELINE_INTEGRATION_TEST",
            "freeze_training_sha256": sha(FREEZER),
            "aggregate_validation_sha256": sha(AGGREGATOR),
            "training_manifest_sha256": sha(manifest),
            "validation_analysis_sha256": sha(analysis_json),
            "validation_analysis_receipt_sha256": sha(analysis_receipt),
            "synthetic_run_count": 6,
            "synthetic_validation_complex_count": 4,
        }
        output_receipt = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "SYNTHETIC_PIPELINE_TEST_RECEIPT.json"
        write(output_receipt, receipt)
        print("PASS", output_receipt, sha(output_receipt))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
