#!/usr/bin/env python3
"""Freeze full-development training for the confirmed task-routed candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


FRESH_ANALYSIS_SHA256 = "0e6c6c1a029f35c6bc8f42e6d33aa15f8bac58cb376a5e89f38bc8ce84747800"
BASELINE_PROTOCOL_SHA256 = "cb5c6c7569a26ed86f3dbc2b0167ecd0260f8fe6bd7de0ad4bde94bc995d49ef"
BASELINE_RECEIPT_SHA256 = "e64c57f4181d56fde7dbecf8858e655b57f7ecfdb8764010be618bc576876275"
CANDIDATE_TEMPLATE_PROTOCOL_SHA256 = "57893a4c83cc6fd78405db1820880ac46b3598119deac2a825891457b5807c9a"
CANDIDATE_SOURCE_MANIFEST_SHA256 = "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55"
CANDIDATE_TRAINER_SHA256 = "ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e"
CANDIDATE_FRAMECONTACT_SHA256 = "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
TRAIN_PANEL_SHA256 = "fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6"
VALIDATION_PANEL_SHA256 = "222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5"
SEEDS = (17, 29, 43)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-protocol", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--candidate-template-protocol", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--candidate-source-manifest", type=Path, required=True)
    parser.add_argument("--fresh-analysis", type=Path, required=True)
    parser.add_argument("--full-split", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--training-sbatch", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    for path, expected, label in (
        (args.baseline_protocol, BASELINE_PROTOCOL_SHA256, "baseline protocol"),
        (args.baseline_receipt, BASELINE_RECEIPT_SHA256, "baseline receipt"),
        (
            args.candidate_template_protocol,
            CANDIDATE_TEMPLATE_PROTOCOL_SHA256,
            "candidate template protocol",
        ),
        (
            args.candidate_source_manifest,
            CANDIDATE_SOURCE_MANIFEST_SHA256,
            "candidate source manifest",
        ),
        (
            args.candidate_source_root / "scripts/train_framecontact.py",
            CANDIDATE_TRAINER_SHA256,
            "candidate trainer",
        ),
        (
            args.candidate_source_root / "src/interaction_first/framecontact.py",
            CANDIDATE_FRAMECONTACT_SHA256,
            "candidate FrameContact",
        ),
        (args.fresh_analysis, FRESH_ANALYSIS_SHA256, "fresh156 analysis"),
        (args.full_split, FULL_SPLIT_SHA256, "full development split"),
        (args.train_panel, TRAIN_PANEL_SHA256, "training partner panel"),
        (
            args.validation_panel,
            VALIDATION_PANEL_SHA256,
            "validation partner panel",
        ),
    ):
        require_sha(path, expected, label)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    fresh = json.loads(args.fresh_analysis.read_text())
    if (
        fresh.get("status") != "PASS_FREEZE_FULL_DEVELOPMENT_TASK_ROUTING"
        or fresh.get("full_development_protocol_authorized") is not True
        or fresh.get("old_test_loaded") is not False
        or fresh.get("temporal_holdout_loaded") is not False
        or fresh.get("task_routing", {}).get("routing_is_fixed_per_task") is not True
        or fresh.get("task_routing", {}).get("mixture_weights_tuned") is not False
    ):
        raise ValueError("fresh156 gate does not authorize full development")

    baseline_protocol = json.loads(args.baseline_protocol.read_text())
    baseline_receipt = json.loads(args.baseline_receipt.read_text())
    if (
        baseline_protocol.get("protocol_status")
        != "frozen_before_full_development_training"
        or baseline_protocol.get("arm") != "baseline"
        or baseline_protocol.get("data", {}).get("full_split_sha256")
        != FULL_SPLIT_SHA256
        or baseline_protocol.get("claim_boundary", {}).get("test_loaded") is not False
        or baseline_receipt.get("protocol_file_sha256") != BASELINE_PROTOCOL_SHA256
    ):
        raise ValueError("baseline full-development authority differs")

    baseline_runs: dict[str, dict[str, object]] = {}
    for seed in SEEDS:
        run_root = args.baseline_root / f"baseline/seed-{seed}"
        result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
        receipt_path = run_root / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        authority = result.get("authority", {})
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or result.get("configuration", {}).get("seed") != seed
            or authority.get("split_manifest_sha256") != FULL_SPLIT_SHA256
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or receipt.get("test_loaded") is not False
        ):
            raise ValueError(f"baseline seed-{seed} authority differs")
        baseline_runs[str(seed)] = {
            "run_root": str(run_root),
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "receipt_sha256": sha256_file(receipt_path),
            "selected_epoch": int(result["selection"]["best_epoch"]),
        }

    candidate = copy.deepcopy(
        json.loads(args.candidate_template_protocol.read_text())
    )
    candidate["schema_version"] = (
        "sabdab2_edge_topk2_task_routed_full_development_v1"
    )
    candidate["protocol_status"] = "frozen_before_full_development_training"
    candidate["arm"] = "edge_topk2_supervised"
    candidate["data"]["full_split_sha256"] = FULL_SPLIT_SHA256
    candidate["data"]["full_development"] = {
        "train_count": 2565,
        "validation_count": 549,
        "test_count_not_loaded": 549,
        "split_sha256": FULL_SPLIT_SHA256,
        "test_features_or_labels_loaded": False,
    }
    candidate["data"].pop("screen_split_sha256", None)
    candidate["data"].pop("selection_split_sha256", None)
    candidate["data"].pop("confirmation_split_sha256", None)
    candidate["data"]["reciprocal_partner_panel"] = {
        "format_version": "framecontact_reciprocal_partner_panel_v2",
        "size_per_query": 31,
        "training_sha256": TRAIN_PANEL_SHA256,
        "validation_sha256": VALIDATION_PANEL_SHA256,
    }
    candidate["model_selection"].update(
        {
            "seeds": list(SEEDS),
            "maximum_epochs": 20,
            "early_stopping_patience": 5,
            "primary_metric": "epitope_mean_per_complex_auprc",
            "checkpoint_rule": (
                "maximize validation mean-per-complex epitope AUPRC; break ties "
                "by contact mean-per-complex AUPRC and then earlier epoch"
            ),
        }
    )
    candidate["claim_boundary"] = {
        "development_only": True,
        "full_development_only": True,
        "route_frozen_after_predecessor_confirmation": True,
        "route_independently_confirmed_on_fresh156": True,
        "fresh156_analysis_sha256": FRESH_ANALYSIS_SHA256,
        "full_validation_is_checkpoint_selection_data": True,
        "full_validation_is_independent_confirmation": False,
        "old_sabdab2_test_may_not_be_used_for_selection": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    candidate["task_routing"] = fresh["task_routing"]
    candidate["full_development_gate"] = {
        "median_epitope_delta_minimum": 0.005,
        "epitope_positive_seed_count_minimum": 2,
        "minimum_seed_epitope_delta": -0.01,
        "maximum_absolute_contact_delta": 1e-12,
        "maximum_absolute_paratope_delta": 1e-12,
        "reference_arm": "baseline_current",
        "candidate_arm": "edge_topk2_supervised",
        "test_loaded": False,
    }
    candidate["execution_sources"] = {
        "candidate_source_manifest_sha256": CANDIDATE_SOURCE_MANIFEST_SHA256,
        "candidate_trainer_sha256": CANDIDATE_TRAINER_SHA256,
        "candidate_framecontact_sha256": CANDIDATE_FRAMECONTACT_SHA256,
        "training_sbatch_sha256": sha256_file(args.training_sbatch),
        "analyzer_sha256": sha256_file(args.analyzer),
    }
    candidate["baseline_binding"] = baseline_runs

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        authority = temporary / "authority"
        authority.mkdir(parents=True)
        (temporary / "logs").mkdir()
        (temporary / "runs").mkdir()
        protocol_path = authority / "PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1.json"
        protocol_path.write_bytes(canonical_bytes(candidate))
        protocol_sha256 = sha256_file(protocol_path)
        receipt = {
            "format_version": "sabdab2_edge_topk2_routed_full_receipt_v1",
            "status": candidate["protocol_status"],
            "protocol_file_sha256": protocol_sha256,
            "fresh156_analysis_sha256": FRESH_ANALYSIS_SHA256,
            "full_split_sha256": FULL_SPLIT_SHA256,
            "baseline_run_count": len(baseline_runs),
            "candidate_source_manifest_sha256": CANDIDATE_SOURCE_MANIFEST_SHA256,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        receipt_path = authority / "PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        campaign = {
            "format_version": "sabdab2_edge_topk2_routed_full_campaign_v1",
            "status": "FROZEN_BEFORE_FULL_DEVELOPMENT_TRAINING",
            "protocol_sha256": protocol_sha256,
            "protocol_receipt_sha256": sha256_file(receipt_path),
            "fresh156_analysis_sha256": FRESH_ANALYSIS_SHA256,
            "baseline_binding": baseline_runs,
            "candidate_training_started": False,
            "full_validation_is_independent_confirmation": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        campaign_path = authority / "CAMPAIGN_EDGE_TOPK2_ROUTED_FULL_V1.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        protocol_receipt_sha256 = sha256_file(receipt_path)
        campaign_sha256 = sha256_file(campaign_path)
        temporary.replace(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "protocol_sha256": protocol_sha256,
                "protocol_receipt_sha256": protocol_receipt_sha256,
                "campaign_sha256": campaign_sha256,
                "baseline_run_count": len(baseline_runs),
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
