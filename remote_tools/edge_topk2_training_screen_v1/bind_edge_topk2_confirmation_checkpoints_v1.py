#!/usr/bin/env python3
"""Audit and bind all six checkpoints before opening confirmation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CAMPAIGN_SHA256 = "31ac56aca9ed32371abfb34154693faf11b36908fe035201524bac8c4cc6cf75"
MEASUREMENT_CONTRACT_SHA256 = "a0f8a67d43bd3fe30c4769e3ad2a38bd7923f6d42071f7ce9fed8bfe0b609760"
SELECTION_RESULT_SHA256 = "3e74ba4fd4d2f4b6deffd68ae0f1420712ac8425ddba6b4bdcd3b0e1cbad416c"
SELECTION_SPLIT_SHA256 = "39a32540dd4a6d9abcc8414c23dfbe52d446e9154c06ce58921a0dcbd5039686"
TRAIN_PANEL_SHA256 = "c209b72cfe174c63fa986cb59e8dbf6a3831f128493b0c992a604ddf6a6cfca6"
SELECTION_PANEL_SHA256 = "86d613c1917d3a8d3fe6da49439cb0a13ab9f44affd2ea27e30043f3d71ae6e2"
CONFIRMATION_SPLIT_SHA256 = "69350cfe30348c818fa9998bc5063df3c235954633fae46e20a2295f751efad4"
CONFIRMATION_PANEL_SHA256 = "6cf324c4fdd06310a2f8d303d1a7d8bdfa46c6e7254f4a6068c4b34ac785400b"
SOURCE_MANIFEST_SHA256 = {
    "baseline_current": "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81",
    "edge_topk2_supervised": "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55",
}
FRAMECONTACT_SHA256 = {
    "baseline_current": "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c",
    "edge_topk2_supervised": "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe",
}
METRICS = ("contact", "epitope", "paratope")


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


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def audit_run(
    run_root: Path,
    *,
    arm: str,
    seed: int,
    protocol_sha256: str,
    protocol_receipt_sha256: str,
    expected_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    hashes = {
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
    }
    if expected_artifacts is not None:
        for name, observed in hashes.items():
            if observed != expected_artifacts[name]:
                raise ValueError(f"{arm} seed-{seed} {name} differs")
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    authority = result.get("authority", {})
    configuration = result.get("configuration", {})
    selection = result.get("selection", {})
    if (
        receipt.get("result_sha256") != hashes["result_sha256"]
        or receipt.get("selected_checkpoint_sha256") != hashes["checkpoint_sha256"]
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != seed
        or configuration.get("dataset_backend") != "sabdab2"
        or configuration.get("device") != "cuda"
        or configuration.get("precision") != "bf16"
        or configuration.get("epochs") != 12
        or configuration.get("early_stopping_patience") != 3
        or authority.get("protocol_file_sha256") != protocol_sha256
        or authority.get("protocol_receipt_sha256") != protocol_receipt_sha256
        or authority.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256[arm]
        or authority.get("split_manifest_sha256") != SELECTION_SPLIT_SHA256
        or authority.get("training_partner_panel_sha256") != TRAIN_PANEL_SHA256
        or authority.get("validation_partner_panel_sha256") != SELECTION_PANEL_SHA256
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
    ):
        raise ValueError(f"{arm} seed-{seed} authority differs")
    selected = result["selected_checkpoint_validation"]
    calibration = {
        "temperature": finite(selected["validation_temperature"]),
        "contact_threshold": finite(
            selected["contact"]["threshold_selected_on_validation"]
        ),
        "paratope_temperature": finite(
            selected["paratope"]["validation_temperature"]
        ),
        "paratope_threshold": finite(
            selected["paratope"]["threshold_selected_on_validation"]
        ),
        "epitope_temperature": finite(
            selected["epitope"]["validation_temperature"]
        ),
        "epitope_threshold": finite(
            selected["epitope"]["threshold_selected_on_validation"]
        ),
    }
    return {
        "run_root": str(run_root),
        **hashes,
        "selected_epoch": int(selection["best_epoch"]),
        "epochs_completed": int(selection["epochs_completed"]),
        "selection_metrics": {
            name: finite(selected[name]["mean_per_complex_auprc"])
            for name in METRICS
        },
        "frozen_calibration": calibration,
        "specificity_score_source": str(selected["specificity_score_source"]),
        "configuration_sha256": hashlib.sha256(
            canonical_bytes(configuration)
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--baseline-seed17-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--measurement-sbatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)

    authority = args.campaign_root / "authority"
    campaign_path = authority / "CAMPAIGN_EDGE_TOPK2_THREE_SEED_CONFIRMATION_V1.json"
    contract_path = authority / "PROTOCOL_CONFIRMATION_MEASUREMENT_CONTRACT_V1.json"
    selection_result_path = args.selection_root / "EDGE_TOPK2_TRAINING_SELECTION_RESULT_V1.json"
    require_sha(campaign_path, CAMPAIGN_SHA256, "campaign")
    require_sha(contract_path, MEASUREMENT_CONTRACT_SHA256, "measurement contract")
    require_sha(selection_result_path, SELECTION_RESULT_SHA256, "selection result")
    require_sha(
        authority / "split-train512-selection63.csv",
        SELECTION_SPLIT_SHA256,
        "selection split",
    )
    require_sha(
        authority / "train512-reciprocal-6.json",
        TRAIN_PANEL_SHA256,
        "training panel",
    )
    require_sha(
        authority / "validation-selection63-reciprocal-6.json",
        SELECTION_PANEL_SHA256,
        "selection panel",
    )
    require_sha(
        authority / "split-confirmation65-measurement-only.csv",
        CONFIRMATION_SPLIT_SHA256,
        "confirmation split",
    )
    require_sha(
        authority / "validation-confirmation65-reciprocal-6.json",
        CONFIRMATION_PANEL_SHA256,
        "confirmation panel",
    )
    campaign = json.loads(campaign_path.read_text())
    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_result_path.read_text())
    if (
        campaign.get("status") != "FROZEN_BEFORE_TRAINING"
        or campaign.get("confirmation_labels_read") is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or contract.get("status") != "FROZEN_BEFORE_CHECKPOINT_BINDING"
        or contract.get("checkpoint_binding")
        != "required_after_training_before_measurement"
        or contract.get("full_development_started") is not False
        or contract.get("old_test") is not False
        or contract.get("temporal_holdout") is not False
        or selection.get("status") != "PASS_FREEZE_THREE_SEED_CONFIRMATION"
    ):
        raise ValueError("confirmation authority differs")

    selection_runs = selection["runs"]
    existing = {
        "baseline_current": selection_runs["baseline_current"],
        "edge_topk2_supervised": selection_runs["edge_topk2_supervised"],
    }
    run_roots = {
        "baseline_current": {
            17: args.baseline_seed17_root,
            29: args.campaign_root / "runs/baseline_current/seed-29",
            43: args.campaign_root / "runs/baseline_current/seed-43",
        },
        "edge_topk2_supervised": {
            17: args.selection_root / "runs/edge_topk2_supervised/seed-17",
            29: args.campaign_root / "runs/edge_topk2_supervised/seed-29",
            43: args.campaign_root / "runs/edge_topk2_supervised/seed-43",
        },
    }
    bound_runs: dict[str, dict[str, Any]] = {}
    for arm, by_seed in run_roots.items():
        protocol_sha256 = (
            "7cb1ab917137dfe1de821a1eea846fd22e656a12c1d3dd99a45121bf9f77e181"
            if arm == "baseline_current"
            else "cac9957ec6c34d27f2c29cbde8d012ccfcb1ca779d4ea378376429d1a238ca94"
        )
        protocol_receipt_sha256 = (
            "2b48039ea9e39d866e07ac8c718403109509d3db012b713bbfc1c8b028696db5"
            if arm == "baseline_current"
            else "d0c233e4570bdbcda715a93ec9753f4b0fd4131a5d579d6ac07c0d01bdbfbc51"
        )
        bound_runs[arm] = {}
        for seed, run_root in by_seed.items():
            expected = None
            if seed == 17:
                row = existing[arm]
                expected = {
                    "result_sha256": row["result_sha256"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "receipt_sha256": row["receipt_sha256"],
                }
            else:
                protocol_sha256 = campaign["training_protocol_sha256"][arm]
                protocol_receipt_sha256 = campaign[
                    "training_protocol_receipt_sha256"
                ][arm]
            bound_runs[arm][str(seed)] = audit_run(
                run_root,
                arm=arm,
                seed=seed,
                protocol_sha256=protocol_sha256,
                protocol_receipt_sha256=protocol_receipt_sha256,
                expected_artifacts=expected,
            )

    payload = {
        "format_version": "sabdab2_edge_topk2_confirmation_checkpoint_binding_v1",
        "status": "FROZEN_BEFORE_CONFIRMATION_MEASUREMENT",
        "campaign_sha256": CAMPAIGN_SHA256,
        "measurement_contract_sha256": MEASUREMENT_CONTRACT_SHA256,
        "selection_result_sha256": SELECTION_RESULT_SHA256,
        "confirmation_split_sha256": CONFIRMATION_SPLIT_SHA256,
        "confirmation_partner_panel_sha256": CONFIRMATION_PANEL_SHA256,
        "runs": bound_runs,
        "execution_sources": {
            "evaluator_sha256": sha256_file(args.evaluator),
            "analyzer_sha256": sha256_file(args.analyzer),
            "measurement_sbatch_sha256": sha256_file(args.measurement_sbatch),
        },
        "confirmation_labels_read": False,
        "selection_performed_on_confirmation": False,
        "model_parameters_updated_during_confirmation": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "bound_checkpoints": sum(len(rows) for rows in bound_runs.values()),
                "output_sha256": sha256_file(args.output),
                "confirmation_labels_read": False,
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
