#!/usr/bin/env python3
"""Analyze the frozen seed-17 strict TopK2 transfer retraining screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("paratope_topk2", "contact_topk2_context")
REFERENCE_RESULT_SHA256 = "bbefaf74c4c785009470cd4f51c255dfc04286ada86139a7125e0a8f182caa42"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    args = parser.parse_args()
    output = args.campaign_root / "MULTITASK_TOPK2_TRAINING_RESULT_V1.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if sha256_file(args.reference_result) != REFERENCE_RESULT_SHA256:
        raise ValueError("reference result differs")
    reference_result = json.loads(args.reference_result.read_text())
    reference_metrics = reference_result["selected_checkpoint_validation"]
    reference = {
        task: float(reference_metrics[task]["mean_per_complex_auprc"])
        for task in ("contact", "epitope", "paratope")
    }
    campaign_path = args.campaign_root / "authority/CAMPAIGN_MULTITASK_TOPK2_TRAINING_V1.json"
    campaign = json.loads(campaign_path.read_text())
    if campaign["status"] != "FROZEN_BEFORE_SEED17_TRAINING":
        raise ValueError("campaign differs")
    rows = {}
    for arm in ARMS:
        authority = campaign["arms"][arm]
        protocol_path = Path(authority["protocol_path"])
        receipt_path = Path(authority["receipt_path"])
        if sha256_file(protocol_path) != authority["protocol_sha256"]:
            raise ValueError(f"{arm} protocol differs")
        if sha256_file(receipt_path) != authority["receipt_sha256"]:
            raise ValueError(f"{arm} receipt differs")
        run_root = args.campaign_root / f"runs/{arm}/seed-17"
        result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
        run_receipt_path = run_root / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text())
        run_receipt = json.loads(run_receipt_path.read_text())
        protocol = json.loads(protocol_path.read_text())
        config = result["configuration"]
        auth = result["authority"]
        if (
            config["seed"] != 17
            or config["epitope_pooling"] != "topk_mean"
            or config["epitope_pooling_k"] != 2
            or config["paratope_pooling"] != authority["paratope_pooling"]
            or config["paratope_pooling_k"] != 2
            or config["contact_readout"] != authority["contact_readout"]
            or protocol["paratope_readout"]["node_logit_retained"] is not True
            or protocol["contact_readout"]["pair_index_preserved"] is not True
            or protocol["claim_boundary"]["test_loaded"] is not False
            or protocol["claim_boundary"]["temporal_holdout_loaded"] is not False
            or auth["protocol_file_sha256"] != authority["protocol_sha256"]
            or auth["protocol_receipt_sha256"] != authority["receipt_sha256"]
            or auth["test_loaded"] is not False
            or auth["external_test_loaded"] is not False
            or run_receipt["test_loaded"] is not False
        ):
            raise ValueError(f"{arm} run authority differs")
        metrics = result["selected_checkpoint_validation"]
        row = {
            "selected_epoch": result["selection"]["best_epoch"],
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "receipt_sha256": sha256_file(run_receipt_path),
        }
        for task in ("contact", "epitope", "paratope"):
            value = float(metrics[task]["mean_per_complex_auprc"])
            row[f"{task}_mean_per_complex_ap"] = value
            row[f"{task}_delta_vs_reference"] = value - reference[task]
        rows[arm] = row
    payload = {
        "format_version": "sabdab2_multitask_topk2_training_result_v1",
        "status": "PASS_COMPLETE_SINGLE_SEED_DEVELOPMENT_SCREEN",
        "campaign_sha256": sha256_file(campaign_path),
        "reference": reference,
        "rows": rows,
        "claim_boundary": campaign["claim_boundary"],
    }
    encoded = canonical_bytes(payload)
    output.write_bytes(encoded)
    print(json.dumps({"status": payload["status"], "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
