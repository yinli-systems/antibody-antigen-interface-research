#!/usr/bin/env python3
"""Analyze the frozen seed-17 epitope-pooling development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = (
    "topk1_max",
    "topk2",
    "topk4",
    "topk8",
    "noisy_or",
    "logmeanexp",
    "learned_softmax",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.campaign_root / "EPITOPE_POOLING_ABLATION_RESULT_V1.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    campaign_path = args.campaign_root / "authority/CAMPAIGN_EPITOPE_POOLING_ABLATION_V1.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign["status"] != "FROZEN_BEFORE_SEED17_TRAINING":
        raise ValueError("campaign authority differs")
    rows = {}
    for arm in ARMS:
        record = campaign["arms"][arm]
        protocol_path = Path(record["protocol_path"])
        receipt_path = Path(record["receipt_path"])
        if sha256_file(protocol_path) != record["protocol_sha256"]:
            raise ValueError(f"{arm} protocol SHA differs")
        if sha256_file(receipt_path) != record["receipt_sha256"]:
            raise ValueError(f"{arm} receipt SHA differs")
        run = args.campaign_root / f"runs/{arm}/seed-17"
        result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        checkpoint_path = run / "FRAMECONTACT_SELECTED.pt"
        run_receipt_path = run / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
        configuration = result["configuration"]
        authority = result["authority"]
        if (
            configuration["seed"] != 17
            or configuration["epitope_pooling"] != record["pooling"]
            or configuration["epitope_pooling_k"] != (record["k"] or 2)
            or authority["protocol_file_sha256"] != record["protocol_sha256"]
            or authority["protocol_receipt_sha256"] != record["receipt_sha256"]
            or authority["test_loaded"] is not False
            or authority["external_test_loaded"] is not False
            or run_receipt["test_loaded"] is not False
        ):
            raise ValueError(f"{arm} run authority differs")
        metrics = result["selected_checkpoint_validation"]
        rows[arm] = {
            "pooling": record["pooling"],
            "k": record["k"],
            "selected_epoch": result["selection"]["best_epoch"],
            "contact_mean_per_complex_auprc": metrics["contact"]["mean_per_complex_auprc"],
            "epitope_mean_per_complex_auprc": metrics["epitope"]["mean_per_complex_auprc"],
            "paratope_mean_per_complex_auprc": metrics["paratope"]["mean_per_complex_auprc"],
            "result_sha256": sha256_file(result_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "receipt_sha256": sha256_file(run_receipt_path),
        }
    reference = rows["topk2"]
    for row in rows.values():
        for task in ("contact", "epitope", "paratope"):
            key = f"{task}_mean_per_complex_auprc"
            row[f"{task}_delta_vs_topk2"] = row[key] - reference[key]
    payload = {
        "format_version": "sabdab2_epitope_pooling_ablation_result_v1",
        "status": "PASS_COMPLETE_SINGLE_SEED_DEVELOPMENT_SCREEN",
        "campaign_sha256": sha256_file(campaign_path),
        "reference_arm": "topk2",
        "topk1_equals_max_by_definition": True,
        "rows": rows,
        "claim_boundary": {
            "single_seed": 17,
            "development_only": True,
            "sufficient_for_optimal_k_claim": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    encoded = canonical_bytes(payload)
    output.write_bytes(encoded)
    print(json.dumps({"status": payload["status"], "sha256": hashlib.sha256(encoded).hexdigest()}))


if __name__ == "__main__":
    main()
