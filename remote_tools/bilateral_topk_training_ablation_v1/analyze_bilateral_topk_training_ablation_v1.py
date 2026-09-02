#!/usr/bin/env python3
"""Analyze the eight-arm seed-17 bilateral Top-K training ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


ARMS = tuple([f"epitope_k{k}" for k in (1, 2, 4, 8)] + [f"paratope_k{k}" for k in (1, 2, 4, 8)])


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    campaign_path = args.campaign_root / "authority/BILATERAL_TOPK_TRAINING_CAMPAIGN_V1.json"
    campaign_raw = campaign_path.read_bytes()
    campaign_sha = hashlib.sha256(campaign_raw).hexdigest()
    campaign = json.loads(campaign_raw)
    if campaign["status"] != "FROZEN_BEFORE_EIGHT_ARM_SEED17_TRAINING":
        raise ValueError("campaign is not frozen")

    results: dict[str, Any] = {}
    hashes: dict[str, Any] = {}
    for arm in ARMS:
        root = args.campaign_root / f"runs/{arm}/seed-17"
        result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        receipt_path = root / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        result_sha = sha256_file(result_path)
        if receipt["result_sha256"] != result_sha:
            raise ValueError(f"result receipt differs: {arm}")
        if result["authority"]["test_loaded"] is not False:
            raise ValueError(f"test accessed: {arm}")
        if result["authority"]["external_test_loaded"] is not False:
            raise ValueError(f"external test accessed: {arm}")
        if float(result["configuration"]["distance_loss_weight"]) != 0.0:
            raise ValueError(f"distance auxiliary enabled: {arm}")
        metrics = result["selected_checkpoint_validation"]
        results[arm] = {
            "selected_epoch": int(result["selection"]["best_epoch"]),
            "contact": metrics["contact"],
            "epitope": metrics["epitope"],
            "paratope": metrics["paratope"],
        }
        hashes[arm] = {
            "result_sha256": result_sha,
            "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
            "receipt_sha256": sha256_file(receipt_path),
        }

    reference = results["epitope_k2"]
    deltas = {
        "epitope": {
            arm: results[arm]["epitope"]["mean_per_complex_auprc"]
            - reference["epitope"]["mean_per_complex_auprc"]
            for arm in ("epitope_k1", "epitope_k4", "epitope_k8")
        },
        "paratope": {
            arm: results[arm]["paratope"]["mean_per_complex_auprc"]
            - reference["paratope"]["mean_per_complex_auprc"]
            for arm in ("paratope_k1", "paratope_k2", "paratope_k4", "paratope_k8")
        },
    }
    analysis = {
        "format_version": "bilateral_topk_training_ablation_analysis_v1",
        "status": "PASS_BILATERAL_TOPK_SEED17_FROM_SCRATCH_ABLATION",
        "campaign_sha256": campaign_sha,
        "results": results,
        "hashes": hashes,
        "mean_per_complex_ap_deltas_vs_shared_epitope_k2_reference": deltas,
        "claim_boundary": campaign["claim_boundary"],
        "interpretation": "single-seed development training ablation; report all arms and do not claim an optimal K",
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        path = temporary / "BILATERAL_TOPK_TRAINING_ABLATION_ANALYSIS_V1.json"
        path.write_bytes(canonical_bytes(analysis))
        analysis_sha = sha256_file(path)
        (temporary / "BILATERAL_TOPK_TRAINING_ABLATION_ANALYSIS_V1_RECEIPT.json").write_bytes(
            canonical_bytes(
                {
                    "format_version": "bilateral_topk_training_ablation_analysis_receipt_v1",
                    "status": analysis["status"],
                    "campaign_sha256": campaign_sha,
                    "analysis_sha256": analysis_sha,
                }
            )
        )
        temporary.rename(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"status": analysis["status"], "analysis_sha256": analysis_sha},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
