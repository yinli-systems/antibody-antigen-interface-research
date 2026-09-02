#!/usr/bin/env python3
"""Freeze a systems-only recovery of the bilateral Top-K analyzer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any


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
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    source_protocol = args.campaign_root / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1.json"
    source_receipt = args.campaign_root / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1_RECEIPT.json"
    source_protocol_sha = sha256_file(source_protocol)
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != source_protocol_sha:
        raise ValueError("source protocol receipt differs")

    result_records: dict[str, Any] = {}
    for task in ("epitope", "paratope"):
        for seed in (17, 29, 43):
            root = args.campaign_root / f"runs/{task}/seed-{seed}"
            result_path = root / "BILATERAL_TOPK_READOUT_RESULT_V1.json"
            receipt_path = root / "BILATERAL_TOPK_READOUT_RESULT_V1_RECEIPT.json"
            result_sha = sha256_file(result_path)
            result_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if result_receipt["result_sha256"] != result_sha:
                raise ValueError(f"result receipt differs: {task}/{seed}")
            result_records[f"{task}:seed-{seed}"] = {
                "result_path": str(result_path),
                "result_sha256": result_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
            }

    protocol = {
        "format_version": "bilateral_topk_analysis_recovery_protocol_v1",
        "status": "FROZEN_BEFORE_SYSTEMS_ONLY_ANALYSIS_RECOVERY",
        "source_measurement_protocol_sha256": source_protocol_sha,
        "source_results": result_records,
        "execution": {
            "analyzer_sha256": sha256_file(args.analyzer),
            "launcher_sha256": sha256_file(args.launcher),
        },
        "failure": {
            "failed_analyzer_job": 1538231,
            "reason": "pooled average precision is stored under the metric key auprc rather than average_precision",
        },
        "invariants": {
            "model_or_checkpoint_changed": False,
            "data_or_labels_changed": False,
            "readout_scores_changed": False,
            "metric_values_changed": False,
            "result_selection_performed": False,
            "only_metric_key_mapping_changed": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        protocol_path = temporary / "BILATERAL_TOPK_ANALYSIS_RECOVERY_PROTOCOL_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        protocol_sha = sha256_file(protocol_path)
        (temporary / "BILATERAL_TOPK_ANALYSIS_RECOVERY_PROTOCOL_V1_RECEIPT.json").write_bytes(
            canonical_bytes(
                {
                    "format_version": "bilateral_topk_analysis_recovery_protocol_receipt_v1",
                    "status": protocol["status"],
                    "protocol_sha256": protocol_sha,
                    "source_measurement_protocol_sha256": source_protocol_sha,
                }
            )
        )
        temporary.rename(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {"status": protocol["status"], "protocol_sha256": protocol_sha},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
