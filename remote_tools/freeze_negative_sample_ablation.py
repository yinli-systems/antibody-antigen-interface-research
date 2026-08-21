#!/usr/bin/env python3
"""Freeze development-only no-partner-negative protocols for AsEP and SAbDab2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_json(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze(
    *,
    dataset: str,
    base_protocol: Path,
    trainer: Path,
    output_root: Path,
) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    base_raw = base_protocol.read_bytes()
    base = json.loads(base_raw)
    protocol = copy.deepcopy(base)
    protocol["schema_version"] = (
        f"framecontact_{dataset}_partner_negative_ablation_protocol_v1"
    )
    protocol["frozen_on"] = "2026-08-21"
    protocol["protocol_status"] = (
        "development_only_frozen_before_no_negative_training"
    )
    protocol["claim_boundary"] = (
        f"{dataset} development train/validation comparison of K=1 standard "
        "wrong-partner specificity supervision against an otherwise matched "
        "zero-specificity-loss arm. Test and external-test roles remain sealed."
    )
    objective = protocol.setdefault("training_objective", {})
    objective["loss_weights"] = {
        "contact": 1.0,
        "epitope": 0.25,
        "paratope": 0.25,
        "specificity": 0.0,
    }
    objective["specificity"] = (
        "disabled: authenticated wrong-partner examples follow the matched "
        "execution path but have exactly zero objective weight"
    )
    objective["negative_samples_contribute_to_gradient"] = False
    objective["specificity_margin"] = 0.05
    objective["score_fraction"] = 0.01
    protocol["negative_sample_ablation"] = {
        "arm": "no_partner_negative",
        "base_protocol_file_sha256": hashlib.sha256(base_raw).hexdigest(),
        "base_protocol_path": str(base_protocol),
        "comparison_arm": {
            "specificity_loss_weight": 0.25,
            "specificity_negative_count": 1,
        },
        "frozen_fields": [
            "dataset authority",
            "split",
            "model",
            "seed",
            "optimizer",
            "learning rate",
            "epoch limit",
            "early stopping",
            "partner panels",
            "calibration",
        ],
        "no_negative_arm": {
            "specificity_loss_weight": 0.0,
            "wrong_partner_gradient_contribution": False,
        },
        "test_loaded": False,
        "external_test_loaded": False,
    }
    output_root.mkdir(parents=True)
    protocol_path = output_root / f"framecontact_{dataset}_no_partner_negative_v1.json"
    publish_json(protocol_path, protocol)
    receipt = {
        "format_version": "framecontact_partner_negative_ablation_receipt_v1",
        "dataset": dataset,
        "protocol_file_sha256": sha256_file(protocol_path),
        "trainer_file_sha256": sha256_file(trainer),
        "base_protocol_file_sha256": hashlib.sha256(base_raw).hexdigest(),
        "base_protocol_path": str(base_protocol),
        "specificity_loss_weight": 0.0,
        "test_loaded": False,
        "external_test_loaded": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    publish_json(output_root / "PROTOCOL_RECEIPT.json", receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("asep", "sabdab2"), required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    freeze(
        dataset=args.dataset,
        base_protocol=args.base_protocol,
        trainer=args.trainer,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
