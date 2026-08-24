#!/usr/bin/env python3
"""Audit and apply the pre-registered gate to the ABPairFM residual-fusion screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


EXPECTED_SEEDS = (17, 29, 43)
EXPECTED_PROTOCOL_SHA256 = "57dde0903f934ade9fb709343291de7df5358b452d942ab2c61ceef87c31e813"
EXPECTED_RECEIPT_SHA256 = "35eb4bcb9cc9c8f2e796e3853876ef6ecb35571537942d9f80e2fb21b73d79e8"
EXPECTED_SOURCE_SHA256 = "bb1e832cd9adadc6d638733f1931720a9072d5f9ae7b6658377802cd2d8922d7"
EXPECTED_JOINT_SHA256 = "d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323"
EXPECTED_FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
EXPECTED_SCREEN_SPLIT_SHA256 = "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
REFERENCE_EPITOPE = 0.2532489185677912
REFERENCE_CONTACT = 0.024830761686624868


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Screen metric is not finite")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    seed_records = []
    for seed in EXPECTED_SEEDS:
        root = (
            arguments.run_root
            / "abpairfm_epiformer_residual_fusion_same_protocol"
            / f"seed-{seed}"
        )
        result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        receipt_path = root / "RUN_RECEIPT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        result_sha256 = _sha256(result_path)
        authority = result["authority"]
        if (
            result.get("experiment")
            != "abpairfm_epiformer_residual_fusion_same_protocol"
            or result.get("data", {}).get("feature_mode")
            != "abpairfm_joint_residual_concatenation_v1"
            or result.get("data", {}).get("train_complexes") != 512
            or result.get("data", {}).get("validation_complexes") != 128
            or result.get("configuration", {}).get("seed") != seed
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or authority.get("protocol_file_sha256") != EXPECTED_PROTOCOL_SHA256
            or authority.get("protocol_receipt_sha256") != EXPECTED_RECEIPT_SHA256
            or authority.get("source_manifest_sha256") != EXPECTED_SOURCE_SHA256
            or authority.get("joint_embedding_manifest_sha256") != EXPECTED_JOINT_SHA256
            or authority.get("joint_embedding_source_split_sha256")
            != EXPECTED_FULL_SPLIT_SHA256
            or authority.get("split_manifest_sha256") != EXPECTED_SCREEN_SPLIT_SHA256
            or receipt.get("test_loaded") is not False
            or receipt.get("external_test_loaded") is not False
            or receipt.get("seed") != seed
            or receipt.get("result_sha256") != result_sha256
        ):
            raise ValueError(f"Seed {seed} authority or receipt differs")

        metrics = result["selected_checkpoint_validation"]
        seed_records.append(
            {
                "seed": seed,
                "selected_epoch": int(result["selection"]["best_epoch"]),
                "result_sha256": result_sha256,
                "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                "contact": {
                    key: _finite(metrics["contact"][key])
                    for key in ("mean_per_complex_auprc", "auprc", "brier", "ece_15")
                },
                "epitope": {
                    key: _finite(metrics["epitope"][key])
                    for key in ("mean_per_complex_auprc", "auprc", "brier", "ece_15")
                },
                "paratope": {
                    key: _finite(metrics["paratope"][key])
                    for key in ("mean_per_complex_auprc", "auprc", "brier", "ece_15")
                },
            }
        )

    epitope_values = [x["epitope"]["mean_per_complex_auprc"] for x in seed_records]
    contact_values = [x["contact"]["mean_per_complex_auprc"] for x in seed_records]
    checks = {
        "median_epitope_strictly_exceeds_reference": median(epitope_values)
        > REFERENCE_EPITOPE,
        "median_contact_strictly_exceeds_reference": median(contact_values)
        > REFERENCE_CONTACT,
        "no_seed_epitope_regresses_more_than_0_01": min(epitope_values)
        >= REFERENCE_EPITOPE - 0.01,
        "no_seed_contact_regresses_more_than_0_005": min(contact_values)
        >= REFERENCE_CONTACT - 0.005,
    }
    payload = {
        "format_version": "abpairfm_joint_residual_same_protocol_screen_result_v2",
        "status": "PASS_AUDIT",
        "promotion_status": "PASS" if all(checks.values()) else "FAIL",
        "authority": {
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "protocol_receipt_sha256": EXPECTED_RECEIPT_SHA256,
            "source_manifest_sha256": EXPECTED_SOURCE_SHA256,
            "joint_embedding_manifest_sha256": EXPECTED_JOINT_SHA256,
            "full_split_sha256": EXPECTED_FULL_SPLIT_SHA256,
            "screen_split_sha256": EXPECTED_SCREEN_SPLIT_SHA256,
            "test_loaded": False,
        },
        "reference": {
            "name": "epiformer_style_same_protocol_seed_17",
            "epitope_mean_per_complex_auprc": REFERENCE_EPITOPE,
            "contact_mean_per_complex_auprc": REFERENCE_CONTACT,
        },
        "seeds": seed_records,
        "aggregate": {
            "epitope_mean_per_complex_auprc_median": median(epitope_values),
            "contact_mean_per_complex_auprc_median": median(contact_values),
            "epitope_values": epitope_values,
            "contact_values": contact_values,
        },
        "promotion_checks": checks,
    }
    arguments.output.write_bytes(_canonical(payload))
    print(json.dumps(payload["aggregate"], sort_keys=True))
    print(json.dumps({"promotion_status": payload["promotion_status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
