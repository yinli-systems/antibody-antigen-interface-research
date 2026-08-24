#!/usr/bin/env python3
"""Audit paired full-development results and freeze the temporal-holdout gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


SEEDS = (17, 29, 43)
SOURCE_SHA256 = "ca31451b4be85ee1e07045609a41ed3c90789971368191383c28cbf4251b581e"
JOINT_SHA256 = "d081d1752acf0e94757b248141a1c8861b76fab66a25fcaca89580fbee768323"
FULL_SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
PROMOTION_SHA256 = "370b1f4f9b0aec6ce86dc3664489b63f9b8f44381ea1338e043e80b578a1f292"
PROTOCOLS = {
    "baseline": (
        "cb5c6c7569a26ed86f3dbc2b0167ecd0260f8fe6bd7de0ad4bde94bc995d49ef",
        "e64c57f4181d56fde7dbecf8858e655b57f7ecfdb8764010be618bc576876275",
    ),
    "energy_balanced": (
        "90d0844b23c268c2b595c873405222503ae63e870bdd4584a80e23001d988944",
        "e78eaf54b7068ffc3d2757f10aabbd6464346c220285f95d11dbbf18432949b4",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite full-development value")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(arguments.output)

    arms: dict[str, list[dict[str, Any]]] = {}
    for arm, (protocol_sha, receipt_sha) in PROTOCOLS.items():
        expected_joint = JOINT_SHA256 if arm == "energy_balanced" else None
        records = []
        for seed in SEEDS:
            root = arguments.run_root / arm / f"seed-{seed}"
            result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            receipt_path = root / "RUN_RECEIPT.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            result_sha = _sha256(result_path)
            authority = result["authority"]
            selected = result["selected_checkpoint_validation"]
            if (
                result.get("data", {}).get("train_complexes") != 2565
                or result.get("data", {}).get("validation_complexes") != 549
                or result.get("configuration", {}).get("seed") != seed
                or result.get("selection", {}).get("primary_metric")
                != "epitope_mean_per_complex_auprc"
                or authority.get("protocol_file_sha256") != protocol_sha
                or authority.get("protocol_receipt_sha256") != receipt_sha
                or authority.get("source_manifest_sha256") != SOURCE_SHA256
                or authority.get("split_manifest_sha256") != FULL_SPLIT_SHA256
                or authority.get("joint_embedding_manifest_sha256") != expected_joint
                or authority.get("test_loaded") is not False
                or authority.get("external_test_loaded") is not False
                or receipt.get("result_sha256") != result_sha
                or receipt.get("test_loaded") is not False
                or receipt.get("external_test_loaded") is not False
            ):
                raise ValueError(f"{arm} seed {seed} authority differs")
            metrics = {}
            for task in ("contact", "epitope", "paratope"):
                metrics[task] = {
                    key: _finite(selected[task][key])
                    for key in ("mean_per_complex_auprc", "auprc", "brier", "ece_15")
                }
            best_primary = _finite(result["selection"]["best_validation_compatibility_margin"])
            if not math.isclose(
                best_primary,
                metrics["epitope"]["mean_per_complex_auprc"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{arm} seed {seed} selected checkpoint is inconsistent")
            records.append(
                {
                    "seed": seed,
                    "selected_epoch": int(result["selection"]["best_epoch"]),
                    "epochs_completed": int(result["selection"]["epochs_completed"]),
                    "result_sha256": result_sha,
                    "selected_checkpoint_sha256": receipt["selected_checkpoint_sha256"],
                    **metrics,
                }
            )
        arms[arm] = records

    paired = []
    for baseline, candidate in zip(arms["baseline"], arms["energy_balanced"], strict=True):
        if baseline["seed"] != candidate["seed"]:
            raise ValueError("Paired seeds differ")
        deltas = {
            task: candidate[task]["mean_per_complex_auprc"]
            - baseline[task]["mean_per_complex_auprc"]
            for task in ("contact", "epitope", "paratope")
        }
        paired.append({"seed": baseline["seed"], **deltas})
    aggregates = {}
    for arm, records in arms.items():
        aggregates[arm] = {
            f"{task}_mean_per_complex_auprc_median": median(
                row[task]["mean_per_complex_auprc"] for row in records
            )
            for task in ("contact", "epitope", "paratope")
        }
    median_deltas = {
        task: aggregates["energy_balanced"][f"{task}_mean_per_complex_auprc_median"]
        - aggregates["baseline"][f"{task}_mean_per_complex_auprc_median"]
        for task in ("contact", "epitope", "paratope")
    }
    checks = {
        "median_contact_strictly_improves": median_deltas["contact"] > 0.0,
        "median_epitope_strictly_improves": median_deltas["epitope"] > 0.0,
        "no_seed_contact_regresses_more_than_0_005": min(row["contact"] for row in paired)
        >= -0.005,
        "no_seed_epitope_regresses_more_than_0_01": min(row["epitope"] for row in paired)
        >= -0.01,
    }
    payload = {
        "format_version": "abpairfm_energy_balanced_paired_full_development_result_v1",
        "status": "PASS_AUDIT",
        "promotion_status": "PASS" if all(checks.values()) else "FAIL",
        "authority": {
            "promotion_manifest_sha256": PROMOTION_SHA256,
            "source_manifest_sha256": SOURCE_SHA256,
            "joint_manifest_sha256": JOINT_SHA256,
            "full_split_sha256": FULL_SPLIT_SHA256,
            "test_loaded": False,
        },
        "arms": arms,
        "aggregate": aggregates,
        "paired_candidate_minus_baseline": paired,
        "median_candidate_minus_baseline": median_deltas,
        "promotion_checks": checks,
        "temporal_holdout_authorized": all(checks.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical(payload))
    print(json.dumps({"promotion_status": payload["promotion_status"], "median_deltas": median_deltas}, sort_keys=True))


if __name__ == "__main__":
    main()
