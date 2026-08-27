#!/usr/bin/env python3
"""Analyze frozen base ESM-C versus published CDR-ESM-C development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


ARMS = ("base_esmc600m", "published_abcdr_esmc")
SEEDS = (17, 29, 43)
TASKS = ("contact", "epitope", "paratope")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("nonfinite development metric")
    return result


def audit_run(
    run_root: Path,
    arm: str,
    seed: int,
    protocol_sha: str,
    protocol_receipt_sha: str,
    embedding_sha: str,
) -> dict[str, object]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    authority = result.get("authority", {})
    if (
        result.get("format_version") != "framecontact_same_protocol_adapter_development_v2"
        or result.get("experiment") != "epiformer_style_same_protocol"
        or result.get("configuration", {}).get("seed") != seed
        or result.get("data", {}).get("train_complexes") != 2565
        or result.get("data", {}).get("validation_complexes") != 549
        or authority.get("embedding_manifest_sha256") != embedding_sha
        or authority.get("protocol_file_sha256") != protocol_sha
        or authority.get("protocol_receipt_sha256") != protocol_receipt_sha
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or receipt.get("seed") != seed
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise RuntimeError(f"run authority differs: {arm} seed {seed}")
    selected = result["selected_checkpoint_validation"]
    metrics = {}
    for task in TASKS:
        values = selected[task]
        metrics[task] = {
            key: finite(values[key])
            for key in (
                "mean_per_complex_auprc", "auprc", "auroc", "precision", "recall",
                "f1", "mcc", "brier", "ece_15",
            )
        }
    return {
        "arm": arm,
        "seed": seed,
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "epochs_completed": int(result["selection"]["epochs_completed"]),
        "metrics": metrics,
        "partner_selection_metric": finite(selected["selection_metric"]),
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
        "test_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.campaign_root / "authority/PROTOCOL_ESMC_REPRESENTATION_DOWNSTREAM_V1.json"
    protocol_receipt_path = args.campaign_root / "authority/PROTOCOL_ESMC_REPRESENTATION_DOWNSTREAM_V1_RECEIPT.json"
    if sha256_file(protocol_path) != args.protocol_sha256 or sha256_file(protocol_receipt_path) != args.protocol_receipt_sha256:
        raise RuntimeError("frozen protocol authority differs")
    protocol = json.loads(protocol_path.read_text())
    protocol_receipt = json.loads(protocol_receipt_path.read_text())
    if (
        protocol_receipt.get("protocol_file_sha256") != args.protocol_sha256
        or protocol.get("protocol_status") != "frozen_before_same_protocol_development_training"
        or protocol.get("claim_boundary", {}).get("old_test_loaded") is not False
        or protocol.get("claim_boundary", {}).get("temporal_holdout_loaded") is not False
    ):
        raise RuntimeError("protocol contract differs")
    embedding_shas = {
        arm: protocol["representation_arms"][arm]["sha256"] for arm in ARMS
    }
    runs: dict[str, dict[str, dict[str, object]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            runs[arm][str(seed)] = audit_run(
                args.campaign_root / f"runs/{arm}/seed-{seed}",
                arm,
                seed,
                args.protocol_sha256,
                args.protocol_receipt_sha256,
                embedding_shas[arm],
            )

    deltas: dict[str, dict[str, float]] = {}
    medians: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for task in TASKS:
        for arm in ARMS:
            medians[arm][task] = statistics.median(
                runs[arm][str(seed)]["metrics"][task]["mean_per_complex_auprc"]
                for seed in SEEDS
            )
        deltas[task] = {
            str(seed): (
                runs["published_abcdr_esmc"][str(seed)]["metrics"][task]["mean_per_complex_auprc"]
                - runs["base_esmc600m"][str(seed)]["metrics"][task]["mean_per_complex_auprc"]
            )
            for seed in SEEDS
        }
        deltas[task]["median"] = statistics.median(deltas[task][str(seed)] for seed in SEEDS)
    gate = protocol["frozen_analysis_gate"]
    checks = {
        "minimum_positive_contact_seeds": sum(deltas["contact"][str(seed)] > 0 for seed in SEEDS) >= gate["minimum_positive_contact_seeds"],
        "minimum_median_contact_delta": deltas["contact"]["median"] > gate["minimum_median_contact_delta"],
        "maximum_median_epitope_regression": deltas["epitope"]["median"] >= -gate["maximum_median_epitope_regression"],
        "maximum_median_paratope_regression": deltas["paratope"]["median"] >= -gate["maximum_median_paratope_regression"],
        "all_authorities_verified": True,
        "test_not_loaded": True,
    }
    passed = all(checks.values())
    result = {
        "format_version": "sabdab2_esmc_representation_downstream_analysis_v1",
        "status": "PASS_PUBLISHED_CDR_POSITIVE_CONTROL_DEVELOPMENT_GATE" if passed else "FAIL_PUBLISHED_CDR_POSITIVE_CONTROL_DEVELOPMENT_GATE",
        "protocol_sha256": args.protocol_sha256,
        "protocol_receipt_sha256": args.protocol_receipt_sha256,
        "runs": runs,
        "median_mean_per_complex_average_precision": medians,
        "paired_seed_delta_published_cdr_minus_base": deltas,
        "checks": checks,
        "claim_boundary": {
            "development_only": True,
            "validation_used_for_checkpoint_selection": True,
            "published_checkpoint_external_positive_control": True,
            "local_cdr_pretraining_claim": False,
            "independent_test": False,
            "test_loaded": False,
        },
        "authorization": {
            "published_cdr_representation_supported_for_followup": passed,
            "local_cdr_pretraining_claim_authorized": False,
            "test_authorized": False,
        },
    }
    atomic_write(args.output, result)
    receipt = {
        "format_version": "sabdab2_esmc_representation_downstream_analysis_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "published_cdr_representation_supported_for_followup": passed,
        "local_cdr_pretraining_claim_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
