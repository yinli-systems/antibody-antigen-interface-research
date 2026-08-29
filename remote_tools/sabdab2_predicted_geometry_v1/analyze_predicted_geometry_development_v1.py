#!/usr/bin/env python3
"""Audit and summarize the predicted-geometry development sensitivity study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


SEEDS = (17, 29, 43)
TASKS = ("contact", "epitope", "paratope")
BOUND_REFERENCE_SHA256 = (
    "f4368275e38d5134e667e23470702e987bb1a301b1c89aee5e82718d39911824"
)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite metric")
    return number


def selected_metrics(result: dict[str, Any]) -> dict[str, float]:
    selected = result["selected_checkpoint_validation"]
    return {
        task: finite(selected[task]["mean_per_complex_auprc"])
        for task in TASKS
    }


def verify_run(
    root: Path,
    *,
    seed: int,
    protocol_sha256: str,
    protocol_receipt_sha256: str,
    source_sha256: str,
    split_sha256: str,
    structure_sha256: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = root / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    if (
        result.get("experiment") != "epiformer_style_same_protocol"
        or result.get("configuration", {}).get("seed") != seed
        or authority.get("protocol_file_sha256") != protocol_sha256
        or authority.get("protocol_receipt_sha256") != protocol_receipt_sha256
        or authority.get("source_manifest_sha256") != source_sha256
        or authority.get("split_manifest_sha256") != split_sha256
        or authority.get("structure_manifest_sha256") != structure_sha256
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or receipt.get("seed") != seed
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256")
        != sha256_file(checkpoint_path)
        or receipt.get("selected_epoch") != result["selection"]["best_epoch"]
    ):
        raise ValueError(f"run authority differs: {root}")
    return selected_metrics(result), {
        "run_root": str(root),
        "result_sha256": sha256_file(result_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "receipt_sha256": sha256_file(receipt_path),
        "selected_epoch": int(result["selection"]["best_epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--bound-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.receipt_output.exists() or args.receipt_output.is_symlink():
        raise FileExistsError(args.receipt_output)
    if sha256_file(args.bound_reference) != BOUND_REFERENCE_SHA256:
        raise ValueError("bound-complex development reference SHA-256 differs")

    authority_root = args.campaign_root / "authority"
    checksums_path = authority_root / "AUTHORITY_SHA256SUMS.txt"
    campaign_path = (
        authority_root / "CAMPAIGN_PREDICTED_GEOMETRY_DEVELOPMENT_V1.json"
    )
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "FROZEN_BEFORE_TRAINING"
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or campaign.get("selection_or_tuning") is not False
        or campaign.get("seeds") != list(SEEDS)
    ):
        raise ValueError("campaign authority differs")
    for raw_line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, raw_path = raw_line.split(maxsplit=1)
        path = Path(raw_path.strip())
        if sha256_file(path) != expected:
            raise ValueError(f"authority checksum differs: {path}")

    split_sha256 = campaign["development_split_view_sha256"]
    structure_sha256 = campaign["structure_manifest_sha256"]
    arm_metrics: dict[str, dict[int, dict[str, float]]] = {}
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "edge_topk2_supervised"):
        arm_authority = campaign["arms"][arm]
        arm_metrics[arm] = {}
        bindings[arm] = {}
        for seed in SEEDS:
            run_root = args.campaign_root / "runs" / arm / f"seed-{seed}"
            metrics, binding = verify_run(
                run_root,
                seed=seed,
                protocol_sha256=arm_authority["protocol_sha256"],
                protocol_receipt_sha256=arm_authority["receipt_sha256"],
                source_sha256=arm_authority["source_manifest_sha256"],
                split_sha256=split_sha256,
                structure_sha256=structure_sha256,
            )
            arm_metrics[arm][seed] = metrics
            bindings[arm][str(seed)] = binding

    routed = {
        seed: {
            "contact": arm_metrics["baseline"][seed]["contact"],
            "epitope": arm_metrics["edge_topk2_supervised"][seed]["epitope"],
            "paratope": arm_metrics["baseline"][seed]["paratope"],
        }
        for seed in SEEDS
    }
    predicted_median = {
        task: float(median(routed[seed][task] for seed in SEEDS))
        for task in TASKS
    }
    bound = json.loads(args.bound_reference.read_text(encoding="utf-8"))
    bound_by_seed = {
        int(seed): {task: finite(value) for task, value in values.items()}
        for seed, values in bound["routed_metrics_by_seed"].items()
    }
    delta_by_seed = {
        seed: {
            task: routed[seed][task] - bound_by_seed[seed][task]
            for task in TASKS
        }
        for seed in SEEDS
    }
    delta_median = {
        task: float(median(delta_by_seed[seed][task] for seed in SEEDS))
        for task in TASKS
    }
    payload = {
        "format_version": "sabdab2_predicted_geometry_development_analysis_v1",
        "status": "PASS_PREDICTED_GEOMETRY_DEVELOPMENT_SENSITIVITY",
        "campaign_sha256": sha256_file(campaign_path),
        "structure_manifest_sha256": structure_sha256,
        "development_split_view_sha256": split_sha256,
        "bound_reference_analysis_sha256": BOUND_REFERENCE_SHA256,
        "run_bindings": bindings,
        "baseline_metrics_by_seed": {
            str(seed): arm_metrics["baseline"][seed] for seed in SEEDS
        },
        "topk2_expert_metrics_by_seed": {
            str(seed): arm_metrics["edge_topk2_supervised"][seed]
            for seed in SEEDS
        },
        "routed_metrics_by_seed": {str(seed): routed[seed] for seed in SEEDS},
        "routed_median_metrics": predicted_median,
        "predicted_minus_bound_routed_by_seed": {
            str(seed): delta_by_seed[seed] for seed in SEEDS
        },
        "predicted_minus_bound_routed_median": delta_median,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_study": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_predicted_geometry_analysis_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": sha256_file(args.output),
        "campaign_sha256": payload["campaign_sha256"],
        "all_six_runs_verified": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "routed_median_metrics": predicted_median,
                "predicted_minus_bound_routed_median": delta_median,
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": sha256_file(args.receipt_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
