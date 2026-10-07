#!/usr/bin/env python3
"""Aggregate the frozen bilateral Top-K readout audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


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
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    protocol_path = args.campaign_root / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1.json"
    receipt_path = args.campaign_root / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1_RECEIPT.json"
    protocol_raw = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    protocol = json.loads(protocol_raw)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["protocol_sha256"] != protocol_sha:
        raise ValueError("protocol receipt differs")

    recovery_raw = args.recovery_protocol.read_bytes()
    recovery_sha = hashlib.sha256(recovery_raw).hexdigest()
    recovery = json.loads(recovery_raw)
    recovery_receipt = json.loads(
        args.recovery_receipt.read_text(encoding="utf-8")
    )
    if recovery_receipt["protocol_sha256"] != recovery_sha:
        raise ValueError("analysis recovery receipt differs")
    if recovery["status"] != "FROZEN_BEFORE_SYSTEMS_ONLY_ANALYSIS_RECOVERY":
        raise ValueError("analysis recovery is not frozen")
    if recovery["source_measurement_protocol_sha256"] != protocol_sha:
        raise ValueError("analysis recovery references another measurement")
    if sha256_file(Path(__file__)) != recovery["execution"]["analyzer_sha256"]:
        raise ValueError("analysis recovery analyzer differs")

    results: dict[str, dict[int, Any]] = {"epitope": {}, "paratope": {}}
    result_hashes: dict[str, str] = {}
    for task in results:
        for seed in (17, 29, 43):
            root = args.campaign_root / f"runs/{task}/seed-{seed}"
            result_path = root / "BILATERAL_TOPK_READOUT_RESULT_V1.json"
            result_receipt_path = root / "BILATERAL_TOPK_READOUT_RESULT_V1_RECEIPT.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result_receipt = json.loads(
                result_receipt_path.read_text(encoding="utf-8")
            )
            result_sha = sha256_file(result_path)
            frozen_result = recovery["source_results"][f"{task}:seed-{seed}"]
            if result_sha != frozen_result["result_sha256"]:
                raise ValueError(f"recovery result hash differs: {task}/{seed}")
            if sha256_file(result_receipt_path) != frozen_result["receipt_sha256"]:
                raise ValueError(f"recovery receipt hash differs: {task}/{seed}")
            if result_receipt["result_sha256"] != result_sha:
                raise ValueError(f"result receipt differs: {task}/{seed}")
            if result["protocol_sha256"] != protocol_sha:
                raise ValueError(f"protocol differs: {task}/{seed}")
            if result["task"] != task or int(result["seed"]) != seed:
                raise ValueError(f"task or seed differs: {task}/{seed}")
            if result["audit"]["test_loaded"] is not False:
                raise ValueError(f"test accessed: {task}/{seed}")
            if result["audit"]["temporal_holdout_loaded"] is not False:
                raise ValueError(f"temporal accessed: {task}/{seed}")
            results[task][seed] = result
            result_hashes[f"{task}:seed-{seed}"] = result_sha

    metrics = (
        "mean_per_complex_ap",
        "auprc",
        "auroc",
        "f1",
        "mcc",
        "precision",
        "recall",
    )
    summary: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for task, seed_results in results.items():
        arm_names = list(seed_results[17]["metrics"])
        summary[task] = {}
        for arm in arm_names:
            summary[task][arm] = {
                metric: {
                    "by_seed": {
                        str(seed): float(seed_results[seed]["metrics"][arm][metric])
                        for seed in (17, 29, 43)
                    },
                    "median_across_seeds": float(
                        np.median(
                            [
                                seed_results[seed]["metrics"][arm][metric]
                                for seed in (17, 29, 43)
                            ]
                        )
                    ),
                }
                for metric in metrics
            }
        baseline = "native_baseline" if task == "paratope" else "topk2"
        deltas[task] = {}
        for arm in arm_names:
            if arm == baseline:
                continue
            by_seed = {}
            for seed in (17, 29, 43):
                values = np.asarray(
                    seed_results[seed]["per_complex"]["average_precision"][arm]
                )
                reference = np.asarray(
                    seed_results[seed]["per_complex"]["average_precision"][baseline]
                )
                by_seed[str(seed)] = float(np.mean(values - reference))
            deltas[task][f"{arm}-minus-{baseline}"] = {
                "mean_per_complex_ap_delta_by_seed": by_seed,
                "median_delta_across_seeds": float(np.median(list(by_seed.values()))),
            }

    for task in summary:
        for arm in summary[task].values():
            for metric in arm.values():
                if not math.isfinite(metric["median_across_seeds"]):
                    raise ValueError("non-finite summary metric")

    analysis = {
        "format_version": "bilateral_topk_readout_analysis_v1",
        "status": "PASS_BILATERAL_TOPK_FIXED_CHECKPOINT_READOUT_AUDIT",
        "protocol_sha256": protocol_sha,
        "analysis_recovery_protocol_sha256": recovery_sha,
        "result_sha256": result_hashes,
        "summary": summary,
        "paired_complex_deltas": deltas,
        "claim_boundary": protocol["claim_boundary"],
        "interpretation_constraints": [
            "This is a fixed-checkpoint readout sensitivity audit, not a from-scratch K training ablation.",
            "The epitope checkpoint was trained with K=2, so K other than 2 is post-hoc at that checkpoint.",
            "The paratope checkpoint was trained with the native node-plus-log-mean-exp readout, so all paratope Top-K arms are post-hoc.",
            "No K or final task route may be selected solely from this development audit.",
        ],
    }

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        result_path = temporary / "BILATERAL_TOPK_READOUT_ANALYSIS_V1.json"
        result_path.write_bytes(canonical_bytes(analysis))
        analysis_sha = sha256_file(result_path)
        (temporary / "BILATERAL_TOPK_READOUT_ANALYSIS_V1_RECEIPT.json").write_bytes(
            canonical_bytes(
                {
                    "format_version": "bilateral_topk_readout_analysis_receipt_v1",
                    "status": analysis["status"],
                    "protocol_sha256": protocol_sha,
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
            {
                "status": analysis["status"],
                "analysis_sha256": analysis_sha,
                "mean_per_complex_ap_medians": {
                    task: {
                        arm: values["mean_per_complex_ap"]["median_across_seeds"]
                        for arm, values in arms.items()
                    }
                    for task, arms in summary.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
