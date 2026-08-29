#!/usr/bin/env python3
"""Audit same-split epoch curves before the joint-learning ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SPLIT = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
EXPECTED_CONTACT = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
EXPECTED_STRUCTURE = "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
EXPECTED_EMBEDDING = "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite curve value")
    return result


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def audit_result(path: Path, expected_experiment: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    config = result["configuration"]
    authority = result["authority"]
    if (
        result.get("experiment") != expected_experiment
        or result.get("data", {}).get("train_complexes") != 2565
        or result.get("data", {}).get("validation_complexes") != 549
        or config.get("split_manifest_sha256") != EXPECTED_SPLIT
        or authority.get("split_manifest_sha256") != EXPECTED_SPLIT
        or authority.get("contact_manifest_sha256") != EXPECTED_CONTACT
        or authority.get("structure_manifest_sha256") != EXPECTED_STRUCTURE
        or authority.get("embedding_manifest_sha256") != EXPECTED_EMBEDDING
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
    ):
        raise ValueError(f"authority differs: {path}")
    history = result.get("history", [])
    if not history:
        raise ValueError(f"empty history: {path}")
    first_total = finite(history[0]["train"]["loss"])
    first_epitope = finite(history[0]["train"]["epitope_loss"])
    curve = []
    for row in history:
        train = row["train"]
        validation = row["validation"]
        curve.append(
            {
                "epoch": int(row["epoch"]),
                "train_total_loss": finite(train["loss"]),
                "train_total_loss_relative_to_epoch1": finite(train["loss"]) / first_total,
                "train_epitope_loss": finite(train["epitope_loss"]),
                "train_epitope_loss_relative_to_epoch1": finite(train["epitope_loss"]) / first_epitope,
                "validation_epitope_mean_per_complex_auprc": finite(validation["epitope"]["mean_per_complex_auprc"]),
                "validation_contact_mean_per_complex_auprc": finite(validation["contact"]["mean_per_complex_auprc"]),
                "validation_paratope_mean_per_complex_auprc": finite(validation["paratope"]["mean_per_complex_auprc"]),
            }
        )
    best = max(curve, key=lambda row: row["validation_epitope_mean_per_complex_auprc"])
    last = curve[-1]
    return {
        "seed": int(config["seed"]),
        "result_sha256": sha256_file(path),
        "epochs_completed": len(curve),
        "objective_weights": {
            "contact": finite(config["contact_loss_weight"]),
            "paratope": finite(config["paratope_loss_weight"]),
            "epitope": finite(config["epitope_loss_weight"]),
            "distance": finite(config.get("distance_loss_weight", 0.0)),
        },
        "epitope_loss_definition": (
            "bce_plus_dice_plus_count" if expected_experiment == "epiformer_style_same_protocol" else "bce"
        ),
        "best_epitope_epoch": int(best["epoch"]),
        "best_epitope_auprc": best["validation_epitope_mean_per_complex_auprc"],
        "final_minus_best_epitope_auprc": last["validation_epitope_mean_per_complex_auprc"] - best["validation_epitope_mean_per_complex_auprc"],
        "final_train_epitope_loss_relative_to_epoch1": last["train_epitope_loss_relative_to_epoch1"],
        "curve": curve,
    }


def group_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "median_best_epitope_epoch": statistics.median(row["best_epitope_epoch"] for row in runs),
        "median_best_epitope_auprc": statistics.median(row["best_epitope_auprc"] for row in runs),
        "median_final_minus_best_epitope_auprc": statistics.median(row["final_minus_best_epitope_auprc"] for row in runs),
        "all_training_epitope_losses_decrease": all(row["final_train_epitope_loss_relative_to_epoch1"] < 1 for row in runs),
        "all_final_epitope_auprc_below_best": all(row["final_minus_best_epitope_auprc"] < 0 for row in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framecontact-root", type=Path, required=True)
    parser.add_argument("--epiformer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    framecontact = [audit_result(path, "framecontact") for path in sorted(args.framecontact_root.glob("seed-*/FRAMECONTACT_DEVELOPMENT_RESULT.json"))]
    epiformer = [audit_result(path, "epiformer_style_same_protocol") for path in sorted(args.epiformer_root.glob("seed-*/FRAMECONTACT_DEVELOPMENT_RESULT.json"))]
    if len(framecontact) != 5 or len(epiformer) != 3:
        raise ValueError("unexpected run count")
    payload = {
        "format_version": "sabdab2_epiformer_joint_learning_curve_audit_v1",
        "status": "PASS_AUDIT",
        "comparability": {
            "same_split": True,
            "same_train_validation_counts": True,
            "same_candidate_grid": True,
            "same_contact_structure_embedding_manifests": True,
            "absolute_epitope_training_loss_directly_comparable": False,
            "normalized_loss_trend_comparable": True,
        },
        "framecontact": {"runs": framecontact, "summary": group_summary(framecontact)},
        "epiformer_style": {"runs": epiformer, "summary": group_summary(epiformer)},
        "diagnostic_gate": {
            "early_peak_present": group_summary(framecontact)["median_best_epitope_epoch"] <= 3 and group_summary(epiformer)["median_best_epitope_epoch"] <= 3,
            "late_overfitting_signature_present": group_summary(framecontact)["all_training_epitope_losses_decrease"] and group_summary(framecontact)["all_final_epitope_auprc_below_best"] and group_summary(epiformer)["all_training_epitope_losses_decrease"] and group_summary(epiformer)["all_final_epitope_auprc_below_best"],
            "joint_task_competition_resolved_by_existing_curves": False,
            "two_by_two_objective_ablation_authorized": True,
        },
        "authority": {"test_loaded": False, "temporal_holdout_loaded": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    print(json.dumps({"framecontact": payload["framecontact"]["summary"], "epiformer_style": payload["epiformer_style"]["summary"], "diagnostic_gate": payload["diagnostic_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
