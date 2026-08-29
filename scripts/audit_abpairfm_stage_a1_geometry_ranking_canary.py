#!/usr/bin/env python3
"""Audit the frozen train-only Stage-A1 geometry-ranking diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_FORMAT = "abpairfm_stage_a1_geometry_ranking_canary_v17"
EXPECTED_HOLDOUT_PDB_SHA256 = (
    "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("Stage-A1 geometry-ranking metric is nonfinite")
    return result


def _relative_reduction(loss: dict[str, Any]) -> float:
    if loss.get("active_update_count") != 80:
        raise RuntimeError("Stage-A1 geometry-ranking active-update count differs")
    first = _finite(loss["first_20_active_update_mean"])
    final = _finite(loss["final_20_active_update_mean"])
    if first <= 0.0:
        raise RuntimeError("Stage-A1 geometry-ranking initial loss is invalid")
    return (first - final) / first


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists():
        raise FileExistsError("Stage-A1 geometry-ranking audit is create-only")
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    checkpoint = result.get("checkpoint", {})
    provenance = checkpoint.get("provenance_hashes", {})
    if (
        result.get("status") != "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED"
        or result.get("training_protocol_format_version") != EXPECTED_PROTOCOL_FORMAT
        or result.get("source_sha256") != arguments.expected_source_sha256
        or result.get("test_access") is not False
        or result.get("authorization", {}).get("development_evaluation") is not False
        or checkpoint.get("content_sha256") != checkpoint.get("restore_content_sha256")
        or provenance.get("protocol") != arguments.expected_protocol_sha256
        or provenance.get("source") != arguments.expected_source_sha256
    ):
        raise RuntimeError("Stage-A1 geometry-ranking result authority differs")

    holdout = result.get("internal_holdout", {})
    receipt = holdout.get("receipt", {})
    before = holdout.get("before_training", {})
    after = holdout.get("after_training", {})
    if (
        holdout.get("test_access") is not False
        or receipt.get("test_access") is not False
        or receipt.get("holdout_record_count") != 512
        or receipt.get("holdout_pdb_count") != 511
        or receipt.get("training_record_count") != 4455
        or receipt.get("pdb_inventory_sha256") != EXPECTED_HOLDOUT_PDB_SHA256
        or before.get("complex_count") != 512
        or after.get("complex_count") != 512
        or before.get("pair_count") != after.get("pair_count")
    ):
        raise RuntimeError("Stage-A1 geometry-ranking holdout authority differs")

    losses = result["loss"]
    train_contact_reduction = _relative_reduction(losses["contact_loss"])
    train_ranking_reduction = _relative_reduction(losses["ranking_loss"])
    train_distance_reduction = _relative_reduction(losses["distance_loss"])
    auprc_gain = _finite(after["macro_sampled_contact_auprc"]) - _finite(
        before["macro_sampled_contact_auprc"]
    )
    accuracy_gain = _finite(
        after["pairwise_positive_over_negative_accuracy"]
    ) - _finite(before["pairwise_positive_over_negative_accuracy"])
    balanced_bce_reduction = _finite(before["balanced_contact_bce"]) - _finite(
        after["balanced_contact_bce"]
    )
    holdout_ranking_reduction = _finite(before["ranking_softplus"]) - _finite(
        after["ranking_softplus"]
    )
    gates = {
        "checkpoint_restore_exact": True,
        "distance_final_below_uniform_8_bin_cross_entropy": (
            _finite(losses["distance_loss"]["final_20_active_update_mean"])
            < math.log(8.0)
        ),
        "holdout_balanced_bce_decreased": balanced_bce_reduction > 0.0,
        "holdout_macro_sampled_contact_auprc_gain_at_least_0p02": (
            auprc_gain >= 0.02
        ),
        "holdout_pairwise_accuracy_gain_at_least_0p02": accuracy_gain >= 0.02,
        "holdout_ranking_softplus_decreased": holdout_ranking_reduction > 0.0,
        "train_contact_relative_reduction_at_least_0p02": (
            train_contact_reduction >= 0.02
        ),
        "train_distance_relative_reduction_at_least_0p02": (
            train_distance_reduction >= 0.02
        ),
        "train_ranking_relative_reduction_at_least_0p02": (
            train_ranking_reduction >= 0.02
        ),
    }
    passed = all(gates.values())
    payload = {
        "authorization": {
            "development_evaluation": passed,
            "test_access": False,
        },
        "format_version": "abpairfm_stage_a1_geometry_ranking_canary_audit_v1",
        "gates": gates,
        "holdout": {
            "after_training": after,
            "before_training": before,
            "balanced_bce_absolute_reduction": balanced_bce_reduction,
            "macro_sampled_contact_auprc_absolute_gain": auprc_gain,
            "pairwise_accuracy_absolute_gain": accuracy_gain,
            "ranking_softplus_absolute_reduction": holdout_ranking_reduction,
            "receipt": receipt,
        },
        "result_sha256": _sha256(arguments.result),
        "status": (
            "PASS_STAGE_A1_GEOMETRY_RANKING_CANARY_GATE"
            if passed
            else "FAIL_STAGE_A1_GEOMETRY_RANKING_CANARY_GATE"
        ),
        "test_access": False,
        "train_loss_relative_reduction": {
            "contact": train_contact_reduction,
            "distance": train_distance_reduction,
            "ranking": train_ranking_reduction,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
