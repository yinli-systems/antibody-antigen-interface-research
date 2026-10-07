#!/usr/bin/env python3
"""Audit the frozen ABF-PairContact float32 zero-sum confirmation canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_PROTOCOL_FORMAT = "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20"
EXPECTED_ACTIVE_UPDATES = 160


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("PairContact float32 zero-sum audit metric is nonfinite")
    return number


def _relative_reduction(summary: dict) -> float:
    first = _finite(summary["first_20_active_update_mean"])
    final = _finite(summary["final_20_active_update_mean"])
    if summary.get("active_update_count") != EXPECTED_ACTIVE_UPDATES or first <= 0.0:
        raise RuntimeError("PairContact float32 zero-sum loss summary differs")
    return (first - final) / first


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists():
        raise FileExistsError("PairContact float32 zero-sum audit output is create-only")
    if _sha256(arguments.result) != arguments.expected_result_sha256:
        raise RuntimeError("PairContact float32 zero-sum result SHA differs")
    if _sha256(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("PairContact float32 zero-sum protocol SHA differs")
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    checkpoint = result.get("checkpoint", {})
    provenance = checkpoint.get("provenance_hashes", {})
    contact_protocol = protocol.get("loss", {}).get("contact", {})
    if (
        protocol.get("format_version") != EXPECTED_PROTOCOL_FORMAT
        or result.get("training_protocol_format_version") != EXPECTED_PROTOCOL_FORMAT
        or result.get("authorization") != protocol.get("authorization")
        or result.get("test_access") is not False
        or protocol.get("test_access") is not False
        or contact_protocol.get("interaction_residualization") != "endpoint_zero_sum"
        or not contact_protocol.get("centering", "").startswith(
            "analytic zero-sum gauge in float32"
        )
        or contact_protocol.get("centering_weight") != 0.0
        or contact_protocol.get("distance_label_manifest_permitted") is not False
        or contact_protocol.get("distance_bin_weight") != 0.0
        or protocol.get("optimization", {}).get("updates") != EXPECTED_ACTIVE_UPDATES
        or result.get("source_sha256") != arguments.expected_source_sha256
        or provenance.get("source") != arguments.expected_source_sha256
        or provenance.get("protocol") != arguments.expected_protocol_sha256
        or checkpoint.get("content_sha256") != checkpoint.get("restore_content_sha256")
    ):
        raise RuntimeError("PairContact float32 zero-sum result authority differs")

    holdout = result.get("internal_holdout", {})
    receipt = holdout.get("receipt", {})
    before = holdout.get("before_training", {})
    after = holdout.get("after_training", {})
    expected_holdout = protocol["internal_holdout"]
    if (
        holdout.get("test_access") is not False
        or receipt.get("test_access") is not False
        or receipt.get("holdout_record_count") != 512
        or receipt.get("holdout_pdb_count") != 508
        or receipt.get("excluded_pdb_count") != 1022
        or receipt.get("ordering_salt") != "v20_fp32_confirmatory_holdout"
        or receipt.get("pdb_inventory_sha256")
        != expected_holdout["pdb_inventory_sha256"]
        or before.get("pair_count") != after.get("pair_count")
        or before.get("complex_count") != after.get("complex_count")
        or before.get("row_matched_comparison_count")
        != after.get("row_matched_comparison_count")
        or before.get("column_matched_comparison_count")
        != after.get("column_matched_comparison_count")
    ):
        raise RuntimeError("PairContact float32 zero-sum holdout authority differs")

    losses = result["loss"]
    contact_reduction = _relative_reduction(losses["contact_loss"])
    ranking_reduction = _relative_reduction(losses["ranking_loss"])
    distance_values = losses["distance_loss"]
    if (
        _finite(distance_values["first_20_active_update_mean"]) != 0.0
        or _finite(distance_values["final_20_active_update_mean"]) != 0.0
    ):
        raise RuntimeError("PairContact float32 zero-sum canary used a distance loss")

    gains = {
        "column_matched_accuracy": _finite(after["column_matched_accuracy"])
        - _finite(before["column_matched_accuracy"]),
        "full_macro_sampled_contact_ap": _finite(
            after["full_macro_sampled_contact_ap"]
        )
        - _finite(before["full_macro_sampled_contact_ap"]),
        "interaction_macro_sampled_contact_ap": _finite(
            after["interaction_macro_sampled_contact_ap"]
        )
        - _finite(before["interaction_macro_sampled_contact_ap"]),
        "row_matched_accuracy": _finite(after["row_matched_accuracy"])
        - _finite(before["row_matched_accuracy"]),
    }
    gates = protocol["promotion_gates"]
    maximum_centering_error = max(
        _finite(before["centering_penalty"]),
        _finite(after["centering_penalty"]),
    )
    checks = {
        "checkpoint_restore_exact": True,
        "column_matched_accuracy_gain_at_least_0p02": gains[
            "column_matched_accuracy"
        ]
        >= gates["minimum_internal_holdout_column_matched_accuracy_absolute_gain"],
        "distance_loss_exactly_zero": True,
        "endpoint_centering_error_at_most_1e_8": maximum_centering_error
        <= gates["maximum_endpoint_centering_error"],
        "full_macro_ap_gain_at_least_0p01": gains[
            "full_macro_sampled_contact_ap"
        ]
        >= gates["minimum_internal_holdout_full_macro_ap_absolute_gain"],
        "interaction_macro_ap_gain_at_least_0p01": gains[
            "interaction_macro_sampled_contact_ap"
        ]
        >= gates["minimum_internal_holdout_interaction_macro_ap_absolute_gain"],
        "row_matched_accuracy_gain_at_least_0p02": gains["row_matched_accuracy"]
        >= gates["minimum_internal_holdout_row_matched_accuracy_absolute_gain"],
        "test_access_false": True,
        "tie_aware_ap_contract": protocol["evaluation"]["average_precision"].startswith(
            "non-interpolated AP evaluated only at distinct score thresholds"
        ),
        "train_contact_relative_reduction_at_least_0p01": contact_reduction
        >= gates["minimum_train_contact_relative_reduction"],
        "train_ranking_relative_reduction_at_least_0p02": ranking_reduction
        >= gates["minimum_train_ranking_relative_reduction"],
    }
    passed = all(checks.values())
    audit = {
        "checks": checks,
        "claim_boundary": protocol["claim_boundary"],
        "format_version": "abf_paircontact_zero_sum_fp32_canary_audit_v1",
        "holdout": {
            "after_training": after,
            "before_training": before,
            "gains": gains,
            "maximum_endpoint_centering_error": maximum_centering_error,
            "receipt": receipt,
        },
        "provenance": {
            "protocol_sha256": arguments.expected_protocol_sha256,
            "result_sha256": arguments.expected_result_sha256,
            "source_sha256": arguments.expected_source_sha256,
        },
        "status": (
            "PASS_PAIRCONTACT_ZERO_SUM_FP32_PRECISION_GATE"
            if passed
            else "FAIL_PAIRCONTACT_ZERO_SUM_FP32_PRECISION_GATE"
        ),
        "test_access": False,
        "train_relative_reduction": {
            "contact": contact_reduction,
            "ranking": ranking_reduction,
        },
    }
    arguments.output.parent.mkdir(parents=True)
    with arguments.output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, sort_keys=True))
    if not passed:
        raise RuntimeError("PairContact float32 zero-sum canary failed its frozen gate")


if __name__ == "__main__":
    main()
