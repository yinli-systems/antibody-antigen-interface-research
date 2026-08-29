#!/usr/bin/env python3
"""Audit the one-seed epitope-head continuation and apply its frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


INITIAL_CHECKPOINT_SHA256 = (
    "5c2448c28a335dac0dcfa03d97a63be7331d664c3181c83a38bb563ca61b9a22"
)
METRIC_NAMES = ("contact", "epitope", "paratope")


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def metric(result: dict, name: str) -> float:
    return float(
        result["selected_checkpoint_validation"][name]["mean_per_complex_auprc"]
    )


def compare_checkpoint_state(
    base_path: Path, candidate_path: Path
) -> dict[str, object]:
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    base_state = base["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    if set(base_state) != set(candidate_state):
        raise ValueError("candidate checkpoint state-key set differs")
    frozen_names = [
        name for name in base_state if not name.startswith("epitope_head.")
    ]
    changed_frozen = [
        name
        for name in frozen_names
        if not torch.equal(base_state[name], candidate_state[name])
    ]
    changed_head = [
        name
        for name in base_state
        if name.startswith("epitope_head.")
        and not torch.equal(base_state[name], candidate_state[name])
    ]
    return {
        "state_key_count": len(base_state),
        "frozen_tensor_count": len(frozen_names),
        "changed_frozen_tensor_names": changed_frozen,
        "changed_epitope_head_tensor_names": changed_head,
        "frozen_tensors_byte_identical": not changed_frozen,
        "epitope_head_changed": bool(changed_head),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--base-result-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, digest, label in (
        (args.protocol, args.protocol_sha256, "protocol"),
        (args.protocol_receipt, args.protocol_receipt_sha256, "protocol receipt"),
        (args.base_result, args.base_result_sha256, "base result"),
        (args.base_checkpoint, INITIAL_CHECKPOINT_SHA256, "base checkpoint"),
    ):
        authenticate(path, digest, label)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_receipt = json.loads(args.protocol_receipt.read_text(encoding="utf-8"))
    base = json.loads(args.base_result.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_result.read_text(encoding="utf-8"))
    receipt = json.loads(args.candidate_receipt.read_text(encoding="utf-8"))
    if protocol_receipt.get("protocol_file_sha256") != args.protocol_sha256:
        raise ValueError("protocol receipt does not authenticate protocol")
    if receipt.get("result_sha256") != sha256_file(args.candidate_result):
        raise ValueError("candidate receipt does not authenticate result")
    if receipt.get("selected_checkpoint_sha256") != sha256_file(
        args.candidate_checkpoint
    ):
        raise ValueError("candidate receipt does not authenticate checkpoint")
    candidate_authority = candidate.get("authority", {})
    model = candidate.get("model", {})
    if (
        protocol.get("protocol_status") != "frozen_before_training"
        or protocol.get("claim_boundary", {}).get("test_loaded") is not False
        or protocol.get("claim_boundary", {}).get("temporal_holdout_loaded")
        is not False
        or candidate_authority.get("test_loaded") is not False
        or candidate_authority.get("external_test_loaded") is not False
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or int(receipt.get("seed", -1)) != 17
        or model.get("training_policy") != "epitope_head_only"
        or model.get("initial_checkpoint_sha256") != INITIAL_CHECKPOINT_SHA256
        or model.get("trainable_parameter_names")
        != ["epitope_head.weight", "epitope_head.bias"]
        or int(model.get("trainable_parameter_count", -1)) != 129
        or model.get("frozen_state_unchanged") is not True
        or model.get("frozen_initial_state_sha256")
        != model.get("frozen_selected_state_sha256")
    ):
        raise ValueError("candidate continuation authority or invariants differ")

    base_metrics = {name: metric(base, name) for name in METRIC_NAMES}
    candidate_metrics = {name: metric(candidate, name) for name in METRIC_NAMES}
    deltas = {
        name: candidate_metrics[name] - base_metrics[name] for name in METRIC_NAMES
    }
    history_drifts = {
        name: max(
            abs(
                float(row["validation"][name]["mean_per_complex_auprc"])
                - base_metrics[name]
            )
            for row in candidate["history"]
        )
        for name in ("contact", "paratope")
    }
    checkpoint_audit = compare_checkpoint_state(
        args.base_checkpoint, args.candidate_checkpoint
    )
    gate = protocol["promotion_gate"]
    checks = {
        "epitope_delta_at_least_0_005": deltas["epitope"]
        >= float(gate["minimum_epitope_mean_per_complex_auprc_delta"]),
        "selected_contact_metric_unchanged": abs(deltas["contact"])
        <= float(gate["maximum_absolute_contact_metric_drift"]),
        "selected_paratope_metric_unchanged": abs(deltas["paratope"])
        <= float(gate["maximum_absolute_paratope_metric_drift"]),
        "all_epoch_contact_metric_unchanged": history_drifts["contact"]
        <= float(gate["maximum_absolute_contact_metric_drift"]),
        "all_epoch_paratope_metric_unchanged": history_drifts["paratope"]
        <= float(gate["maximum_absolute_paratope_metric_drift"]),
        "frozen_tensors_byte_identical": checkpoint_audit[
            "frozen_tensors_byte_identical"
        ],
        "epitope_head_changed": checkpoint_audit["epitope_head_changed"],
    }
    passed = all(bool(value) for value in checks.values())
    result = {
        "format_version": "sabdab2_epitope_head_refinement_result_v1",
        "status": "PASS_PROMOTE_SEEDS_29_43" if passed else "FAIL_CLEAN",
        "seed": 17,
        "metrics": {
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "delta": deltas,
            "maximum_history_drift": history_drifts,
        },
        "checkpoint_audit": checkpoint_audit,
        "gate_checks": checks,
        "authorization": {
            "seeds_29_43_development_only": passed,
            "full_development": False,
            "old_test": False,
            "temporal_holdout": False,
        },
        "authority": {
            "protocol_sha256": args.protocol_sha256,
            "protocol_receipt_sha256": args.protocol_receipt_sha256,
            "base_result_sha256": args.base_result_sha256,
            "base_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
            "candidate_result_sha256": sha256_file(args.candidate_result),
            "candidate_receipt_sha256": sha256_file(args.candidate_receipt),
            "candidate_checkpoint_sha256": sha256_file(args.candidate_checkpoint),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "epitope_delta": deltas["epitope"],
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
