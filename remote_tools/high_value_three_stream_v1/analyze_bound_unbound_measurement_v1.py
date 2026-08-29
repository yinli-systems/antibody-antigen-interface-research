#!/usr/bin/env python3
"""Audit and summarize the frozen three-seed bound/unbound measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("baseline_current", "edge_topk2_supervised")
SEEDS = (17, 29, 43)
CONDITIONS = (
    "bound_bound",
    "unbound_ab_bound_ag",
    "bound_ab_unbound_ag",
    "unbound_unbound",
)
TASK_ROUTE = {
    "contact": "baseline_current",
    "paratope": "baseline_current",
    "epitope": "edge_topk2_supervised",
}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260828


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def bootstrap_case_paired_delta(
    per_seed_delta: dict[int, dict[str, float]],
) -> dict[str, object]:
    case_sets = [set(values) for values in per_seed_delta.values()]
    if not case_sets or any(cases != case_sets[0] for cases in case_sets[1:]):
        raise ValueError("eligible cases differ across seeds")
    cases = sorted(case_sets[0])
    case_seed_mean = np.asarray(
        [np.mean([per_seed_delta[seed][case] for seed in SEEDS]) for case in cases],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(cases), size=len(cases))
        draws[index] = float(case_seed_mean[sample].mean())
    return {
        "point_estimate": float(case_seed_mean.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "case_count": len(cases),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_handling": "mean paired per-case delta across the three fixed trained seeds before case resampling",
        "uncertainty_boundary": "conditional on the three fixed trained seeds; does not include retraining uncertainty",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.protocol, args.protocol_sha256, "protocol")
    data_manifest_path = args.data_root / "BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json"
    require_sha(data_manifest_path, args.data_manifest_sha256, "data manifest")
    protocol = json.loads(args.protocol.read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    if (
        protocol.get("format_version") != "framecontact_bound_unbound_measurement_protocol_v1"
        or protocol.get("execution", {}).get("analyzer_sha256") != sha256_file(Path(__file__))
        or data_manifest.get("status") != "PASS_BOUND_UNBOUND_MEASUREMENT_DATA_MATERIALIZED"
        or data_manifest.get("counts", {}).get("cases") != 52
    ):
        raise ValueError("analysis authority differs")

    components: dict[str, dict[int, dict[str, object]]] = {arm: {} for arm in ARMS}
    component_bindings: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            result_path = args.components_root / arm / f"seed-{seed}" / "RESULT.json"
            receipt_path = args.components_root / arm / f"seed-{seed}" / "RECEIPT.json"
            result = json.loads(result_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            if (
                result.get("status") != "PASS_BOUND_UNBOUND_COMPONENT_MEASUREMENT"
                or result.get("arm") != arm
                or result.get("seed") != seed
                or result.get("audit", {}).get("protocol_sha256") != args.protocol_sha256
                or result.get("audit", {}).get("data_manifest_sha256") != args.data_manifest_sha256
                or result.get("audit", {}).get("selection_performed") is not False
                or result.get("audit", {}).get("old_exposed_test_access") is not False
                or result.get("audit", {}).get("temporal_holdout_access") is not False
                or receipt.get("result_sha256") != sha256_file(result_path)
                or receipt.get("selection_performed") is not False
            ):
                raise ValueError(f"component audit failed: {arm}/seed-{seed}")
            if set(result["conditions"]) != set(CONDITIONS):
                raise ValueError(f"condition set differs: {arm}/seed-{seed}")
            components[arm][seed] = result
            component_bindings[arm][str(seed)] = {
                "result_sha256": sha256_file(result_path),
                "receipt_sha256": sha256_file(receipt_path),
                "checkpoint_sha256": result["audit"]["checkpoint_sha256"],
                "model_state_sha256": result["audit"]["model_state_sha256"],
            }

    routed_metrics: dict[str, dict[str, dict[str, object]]] = {}
    paired_delta: dict[str, dict[str, dict[str, float]]] = {}
    bootstrap: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        routed_metrics[condition] = {}
        paired_delta[condition] = {}
        bootstrap[condition] = {}
        for task, arm in TASK_ROUTE.items():
            routed_metrics[condition][task] = {}
            per_seed_case_delta: dict[int, dict[str, float]] = {}
            seed_mpc_deltas: list[float] = []
            for seed in SEEDS:
                current = components[arm][seed]["conditions"][condition][task]
                reference = components[arm][seed]["conditions"]["bound_bound"][task]
                routed_metrics[condition][task][str(seed)] = {
                    key: current[key]
                    for key in (
                        "average_precision",
                        "auroc",
                        "brier",
                        "ece_15",
                        "precision",
                        "recall",
                        "f1",
                        "mcc",
                        "mean_per_eligible_complex_average_precision",
                        "eligible_complex_count",
                        "total_complex_count",
                    )
                }
                current_per_case = current["per_complex_average_precision"]
                reference_per_case = reference["per_complex_average_precision"]
                if set(current_per_case) != set(reference_per_case):
                    raise ValueError(f"eligible cases differ within {condition}/{task}/seed-{seed}")
                per_seed_case_delta[seed] = {
                    case: float(current_per_case[case] - reference_per_case[case])
                    for case in current_per_case
                }
                seed_mpc_deltas.append(
                    float(
                        current["mean_per_eligible_complex_average_precision"]
                        - reference["mean_per_eligible_complex_average_precision"]
                    )
                )
            paired_delta[condition][task] = {
                "by_seed": {str(seed): value for seed, value in zip(SEEDS, seed_mpc_deltas)},
                "median_across_seeds": float(np.median(seed_mpc_deltas)),
            }
            bootstrap[condition][task] = bootstrap_case_paired_delta(per_seed_case_delta)

    payload = {
        "format_version": "framecontact_bound_unbound_measurement_result_v1",
        "status": "PASS_BOUND_UNBOUND_MEASUREMENT_COMPLETE",
        "protocol_sha256": args.protocol_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "task_routing": TASK_ROUTE,
        "reference_condition": "bound_bound",
        "conditions": {
            "bound_bound": "bound antibody and bound antigen intra-chain geometry",
            "unbound_ab_bound_ag": "unbound antibody and bound antigen intra-chain geometry",
            "bound_ab_unbound_ag": "bound antibody and unbound antigen intra-chain geometry",
            "unbound_unbound": "unbound antibody and unbound antigen intra-chain geometry",
        },
        "routed_metrics_by_condition_task_seed": routed_metrics,
        "paired_mean_per_eligible_complex_ap_delta_vs_bound_bound": paired_delta,
        "case_paired_bootstrap_delta_vs_bound_bound": bootstrap,
        "component_bindings": component_bindings,
        "claim_boundary": {
            "robustness_measurement_only": True,
            "model_or_checkpoint_selection_performed": False,
            "calibration_refit": False,
            "same_bound_derived_labels_across_conditions": True,
            "old_exposed_test_access": False,
            "temporal_holdout_access": False,
            "dataset_license_absent_at_pinned_commit": True,
            "redistribution_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "framecontact_bound_unbound_measurement_result_receipt_v1",
        "status": payload["status"],
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "selection_performed": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
