#!/usr/bin/env python3
"""Apply the frozen seed-17 topology-routing promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-result", type=Path, required=True)
    parser.add_argument("--routed-result-sha256", required=True)
    parser.add_argument("--dense-result", type=Path, required=True)
    parser.add_argument("--dense-result-sha256", required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-audit-sha256", required=True)
    parser.add_argument("--gpu-smoke", type=Path, required=True)
    parser.add_argument("--gpu-smoke-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_authenticated(path: Path, expected: str, label: str) -> dict[str, Any]:
    authenticate_file(path, expected, label)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    validation = result["selected_checkpoint_validation"]
    return {
        "selection_metric": float(validation["selection_metric"]),
        "contact_macro_auprc": float(
            validation["contact"]["mean_per_complex_auprc"]
        ),
        "contact_pooled_auprc": float(validation["contact"]["auprc"]),
        "contact_p_at_10": float(
            validation["contact"]["ranked_pair_macro"]["10"]["mean_precision"]
        ),
        "epitope_macro_auprc": float(
            validation["epitope"]["mean_per_complex_auprc"]
        ),
        "epitope_pooled_auprc": float(validation["epitope"]["auprc"]),
        "paratope_macro_auprc": float(
            validation["paratope"]["mean_per_complex_auprc"]
        ),
        "paratope_pooled_auprc": float(validation["paratope"]["auprc"]),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite pilot gate: {arguments.output}")
    protocol = _load_authenticated(
        arguments.protocol, arguments.protocol_sha256, "topology protocol"
    )
    receipt = _load_authenticated(
        arguments.protocol_receipt,
        arguments.protocol_receipt_sha256,
        "topology protocol receipt",
    )
    if (
        receipt.get("protocol_file_sha256") != arguments.protocol_sha256
        or receipt.get("frozen_test_authorized") is not False
        or protocol.get("model_selection", {}).get("pilot_seed") != 17
    ):
        raise ValueError("Topology pilot authority is not eligible")
    routed = _load_authenticated(
        arguments.routed_result,
        arguments.routed_result_sha256,
        "routed development result",
    )
    dense = _load_authenticated(
        arguments.dense_result,
        arguments.dense_result_sha256,
        "dense development result",
    )
    audit = _load_authenticated(
        arguments.gate_audit,
        arguments.gate_audit_sha256,
        "routed gate audit",
    )
    smoke = _load_authenticated(
        arguments.gpu_smoke,
        arguments.gpu_smoke_sha256,
        "routed GPU smoke",
    )
    for name, result, experiment in (
        ("routed", routed, "topology_routed_multiscale"),
        ("dense", dense, "sequence_dense"),
    ):
        if (
            result.get("experiment") != experiment
            or result.get("configuration", {}).get("seed") != 17
            or result.get("authority", {}).get("test_loaded") is not False
            or result.get("authority", {}).get("external_test_loaded") is not False
        ):
            raise ValueError(f"{name} result has an ineligible authority")
    if (
        audit.get("authority", {}).get("test_loaded") is not False
        or audit.get("aggregate", {}).get("nonzero_fraction_gt_1e_8", 0.0) <= 0.0
        or smoke.get("checks", {}).get("forward_backward_finite") is not True
        or smoke.get("checks", {}).get("official_test_loaded") is not False
        or max(smoke.get("independent_se3_invariance_max_abs", {}).values())
        > 2e-5
    ):
        raise ValueError("Numerical, routing, or test-sealing gate failed")

    routed_metrics = _metrics(routed)
    dense_metrics = _metrics(dense)
    deltas = {
        key: routed_metrics[key] - dense_metrics[key]
        for key in routed_metrics
    }
    improved_primary = (
        deltas["selection_metric"] > 0.0
        or deltas["contact_macro_auprc"] > 0.0
    )
    noninferior_primary = (
        deltas["selection_metric"] >= -0.01
        and deltas["contact_macro_auprc"] >= -0.01
    )
    advance = improved_primary and noninferior_primary
    payload = {
        "format_version": "topology_routing_seed17_pilot_gate_v1",
        "authority": {
            "protocol_file_sha256": arguments.protocol_sha256,
            "protocol_receipt_sha256": arguments.protocol_receipt_sha256,
            "routed_result_sha256": arguments.routed_result_sha256,
            "dense_result_sha256": arguments.dense_result_sha256,
            "gate_audit_sha256": arguments.gate_audit_sha256,
            "gpu_smoke_sha256": arguments.gpu_smoke_sha256,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "seed": 17,
        "routed": routed_metrics,
        "sequence_dense": dense_metrics,
        "routed_minus_dense": deltas,
        "registered_gate": {
            "improved_selection_or_contact_macro": improved_primary,
            "selection_and_contact_macro_within_minus_0_01": (
                noninferior_primary
            ),
            "runtime_numerical_and_router_checks_passed": True,
        },
        "advance_to_five_seed": advance,
        "decision": (
            "advance_to_registered_five_seed_expansion"
            if advance
            else "stop_topology_routing_after_seed17"
        ),
        "claim_boundary": (
            "Matched-seed AsEP development-only pilot decision. It is not a "
            "multi-seed result, test result, external result, generalization "
            "result, benchmark result, or state-of-the-art claim."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "decision": payload["decision"],
                "routed_minus_dense": deltas,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
