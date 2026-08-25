#!/usr/bin/env python3
"""Audit the frozen single-seed EpiFormer top-2 development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE_RESULT_SHA256 = "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184"
PRIOR_RESULT_SHA256 = "491c27569027c2761e79a7a91059d8df791f441b7f50357bb8180db2e819360d"
SOURCE_MANIFEST_SHA256 = "757373282bec0340f3f0171b7471a881fbd4f854a088bc81b72315e9bba24157"


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


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric")
    return result


def metrics(result: dict[str, Any]) -> dict[str, Any]:
    selected = result["selected_checkpoint_validation"]
    return {
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "contact_mean_per_complex_auprc": finite(
            selected["contact"]["mean_per_complex_auprc"]
        ),
        "epitope_mean_per_complex_auprc": finite(
            selected["epitope"]["mean_per_complex_auprc"]
        ),
        "paratope_mean_per_complex_auprc": finite(
            selected["paratope"]["mean_per_complex_auprc"]
        ),
        "epochs_completed": len(result["history"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    authority_path = args.run_root / "authority" / "CAMPAIGN_AUTHORITY_V1.json"
    authority = json.loads(authority_path.read_text())
    protocol_path = args.run_root / "authority" / "PROTOCOL_EPIFORMER_TOP2_V1.json"
    protocol_receipt_path = (
        args.run_root / "authority" / "PROTOCOL_EPIFORMER_TOP2_V1_RECEIPT.json"
    )
    protocol = json.loads(protocol_path.read_text())
    protocol_receipt = json.loads(protocol_receipt_path.read_text())
    if (
        authority.get("test_loaded") is not False
        or authority.get("temporal_holdout_loaded") is not False
        or protocol.get("protocol_status") != "frozen_before_training"
        or protocol.get("arm") != "epiformer_top2"
        or protocol_receipt.get("frozen_before_training") is not True
        or protocol_receipt.get("protocol_file_sha256")
        != sha256_file(protocol_path)
        or authority.get("protocol_sha256") != sha256_file(protocol_path)
        or authority.get("protocol_receipt_sha256")
        != sha256_file(protocol_receipt_path)
        or authority.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("campaign authority differs")

    result_path = (
        args.run_root
        / "epiformer_top2"
        / "seed-17"
        / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    )
    receipt_path = args.run_root / "epiformer_top2" / "seed-17" / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    result_sha = sha256_file(result_path)
    if (
        receipt.get("result_sha256") != result_sha
        or receipt.get("test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or result.get("authority", {}).get("source_manifest_sha256")
        != SOURCE_MANIFEST_SHA256
        or result.get("authority", {}).get("protocol_file_sha256")
        != sha256_file(protocol_path)
    ):
        raise ValueError("run authority differs")

    baseline_path = args.baseline_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    prior_path = args.prior_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    authenticate(baseline_path, BASELINE_RESULT_SHA256, "baseline result")
    authenticate(prior_path, PRIOR_RESULT_SHA256, "prior candidate result")
    baseline_result = json.loads(baseline_path.read_text())
    prior_result = json.loads(prior_path.read_text())
    for label, item in (("baseline", baseline_result), ("prior", prior_result)):
        if (
            item.get("authority", {}).get("test_loaded") is not False
            or item.get("authority", {}).get("external_test_loaded") is not False
        ):
            raise ValueError(f"{label} authority differs")

    candidate = metrics(result)
    baseline = metrics(baseline_result)
    prior = metrics(prior_result)
    deltas = {
        "contact_vs_control": candidate["contact_mean_per_complex_auprc"]
        - baseline["contact_mean_per_complex_auprc"],
        "epitope_vs_control": candidate["epitope_mean_per_complex_auprc"]
        - baseline["epitope_mean_per_complex_auprc"],
        "epitope_vs_prior_candidate": candidate["epitope_mean_per_complex_auprc"]
        - prior["epitope_mean_per_complex_auprc"],
        "paratope_vs_control": candidate["paratope_mean_per_complex_auprc"]
        - baseline["paratope_mean_per_complex_auprc"],
    }
    gate = {
        "epitope_delta_vs_control": deltas["epitope_vs_control"] >= 0.0075,
        "epitope_regression_vs_prior_candidate": deltas[
            "epitope_vs_prior_candidate"
        ]
        >= -0.005,
        "contact_no_regression": deltas["contact_vs_control"] >= -0.005,
        "paratope_no_regression": deltas["paratope_vs_control"] >= -0.01,
    }
    passes = all(gate.values())
    payload = {
        "format_version": "sabdab2_epiformer_top2_screen_result_v1",
        "status": "PASS_AUDIT_PROMOTE" if passes else "PASS_AUDIT_NO_PROMOTION",
        "authority": {
            "campaign_authority_sha256": sha256_file(authority_path),
            "protocol_sha256": sha256_file(protocol_path),
            "protocol_receipt_sha256": sha256_file(protocol_receipt_path),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "runs": {
            "frozen_control": {**baseline, "result_sha256": BASELINE_RESULT_SHA256},
            "prior_epitope_candidate": {
                **prior,
                "result_sha256": PRIOR_RESULT_SHA256,
            },
            "epiformer_top2": {
                **candidate,
                "result_sha256": result_sha,
                "receipt_sha256": sha256_file(receipt_path),
            },
        },
        "deltas": deltas,
        "gate": gate,
        "passes": passes,
        "next_gate": {
            "winner_seeds_29_and_43_authorized": passes,
            "full_2565_549_development_authorized": False,
            "test_or_temporal_holdout_authorized": False,
        },
    }
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_bytes(canonical_bytes(payload))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
