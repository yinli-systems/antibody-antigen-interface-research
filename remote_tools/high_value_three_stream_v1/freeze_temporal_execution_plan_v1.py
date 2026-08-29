#!/usr/bin/env python3
"""Freeze the temporal measurement code and gates before label-free screening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


MINIMUM_DOUBLE_COLD_INSTANCES = 12
MINIMUM_DOUBLE_COLD_UNIQUE_PDBS = 10


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


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol-sha256", required=True)
    parser.add_argument("--label-free-script", type=Path, required=True)
    parser.add_argument("--post-gate-freezer", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.parent_protocol, args.parent_protocol_sha256, "parent protocol")
    parent = json.loads(args.parent_protocol.read_text())
    if (
        parent.get("format_version") != "framecontact_temporal_double_cold_protocol_v1"
        or parent.get("status") != "FROZEN_BEFORE_STRUCTURE_OR_LABEL_ACCESS"
        or parent.get("selection", {}).get("allowed") is not False
        or parent.get("authorization", {}).get("temporal_opening_count_maximum") != 1
    ):
        raise ValueError("parent temporal authority differs")
    if parent.get("label_free_gate", {}).get("script_sha256") != sha256_file(
        args.label_free_script
    ):
        raise ValueError("label-free script differs from parent authority")
    payload = {
        "format_version": "framecontact_temporal_execution_plan_v1",
        "status": "FROZEN_BEFORE_LABEL_FREE_SCREEN_COMPLETION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": str(args.parent_protocol),
        "parent_protocol_sha256": args.parent_protocol_sha256,
        "gate": {
            "minimum_double_cold_instances": MINIMUM_DOUBLE_COLD_INSTANCES,
            "minimum_double_cold_unique_pdbs": MINIMUM_DOUBLE_COLD_UNIQUE_PDBS,
            "thresholds_may_not_be_relaxed_after_label_free_screen": True,
            "task_labels_or_cross_chain_distances_seen_when_frozen": False,
        },
        "contact_definition": {
            "positive": "minimum inter-residue non-hydrogen atom distance <= 4.5 angstrom",
            "candidate_grid": "complete antibody-by-antigen Cartesian product exactly once",
            "residue_axis": "all observed amino-acid residues in declared chains; missing N/CA/C frames retained with an invalid-frame mask",
            "labels_are_pair_contacts": True,
            "paratope_and_epitope_labels": "strict any-axis marginals of the pair-contact matrix",
        },
        "measurement": {
            "task_route": {
                "contact": "baseline_current",
                "paratope": "baseline_current",
                "epitope": "edge_topk2_supervised",
            },
            "seeds": [17, 29, 43],
            "primary": "median across seeds of mean-per-eligible-complex non-interpolated Average Precision",
            "secondary": [
                "pooled_average_precision",
                "pooled_auroc",
                "brier",
                "ece_15",
                "precision",
                "recall",
                "f1",
                "mcc",
            ],
            "zero_positive_policy": "retain in pooled metrics; exclude only from per-complex AP denominator and report denominator",
            "bootstrap": {
                "unit": "unique PDB",
                "replicates": 10000,
                "seed": 20260827,
                "seed_handling": "mean per-complex AP over the three fixed seeds before equal-PDB block resampling",
                "conditional_on_fixed_trained_seeds": True,
            },
            "calibration_refit": False,
            "checkpoint_or_model_selection": False,
        },
        "execution": {
            "partition": "gpu_4090",
            "python": "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10",
            "dependency_site_packages": "/ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages",
            "label_free_script_sha256": sha256_file(args.label_free_script),
            "post_gate_freezer_sha256": sha256_file(args.post_gate_freezer),
            "materializer_sha256": sha256_file(args.materializer),
            "evaluator_sha256": sha256_file(args.evaluator),
            "analyzer_sha256": sha256_file(args.analyzer),
        },
        "opening": {
            "maximum": 1,
            "begins_when_cross_chain_contact_labels_are_first_computed": True,
            "results_may_not_feed_back_into_development": True,
            "old_exposed_test_may_not_be_accessed": True,
        },
    }
    atomic_write(args.output, canonical_bytes(payload))
    receipt = {
        "format_version": "framecontact_temporal_execution_plan_receipt_v1",
        "status": payload["status"],
        "plan_sha256": sha256_file(args.output),
        "parent_protocol_sha256": args.parent_protocol_sha256,
        "frozen_before_label_free_screen_completion": True,
        "task_labels_read": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
