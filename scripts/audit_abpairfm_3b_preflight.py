#!/usr/bin/env python3
"""Fail-closed preflight audit for the ABPairFM 3B training campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _missing(blockers: list[str], condition: bool, code: str) -> None:
    if not condition:
        blockers.append(code)


def audit(
    protocol: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    """Return stage-specific blockers and derived authorizations."""
    if protocol.get("format_version") != "abpairfm_3b_pretraining_v1":
        raise RuntimeError("Unexpected ABPairFM protocol format")
    if inventory.get("format_version") != "abpairfm_3b_inventory_v1":
        raise RuntimeError("Unexpected ABPairFM inventory format")

    test_access = protocol["test_access"]
    observed_development = inventory["development"]
    if any(bool(value) for value in test_access.values()):
        raise RuntimeError("Protocol test access must remain false")
    if observed_development["test_bytes_accessed"]:
        raise RuntimeError("Inventory reports forbidden test-byte access")

    model_blockers: list[str] = []
    model = inventory["model"]
    runtime = inventory["runtime"]
    model_contract = protocol["model"]["architecture"]
    _missing(model_blockers, model["checkpoint_present"], "model_checkpoint_missing")
    _missing(
        model_blockers,
        _valid_sha256(model["checkpoint_sha256"]),
        "model_checkpoint_sha256_missing",
    )
    parameter_count = model["compatible_parameter_count"]
    _missing(
        model_blockers,
        isinstance(parameter_count, int)
        and model_contract["parameter_count_minimum"] <= parameter_count
        <= model_contract["parameter_count_maximum"],
        "compatible_3b_parameter_count_unproven",
    )
    _missing(
        model_blockers,
        model["pair_aware_model_code_present"],
        "pair_aware_model_code_missing",
    )
    _missing(model_blockers, model["fsdp2_runner_present"], "fsdp2_runner_missing")
    _missing(
        model_blockers,
        runtime["fixed_python"] == protocol["optimization"]["python"],
        "fixed_python_mismatch",
    )
    _missing(model_blockers, runtime["modules"]["torch"], "torch_missing")
    _missing(
        model_blockers,
        runtime["torch_fsdp2_fully_shard_present"],
        "torch_fsdp2_missing",
    )
    _missing(model_blockers, runtime["sdpa_present"], "torch_sdpa_missing")

    stage_a_blockers = deepcopy(model_blockers)
    ppi = inventory["ppi"]
    stage_a = protocol["stage_a_pair_interface"]
    count = ppi["post_dedup_training_examples"]
    _missing(
        stage_a_blockers,
        stage_a["example_count_minimum"] <= count <= stage_a["example_count_maximum"],
        "ppi_post_dedup_example_count_outside_frozen_range",
    )
    _missing(
        stage_a_blockers,
        ppi["normalized_training_examples"] >= count > 0,
        "ppi_normalized_count_missing_or_inconsistent",
    )
    _missing(
        stage_a_blockers,
        ppi["unique_biological_pair_count_reported"],
        "ppi_unique_biological_pair_count_missing",
    )
    _missing(
        stage_a_blockers,
        ppi["unique_interface_cluster_count_reported"],
        "ppi_unique_interface_cluster_count_missing",
    )
    for source_name in stage_a["required_sources"]:
        source_key = {
            "pinder_2024_02_train_metadata": "pinder_2024_02",
            "ppiref_zenodo_14845086": "ppiref_zenodo_14845086",
            "string_v12_high_confidence_physical": "string_v12_physical",
        }[source_name]
        source = ppi["sources"][source_key]
        raw_present = source.get("raw_materialized", source.get("index_materialized", False))
        _missing(stage_a_blockers, raw_present, f"{source_key}_raw_missing")
        _missing(stage_a_blockers, source["normalized"], f"{source_key}_not_normalized")
        _missing(
            stage_a_blockers,
            source["decontamination_complete"],
            f"{source_key}_decontamination_incomplete",
        )
        _missing(
            stage_a_blockers,
            source["training_use_authorized"],
            f"{source_key}_training_use_not_authorized",
        )
    pinder = ppi["sources"]["pinder_2024_02"]
    _missing(stage_a_blockers, pinder["metadata_materialized"], "pinder_metadata_missing")
    _missing(stage_a_blockers, pinder["train_val_rows_only"], "pinder_not_train_val_rows_only")
    if pinder["test_structure_bytes_accessed"]:
        raise RuntimeError("PINDER test structure bytes were accessed")
    if pinder["test_rows_used"]:
        raise RuntimeError("PINDER test rows were used")
    _missing(
        stage_a_blockers,
        ppi["sources"]["string_v12_physical"]["evidence_channel_filter_complete"],
        "string_v12_physical_evidence_channel_filter_incomplete",
    )

    stage_b_blockers: list[str] = []
    native = inventory["native_vh_vl"]
    stage_b = protocol["stage_b_cdr_specialization"]
    _missing(
        stage_b_blockers,
        native["admitted_native_pair_count"] >= stage_b["minimum_native_vh_vl_pairs"],
        "native_vh_vl_pair_count_below_5_5m",
    )
    for key in (
        "native_pair_evidence_complete",
        "deduplication_complete",
        "donor_study_clone_split_complete",
        "cdr_numbering_complete",
    ):
        _missing(stage_b_blockers, native[key], f"native_vh_vl_{key}_missing")

    development_blockers: list[str] = []
    for key in (
        "sabdab2_frozen_development_split",
        "asep_frozen_development_split",
        "independent_leakage_audit_passed",
        "evaluation_harness_audited",
        "test_sealed",
    ):
        _missing(development_blockers, observed_development[key], f"development_{key}_missing")

    systems_blockers: list[str] = []
    systems = inventory["systems"]
    for key in (
        "four_gpu_smoke_passed",
        "four_gpu_recovery_passed",
        "eight_gpu_fixed_work_passed",
    ):
        _missing(systems_blockers, systems[key], f"systems_{key}_missing")

    stage_a_gpu_smoke = not stage_a_blockers
    stage_a_production = stage_a_gpu_smoke and not systems_blockers
    stage_b_production = stage_a_production and not stage_b_blockers
    development_evaluation = stage_b_production and not development_blockers
    return {
        "authorization": {
            "development_evaluation": development_evaluation,
            "gpu_smoke": stage_a_gpu_smoke,
            "production_pretraining_stage_a": stage_a_production,
            "production_pretraining_stage_b": stage_b_production,
            "test_access": False,
        },
        "blockers": {
            "development": development_blockers,
            "model_and_stage_a_data": stage_a_blockers,
            "stage_b_data": stage_b_blockers,
            "systems": systems_blockers,
        },
        "claim_boundary": "Preflight authorization only; no training, quality, throughput, biological, or SOTA result is established by this audit.",
        "format_version": "abpairfm_3b_preflight_audit_v1",
        "status": "READY" if development_evaluation else "BLOCKED",
        "test_access": {
            "test_loaded": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    payload = audit(_load(args.protocol), _load(args.inventory))
    payload["inputs"] = {
        "inventory_sha256": _sha256(args.inventory),
        "protocol_sha256": _sha256(args.protocol),
    }
    data = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
    if payload["status"] != "READY" and not args.allow_blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
