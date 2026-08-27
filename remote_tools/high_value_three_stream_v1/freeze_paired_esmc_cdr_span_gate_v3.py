#!/usr/bin/env python3
"""Freeze exact raw-OAS coordinate recovery for the paired ESM-C CDR gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--devclean-source", type=Path, required=True)
    parser.add_argument("--raw-plan", type=Path, required=True)
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--source-access-receipt", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        "parent_protocol": "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea",
        "devclean_source": "d1628da898286d1de6833bcaa1998ade7b5715da815cbe247b59d68c79ec1b68",
        "raw_plan": "b07a53fc216047a05e5775fe718825db58992a8b1664f9b40d42eacdb602503d",
        "raw_summary": "5a74823cc58b5a318cd990d1f9850a3160edc46ae8f8f2f583ab2259e4c064a1",
        "normalizer": "a1bafe64fbee686b40a89cddcb4d5b9e18acb1d3600bd51be4f99dc708e71a17",
        "v1_result": "5e84c4194626b791a4653f0c735ecf6034a73506444b28ff117c5feeaa4e3c74",
        "v2_result": "d603c402dfae41f044d8124f3cfb790f0d8f97e2461612c4f2e3240c469a1431",
        "source_access_receipt": "f9c268ce42f66e9d45e23077f9cf8cebaaf35fa207f3fed3d909936ef2181518",
    }
    paths = {
        "parent_protocol": args.parent_protocol,
        "devclean_source": args.devclean_source,
        "raw_plan": args.raw_plan,
        "raw_summary": args.raw_summary,
        "normalizer": args.normalizer,
        "v1_result": args.v1_result,
        "v2_result": args.v2_result,
        "source_access_receipt": args.source_access_receipt,
    }
    for label, path in paths.items():
        require_sha(path, expected[label], label)

    raw_plan = json.loads(args.raw_plan.read_text())
    raw_summary = json.loads(args.raw_summary.read_text())
    source_access = json.loads(args.source_access_receipt.read_text())
    if (
        raw_plan.get("expected_data_unit_count") != 610
        or raw_plan.get("expected_sequence_count") != 3_087_576
        or raw_plan.get("test_access") is not False
        or raw_summary.get("status") != "MATERIALIZED_NOT_DEDUPLICATED_NOT_DECONTAMINATED"
        or raw_summary.get("counts", {}).get("total_bytes") != 3_541_352_247
        or raw_summary.get("remote_receipt_sha256")
        != "4e27e1ff59e849d22e02d086d88a96abfbc78726246d900c1644a5e708aebbfd"
    ):
        raise RuntimeError("raw OAS authority differs")
    if (
        source_access.get("status") != "DISCLOSED_NON_TASK_SOURCE_SCHEMA_ACCESS"
        or source_access.get("use_in_training_or_selection") is not False
        or source_access.get("task_labels_read") is not False
    ):
        raise RuntimeError("source split access disclosure differs")

    authority = {
        "format_version": "paired_esmc_cdr_span_gate_authority_v3",
        "protocol_status": "frozen_before_exact_coordinate_recovery",
        "supersedes": {
            "v1_result_sha256": expected["v1_result"],
            "v2_result_sha256": expected["v2_result"],
            "reason": (
                "The normalized corpus retained CDR strings but omitted the authenticated raw-OAS "
                "nucleotide coordinates. V1/V2 motif placement is replaced by deterministic "
                "recovery from the exact source row identified by row_hash."
            ),
        },
        "target_source": {
            "uri": str(args.devclean_source),
            "sha256": expected["devclean_source"],
            "expected_rows": 776_634,
            "row_identity": "sha256(study|Run|sequence_id_heavy|sequence_id_light)",
        },
        "raw_oas_authority": {
            "root": "/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/oas_paired_20260821/data_units",
            "plan_uri": str(args.raw_plan),
            "plan_sha256": expected["raw_plan"],
            "summary_uri": str(args.raw_summary),
            "summary_sha256": expected["raw_summary"],
            "remote_receipt_sha256": raw_summary["remote_receipt_sha256"],
            "expected_data_units": 610,
            "expected_source_rows": 3_087_576,
            "expected_total_bytes": 3_541_352_247,
            "normalizer_uri": str(args.normalizer),
            "normalizer_sha256": expected["normalizer"],
        },
        "coordinate_rule": {
            "source_coordinates": "AIRR/OAS 1-based inclusive nucleotide coordinates",
            "aligned_aa_start": "(cdr_start - v_sequence_start) / 3",
            "aligned_aa_end_exclusive": "(cdr_end - v_sequence_start + 1) / 3",
            "requirements": [
                "all source coordinates are finite integers",
                "both nucleotide offsets are divisible by three",
                "gap-stripped sequence_alignment_aa exactly equals the normalized chain",
                "gap-stripped aligned slice exactly equals the normalized CDR string",
                "converted ungapped span slices the same CDR string",
                "CDR1, CDR2, and CDR3 spans are ordered and non-overlapping on each chain",
            ],
            "maximum_model_input_residues_including_separator": 2046,
        },
        "split_rule": {
            "group_key": "sha256(heavy_sequence + NUL + light_sequence)",
            "validation_if": "uint64_be(pair_sha256[0:16]) mod 100 < 2",
            "train_if": "otherwise",
            "checkpoint_selection": False,
        },
        "frozen_gate": {
            "minimum_eligible_rows": 500_000,
            "minimum_eligible_fraction": 0.99,
            "minimum_validation_rows": 5_000,
            "require_exact_row_hash_join": True,
            "require_pair_group_disjoint": True,
            "require_row_hash_unique": True,
            "require_source_row_accounting": True,
        },
        "implementation": {
            "builder_uri": str(args.builder),
            "builder_sha256": sha256_file(args.builder),
            "launcher_uri": str(args.launcher),
            "launcher_sha256": sha256_file(args.launcher),
        },
        "source_split_access_disclosure": {
            "receipt_uri": str(args.source_access_receipt),
            "receipt_sha256": expected["source_access_receipt"],
            "abcdr_validation_or_test_rows_permitted": False,
            "abcdr_official_train_rows_used_by_this_recovery": False,
        },
        "claim_boundary": {
            "development_only": True,
            "local_adaptation_training_authorized_before_gate": False,
            "production_training_authorized": False,
            "sabdab2_old_test_accessed": False,
            "sabdab2_temporal_holdout_accessed": False,
            "task_labels_read": False,
        },
    }
    atomic_write(args.output, authority)
    print(json.dumps({"authority_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
