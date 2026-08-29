#!/usr/bin/env python3
"""Freeze the exact AIRR query-coordinate CDR span recovery gate."""

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
        handle.write(canonical_bytes(value)); handle.flush(); os.fsync(handle.fileno())
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
    parser.add_argument("--source-access-receipt", type=Path, required=True)
    parser.add_argument("--v3-result", type=Path, required=True)
    parser.add_argument("--v4-result", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "parent_protocol": "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea",
        "devclean_source": "d1628da898286d1de6833bcaa1998ade7b5715da815cbe247b59d68c79ec1b68",
        "raw_plan": "b07a53fc216047a05e5775fe718825db58992a8b1664f9b40d42eacdb602503d",
        "raw_summary": "5a74823cc58b5a318cd990d1f9850a3160edc46ae8f8f2f583ab2259e4c064a1",
        "source_access_receipt": "f9c268ce42f66e9d45e23077f9cf8cebaaf35fa207f3fed3d909936ef2181518",
        "v3_result": "40d3d44010682ed6db0daaeaa03d3ce2ea74d0ec1d1f1d6ebe2834c2ac61b91f",
        "v4_result": "39f06c207f10ce9ac65c40ab80c48ce19de7dcf2e2bff1fffa8c7fda5d96c564",
    }
    paths = {
        "parent_protocol": args.parent_protocol, "devclean_source": args.devclean_source,
        "raw_plan": args.raw_plan, "raw_summary": args.raw_summary,
        "source_access_receipt": args.source_access_receipt,
        "v3_result": args.v3_result, "v4_result": args.v4_result,
    }
    for label, path in paths.items():
        require_sha(path, expected[label], label)
    v3, v4 = json.loads(args.v3_result.read_text()), json.loads(args.v4_result.read_text())
    access = json.loads(args.source_access_receipt.read_text())
    if (
        not str(v3.get("status", "")).startswith("FAIL_CLOSED")
        or not str(v4.get("status", "")).startswith("FAIL_CLOSED")
        or access.get("task_labels_read") is not False
        or access.get("use_in_training_or_selection") is not False
    ):
        raise RuntimeError("superseded failure or source-access disclosure differs")
    authority = {
        "format_version": "paired_esmc_cdr_span_gate_authority_v5",
        "protocol_status": "frozen_before_airr_query_coordinate_recovery",
        "research_question": (
            "Can exact AIRR query-sequence coordinates and a chain-consistent query reading frame "
            "recover at least 99% of development-clean paired CDR spans without motif ambiguity, "
            "coordinate rounding, or released AbCDR validation/test members?"
        ),
        "parent_protocol": {"uri": str(args.parent_protocol), "sha256": expected["parent_protocol"]},
        "supersedes_failures": {
            "v3_result_sha256": expected["v3_result"],
            "v4_result_sha256": expected["v4_result"],
            "reason": (
                "AIRR defines cdr*_start/end in the full query sequence, not relative to v_sequence_start. "
                "V3 used the wrong origin; V4 did not use the authenticated query reading frame."
            ),
        },
        "target_source": {
            "uri": str(args.devclean_source), "sha256": expected["devclean_source"],
            "expected_rows": 776_634, "row_identity": "sha256(study|Run|sequence_id_heavy|sequence_id_light)",
        },
        "raw_oas_authority": {
            "root": "/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/oas_paired_20260821/data_units",
            "plan_uri": str(args.raw_plan), "plan_sha256": expected["raw_plan"],
            "summary_uri": str(args.raw_summary), "summary_sha256": expected["raw_summary"],
            "expected_data_units": 610, "expected_source_rows": 3_087_576,
            "expected_total_bytes": 3_541_352_247,
        },
        "airr_coordinate_authority": {
            "uri": "https://docs.airr-community.org/en/v2.0.0/datarep/rearrangements.html",
            "retrieved_utc_date": "2026-08-26",
            "definition": "cdr*_start/end are positions in query sequence using 1-based closed intervals",
        },
        "coordinate_rule": {
            "query_sequence_field": "sequence_{heavy,light}",
            "normalized_chain_field": "gap-stripped sequence_alignment_aa_{heavy,light}",
            "reading_frame": (
                "The common value f in {0,1,2} of (cdr_start-1) mod 3 and cdr_end mod 3 "
                "over CDR1/2/3 on one chain."
            ),
            "normalized_chain_anchor": (
                "Translate the full query sequence from f and require the complete normalized chain "
                "to occur exactly once; let its zero-based amino-acid offset be o."
            ),
            "half_open_span": (
                "start=(cdr_start-1-f)/3-o; end=(cdr_end-f)/3-o"
            ),
            "requirements": [
                "all coordinates are finite integers and all six endpoints have one common frame per chain",
                "the complete normalized chain occurs exactly once in the frame-specific query translation",
                "all six recovered half-open spans exactly slice their authenticated CDR amino-acid strings",
                "CDR1, CDR2, and CDR3 are ordered and non-overlapping on each chain",
                "no nearest-coordinate rounding and no motif-placement tie breaking",
            ],
            "maximum_model_input_residues_including_separator": 2046,
        },
        "split_rule": {
            "group_key": "sha256(heavy_sequence + NUL + light_sequence)",
            "validation_if": "uint64_be(pair_sha256[0:16]) mod 100 < 2", "train_if": "otherwise",
            "checkpoint_selection": False,
        },
        "frozen_gate": {
            "minimum_eligible_rows": 500_000, "minimum_eligible_fraction": 0.99,
            "minimum_validation_rows": 5_000, "require_exact_row_hash_join": True,
            "require_pair_group_disjoint": True, "require_row_hash_unique": True,
            "require_source_row_accounting": True,
        },
        "implementation": {
            "builder_uri": str(args.builder), "builder_sha256": sha256_file(args.builder),
            "launcher_uri": str(args.launcher), "launcher_sha256": sha256_file(args.launcher),
        },
        "source_split_access_disclosure": {
            "receipt_uri": str(args.source_access_receipt), "receipt_sha256": expected["source_access_receipt"],
            "abcdr_official_train_rows_used": False, "abcdr_validation_or_test_rows_used": False,
        },
        "claim_boundary": {
            "development_only": True, "local_adaptation_training_authorized_before_gate": False,
            "production_training_authorized": False, "sabdab2_old_test_accessed": False,
            "sabdab2_temporal_holdout_accessed": False, "task_labels_read": False,
        },
    }
    atomic_write(args.output, authority)
    print(json.dumps({"authority_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
