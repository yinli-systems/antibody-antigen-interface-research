#!/usr/bin/env python3
"""Freeze V2 unique spans plus official AbCDR train-index rescue."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--official-train-audit", type=Path, required=True)
    parser.add_argument("--source-access-receipt", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, required=True)
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--v3-result", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "parent_protocol": "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea",
        "source": "d1628da898286d1de6833bcaa1998ade7b5715da815cbe247b59d68c79ec1b68",
        "official_archive": "86e89ff7854e1565f343b3377ca713dfc0ff0c79aa28fec99c4b7fecf163f977",
        "official_train_audit": "a902f9ec1c16bdce7a679ffc44417dd0dcc19956828427779da4adf518a19a62",
        "source_access_receipt": "f9c268ce42f66e9d45e23077f9cf8cebaaf35fa207f3fed3d909936ef2181518",
        "v1_result": "5e84c4194626b791a4653f0c735ecf6034a73506444b28ff117c5feeaa4e3c74",
        "v2_result": "d603c402dfae41f044d8124f3cfb790f0d8f97e2461612c4f2e3240c469a1431",
        "v3_result": "40d3d44010682ed6db0daaeaa03d3ce2ea74d0ec1d1f1d6ebe2834c2ac61b91f",
    }
    paths = {
        "parent_protocol": args.parent_protocol,
        "source": args.source,
        "official_archive": args.official_archive,
        "official_train_audit": args.official_train_audit,
        "source_access_receipt": args.source_access_receipt,
        "v1_result": args.v1_result,
        "v2_result": args.v2_result,
        "v3_result": args.v3_result,
    }
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"{name} SHA differs")
    audit = json.loads(args.official_train_audit.read_text())
    access = json.loads(args.source_access_receipt.read_text())
    if (
        audit.get("training_member") != "OAS_paired_set/train.csv"
        or audit.get("training_inventory", {}).get("row_count") != 1_577_499
        or audit.get("training_inventory", {}).get("unique_exact_pair_count") != 1_577_499
        or access.get("use_in_training_or_selection") is not False
        or access.get("task_labels_read") is not False
    ):
        raise RuntimeError("official train authority differs")
    authority = {
        "format_version": "paired_esmc_cdr_span_gate_authority_v4",
        "protocol_status": "frozen_before_official_train_index_rescue",
        "research_question": "Can ambiguous exact CDR-string placements be resolved by the published AbCDR training member's precomputed half-open indices without using either released validation or test members?",
        "source": {"uri": str(args.source), "sha256": expected["source"], "expected_rows": 776_634},
        "parent_protocol": {"uri": str(args.parent_protocol), "sha256": expected["parent_protocol"]},
        "supersedes_failures": {"v1": expected["v1_result"], "v2": expected["v2_result"], "v3": expected["v3_result"]},
        "span_rule": {
            "primary": "Use the exactly one jointly ordered non-overlapping CDR1/CDR2/CDR3 placement on each normalized chain.",
            "rescue": "Only when the primary rule is ambiguous, join exact VH, VL, and all six CDR strings to OAS_paired_set/train.csv and use its published zero-based half-open *_inds fields after exact substring and ordering checks.",
            "forbidden": ["official validation.csv", "official test.csv", "nearest-coordinate rounding", "lowering the frozen gate"],
            "maximum_model_input_residues_including_separator": 2046,
        },
        "official_train_authority": {
            "archive_uri": str(args.official_archive),
            "archive_sha256": expected["official_archive"],
            "member": "OAS_paired_set/train.csv",
            "train_audit_uri": str(args.official_train_audit),
            "train_audit_sha256": expected["official_train_audit"],
            "expected_rows": 1_577_499,
            "expected_unique_exact_pairs": 1_577_499,
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
            "require_pair_group_disjoint": True,
            "require_row_hash_unique": True,
            "require_finite_accounting": True,
            "require_official_rescue_exact_sequence_and_six_cdr_match": True,
        },
        "implementation": {
            "builder_uri": str(args.builder),
            "builder_sha256": sha256_file(args.builder),
            "launcher_uri": str(args.launcher),
            "launcher_sha256": sha256_file(args.launcher),
        },
        "access_disclosure": {
            "receipt_uri": str(args.source_access_receipt),
            "receipt_sha256": expected["source_access_receipt"],
            "official_validation_or_test_rows_permitted": False,
        },
        "claim_boundary": {
            "development_only": True,
            "local_adaptation_training_authorized_before_gate": False,
            "production_training_authorized": False,
            "task_labels_read": False,
            "sabdab2_old_test_accessed": False,
            "sabdab2_temporal_holdout_accessed": False,
        },
    }
    atomic_write(args.output, authority)
    print(json.dumps({"authority_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
