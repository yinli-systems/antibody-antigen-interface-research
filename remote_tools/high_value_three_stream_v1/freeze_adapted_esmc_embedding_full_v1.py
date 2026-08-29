#!/usr/bin/env python3
"""Freeze full adapted ESM-C development embedding execution after canary PASS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


PARENT_PROTOCOL_SHA256 = "405752fab79a0316e6d09d9e19a1e60aeba87cbfd4005aa733c321ec856c5b6d"
CANARY_ANALYSIS_SHA256 = "1c82c0631325b0dfe8f9326e81bc94f5c11363ef9c3974efa7c9f2490ad685fb"
CANARY_RECEIPT_SHA256 = "9a96759e17412693ab02a4202df7238f87e53912da5d17d456d2edc84eb474af"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_sha(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} SHA differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--canary-analysis", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path, required=True)
    parser.add_argument("--full-launcher", type=Path, required=True)
    parser.add_argument("--full-finalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    require_sha(args.parent_protocol, PARENT_PROTOCOL_SHA256, "parent protocol")
    require_sha(args.canary_analysis, CANARY_ANALYSIS_SHA256, "canary analysis")
    require_sha(args.canary_receipt, CANARY_RECEIPT_SHA256, "canary receipt")
    parent = json.loads(args.parent_protocol.read_text())
    analysis = json.loads(args.canary_analysis.read_text())
    receipt = json.loads(args.canary_receipt.read_text())
    if (
        parent.get("status") != "FROZEN_AFTER_ADAPTATION_BEFORE_EMBEDDING"
        or parent.get("test_access") is not False
        or analysis.get("status") != "PASS_ADAPTED_ESMC_EMBEDDING_CANARY"
        or analysis.get("protocol_sha256") != PARENT_PROTOCOL_SHA256
        or analysis.get("authorization", {}).get(
            "full_development_embedding_materialization_authorized"
        )
        is not True
        or analysis.get("authorization", {}).get("downstream_training_authorized") is not False
        or analysis.get("authorization", {}).get("test_access") is not False
        or receipt.get("status") != analysis["status"]
        or receipt.get("analysis_sha256") != CANARY_ANALYSIS_SHA256
        or receipt.get("protocol_sha256") != PARENT_PROTOCOL_SHA256
        or receipt.get("full_development_embedding_materialization_authorized") is not True
        or receipt.get("downstream_training_authorized") is not False
        or receipt.get("test_access") is not False
    ):
        raise RuntimeError("adapted embedding canary authority differs")

    execution = {
        "format_version": "sabdab2_adapted_esmc_embedding_full_execution_v1",
        "status": "FROZEN_AFTER_CANARY_BEFORE_FULL_EMBEDDING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol": str(args.parent_protocol),
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "canary_analysis": str(args.canary_analysis),
        "canary_analysis_sha256": CANARY_ANALYSIS_SHA256,
        "canary_receipt": str(args.canary_receipt),
        "canary_receipt_sha256": CANARY_RECEIPT_SHA256,
        "development_sequence_plan": parent["development_sequence_plan"],
        "development_sequence_plan_sha256": parent["development_sequence_plan_sha256"],
        "shard_count_per_arm": 4,
        "adapted_antibody_models": parent["adapted_antibody_models"],
        "antigen_representation": parent["antigen_representation"],
        "outputs": {
            "result_root": (
                "/ssd/scxi253/model_deployment/results/prospective/"
                "framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/"
                "adapted-embedding-full-v1"
            ),
            "whole_chain_adapted_root": (
                "/ssd/scxi253/model_deployment/data/framecontact_20260730/"
                "sabdab2_esmc600m_whole_chain_adapted_dev_v1"
            ),
            "cdr_focused_adapted_root": (
                "/ssd/scxi253/model_deployment/data/framecontact_20260730/"
                "sabdab2_esmc600m_cdr_focused_adapted_dev_v1"
            ),
        },
        "source_binding": {
            "materializer_sha256": parent["source_binding"]["materializer_sha256"],
            "full_launcher_sha256": sha256_file(args.full_launcher),
            "full_finalizer_sha256": sha256_file(args.full_finalizer),
        },
        "authorization": {
            "full_development_embedding_materialization_authorized": True,
            "same_protocol_downstream_development_training_authorized_before_full_audit": False,
            "test_access": False,
        },
        "claim_boundary": {
            "development_only": True,
            "fixed_epoch4_adaptation_not_selected_by_downstream_metrics": True,
            "performance_gain_claim_authorized": False,
            "old_exposed_test_forbidden": True,
            "temporal_holdout_forbidden": True,
        },
        "test_access": False,
    }
    atomic_write(args.output, execution)
    output_receipt = {
        "format_version": "sabdab2_adapted_esmc_embedding_full_execution_receipt_v1",
        "status": execution["status"],
        "execution_sha256": sha256_file(args.output),
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "canary_analysis_sha256": CANARY_ANALYSIS_SHA256,
        "canary_receipt_sha256": CANARY_RECEIPT_SHA256,
        "full_development_embedding_materialization_authorized": True,
        "same_protocol_downstream_development_training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, output_receipt)
    print(
        json.dumps(
            {**output_receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
