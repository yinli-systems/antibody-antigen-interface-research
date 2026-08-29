#!/usr/bin/env python3
"""Validate and seal the CHIMERA protocol before external label access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "chimera_zero_shot_residue_interface_protocol_v1"
EXPECTED_MODES = {"sequence_dense", "spatial_topology"}
EXPECTED_SEEDS = {"17", "42", "73"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(path: Path, *, verify_checkpoints: bool) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("protocol must be a regular non-symlink file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("unexpected protocol schema")
    if payload["data_firewall"]["external_labels_or_model_scores_read_before_freeze"] is not False:
        raise ValueError("data firewall does not certify a pre-access freeze")
    dataset = payload["dataset"]
    if dataset["pinned_revision"] != "54d5a2a1e59a329db7a6c09184652ed225188940":
        raise ValueError("CHIMERA revision is not pinned to the registered authority")
    if dataset["state_at_freeze"] != {
        "duplicate_groups_status": "not_computed",
        "normalization_status": "not_run",
        "raw_status": "not_materialized",
    }:
        raise ValueError("pre-access dataset state differs")
    evaluation = payload["evaluation"]
    if evaluation["primary_comparison"] != "spatial_topology minus sequence_dense":
        raise ValueError("primary comparison differs")
    if evaluation["primary_splits"] != ["temporal.test", "antigen_fold.test"]:
        raise ValueError("primary split order differs")
    if evaluation["bootstrap"] != {
        "confidence": 0.95,
        "group_key_priority": [
            "official antigen cluster or fold group",
            "sha256 of normalized antigen sequence",
        ],
        "replicates": 10000,
        "resampling": "paired group bootstrap; both candidates and every seed stay paired",
        "seed": 20260730,
    }:
        raise ValueError("bootstrap contract differs")
    candidates = payload["model"]["candidates"]
    if set(candidates) != EXPECTED_MODES:
        raise ValueError("candidate set differs")
    seen_paths: set[str] = set()
    for mode, checkpoints in candidates.items():
        if set(checkpoints) != EXPECTED_SEEDS:
            raise ValueError(f"seed set differs for {mode}")
        for seed, authority in checkpoints.items():
            digest = authority.get("checkpoint_sha256")
            checkpoint = Path(authority.get("path", ""))
            if not isinstance(digest, str) or not SHA256.fullmatch(digest):
                raise ValueError(f"invalid checkpoint digest for {mode}/{seed}")
            if str(checkpoint) in seen_paths:
                raise ValueError("checkpoint path is duplicated")
            seen_paths.add(str(checkpoint))
            if verify_checkpoints:
                if checkpoint.is_symlink() or not checkpoint.is_file():
                    raise ValueError(f"missing checkpoint: {checkpoint}")
                observed = sha256_file(checkpoint)
                if observed != digest:
                    raise ValueError(f"checkpoint digest mismatch: {checkpoint}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise ValueError("receipt already exists; freeze is create-only")
    payload = load_and_validate(args.protocol, verify_checkpoints=args.verify_checkpoints)
    receipt = {
        "schema_version": "chimera_zero_shot_frozen_protocol_receipt_v1",
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "frozen_at_utc": payload["data_firewall"]["frozen_at_utc"],
        "checkpoint_count": 6,
        "checkpoint_digests_verified": bool(args.verify_checkpoints),
        "external_labels_or_model_scores_read_before_freeze": False,
        "dataset_pinned_revision": payload["dataset"]["pinned_revision"],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, args.receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
