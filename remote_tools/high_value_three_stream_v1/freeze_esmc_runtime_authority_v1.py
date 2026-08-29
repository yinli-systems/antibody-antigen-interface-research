#!/usr/bin/env python3
"""Freeze exact source, checkpoint, corpus, and comparison semantics for ESM-C."""

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


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verified(path: Path, expected: str) -> dict[str, object]:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"SHA differs for {path}: {observed}")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--corpus-receipt", type=Path, required=True)
    parser.add_argument("--corpus-receipt-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--published-checkpoint", type=Path, required=True)
    parser.add_argument("--published-sha256", required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--verifier-source", type=Path, required=True)
    parser.add_argument("--esm-wheel", type=Path, required=True)
    parser.add_argument("--esm-wheel-sha256", required=True)
    parser.add_argument("--abcdr-source", type=Path, required=True)
    parser.add_argument("--abcdr-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    protocol = verified(args.protocol, args.protocol_sha256)
    corpus_receipt = verified(args.corpus_receipt, args.corpus_receipt_sha256)
    corpus = json.loads(args.corpus_receipt.read_text())
    if (
        corpus.get("status") != "PASS_PROTOCOL_V3_DECONTAMINATED_CORPUS_MATERIALIZED"
        or corpus.get("counts", {}).get("kept_pairs", 0) <= 0
        or corpus.get("authorization", {}).get("training_authorized") is not False
        or corpus.get("audit", {}).get("test_access") is not False
    ):
        raise RuntimeError("decontaminated corpus receipt contract differs")
    authority = {
        "format_version": "paired_esmc_runtime_authority_v1",
        "status": "FROZEN_BEFORE_RUNTIME_CANARY",
        "protocol": protocol,
        "corpus_receipt": corpus_receipt,
        "models": {
            "base": {
                **verified(args.base_checkpoint, args.base_sha256),
                "model_id": "biohub/ESMC-600M",
                "revision": "a7e82012c83126b9eedb055fea9fa84b6c02f094",
            },
            "published_cdr": {
                **verified(args.published_checkpoint, args.published_sha256),
                "model_id": "NOC-Lab/AbCDR-ESMC",
                "revision": "d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8",
                "role": "external published positive control",
            },
        },
        "sources": {
            "minimal_runtime": verified(args.runtime_source, sha256_file(args.runtime_source)),
            "runtime_verifier": verified(args.verifier_source, sha256_file(args.verifier_source)),
            "esm_3_1_4_wheel": verified(args.esm_wheel, args.esm_wheel_sha256),
            "abcdr_zenodo_archive": verified(args.abcdr_source, args.abcdr_source_sha256),
        },
        "comparison": {
            "antibody_base_arm": "base ESM-C 600M",
            "antibody_cdr_arm": "published AbCDR-ESMC",
            "antigen_both_arms": "base ESM-C 600M",
            "separator": "literal hyphen token id 30, retained during forward and removed from residue export",
            "embedding_dimension": 1152,
            "backbone_parameter_count": 573632640,
            "downstream_architecture_loss_split_seeds_fixed": True,
        },
        "authorization": {
            "runtime_canary_authorized": True,
            "embedding_materialization_authorized": False,
            "training_authorized": False,
            "test_access": False,
        },
        "audit": {
            "checkpoint_sha_verified": True,
            "corpus_materialized": True,
            "task_labels_read": False,
            "test_access": False,
            "training_started": False,
        },
    }
    atomic_write(args.output, canonical_bytes(authority))
    receipt = {
        "format_version": "paired_esmc_runtime_authority_receipt_v1",
        "status": authority["status"],
        "authority_sha256": sha256_file(args.output),
        "runtime_source_sha256": authority["sources"]["minimal_runtime"]["sha256"],
        "verifier_source_sha256": authority["sources"]["runtime_verifier"]["sha256"],
        "runtime_canary_authorized": True,
        "embedding_materialization_authorized": False,
        "training_authorized": False,
        "test_access": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
