#!/usr/bin/env python3
"""Materialize the immutable dense-pilot split for DDP parity gate v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from antibody_pretraining.dense_shard import file_sha256, load_dense_shard


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _load_protocol(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    try:
        protocol = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Parity protocol must be ASCII JSON") from error
    if (
        protocol.get("format_version")
        != "antibody_ddp_scientific_parity_gate_v2"
        or protocol.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise RuntimeError("Unexpected parity protocol")
    return protocol, hashlib.sha256(payload).hexdigest()


def _ranking_digest(namespace: str, sample_key: int) -> bytes:
    return hashlib.sha256(
        namespace.encode("ascii")
        + b"\0"
        + struct.pack("<Q", sample_key)
    ).digest()


def split_indices(
    sample_keys: np.ndarray,
    *,
    namespace: str,
    evaluation_count: int,
) -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    """Select the lowest namespace-bound sample-key digests."""
    keys = np.asarray(sample_keys)
    if keys.ndim != 1 or keys.dtype != np.uint64 or not len(keys):
        raise RuntimeError("Sample keys must be a nonempty uint64 vector")
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("Sample keys must be unique")
    if not 0 < evaluation_count < len(keys):
        raise RuntimeError("Evaluation count must define a nonempty split")
    try:
        namespace.encode("ascii")
    except UnicodeEncodeError as error:
        raise RuntimeError("Split namespace must be ASCII") from error
    ranked = [
        (_ranking_digest(namespace, int(key)), index)
        for index, key in enumerate(keys)
    ]
    digests = [item[0] for item in ranked]
    if len(set(digests)) != len(digests):
        raise RuntimeError("Ranking digest collision")
    ranked.sort()
    evaluation = np.asarray(
        sorted(index for _, index in ranked[:evaluation_count]),
        dtype=np.int64,
    )
    training = np.asarray(
        sorted(index for _, index in ranked[evaluation_count:]),
        dtype=np.int64,
    )
    return training, evaluation, [digest for digest, _ in ranked]


def _write_array(path: Path, array: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _array_receipt(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "dtype": str(array.dtype),
        "sha256": file_sha256(path),
        "shape": list(array.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Split output must be create-only")
    protocol, protocol_sha256 = _load_protocol(args.protocol)
    policy = protocol["pilot_split"]
    if args.manifest_sha256 != policy["dense_manifest_payload_sha256"]:
        raise RuntimeError("Dense manifest does not match the parity protocol")
    shard = load_dense_shard(
        args.shard,
        expected_manifest_sha256=args.manifest_sha256,
    )
    if len(shard) != policy["expected_record_count"]:
        raise RuntimeError("Dense pilot record count differs")
    training, evaluation, ranked_digests = split_indices(
        np.asarray(shard.sample_keys),
        namespace=policy["split_namespace"],
        evaluation_count=policy["evaluation_count"],
    )
    if (
        len(training) != policy["training_count"]
        or len(evaluation) != policy["evaluation_count"]
        or len(np.intersect1d(training, evaluation)) != 0
        or len(np.union1d(training, evaluation)) != len(shard)
    ):
        raise RuntimeError("Materialized split accounting differs")
    arrays = {
        "evaluation_indices.npy": evaluation,
        "evaluation_sample_keys.npy": np.asarray(
            shard.sample_keys[evaluation],
            dtype=np.uint64,
        ),
        "training_indices.npy": training,
        "training_sample_keys.npy": np.asarray(
            shard.sample_keys[training],
            dtype=np.uint64,
        ),
    }
    args.output.mkdir(parents=True)
    files = {}
    for name, array in arrays.items():
        path = args.output / name
        _write_array(path, array)
        files[name] = _array_receipt(path, array)
    receipt = {
        "accounting": {
            "complete": True,
            "disjoint": True,
            "evaluation_count": len(evaluation),
            "record_count": len(shard),
            "training_count": len(training),
        },
        "dense_manifest_file_sha256": file_sha256(
            args.shard / "manifest.json"
        ),
        "dense_manifest_payload_sha256": args.manifest_sha256,
        "files": files,
        "format_version": "antibody_ddp_parity_split_receipt_v2",
        "ranking": {
            "evaluation_maximum_digest": ranked_digests[
                policy["evaluation_count"] - 1
            ].hex(),
            "first_training_digest": ranked_digests[
                policy["evaluation_count"]
            ].hex(),
            "method": policy["ranking_digest"],
            "namespace": policy["split_namespace"],
        },
        "source": {
            "materializer_sha256": file_sha256(Path(__file__)),
            "protocol_sha256": protocol_sha256,
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path = args.output / "receipt.json"
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    args.output.chmod(0o555)


if __name__ == "__main__":
    main()
