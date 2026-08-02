#!/usr/bin/env python3
"""Measure exact unique-residue-token and composition counts in audited OAS shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = "oas_production_corpus_scale_audit_result_v1"
MANIFEST_FORMAT = "oas_production_corpus_manifest_v1"
SHARD_FORMAT = "oas_production_dense_bucket_shard_v1"
INDEPENDENT_AUDIT_FORMAT = "oas_production_corpus_independent_audit_v1"
PROTOCOL_FORMAT = "oas_production_corpus_scale_audit_protocol_v1"
USED_ARRAYS = {
    "chain_types.npy": (np.dtype("uint8"), 4),
    "sequence_lengths.npy": (np.dtype("uint16"), 193),
    "status_flags.npy": (np.dtype("uint8"), 16),
    "strata.npy": (np.dtype("uint8"), 4),
    "study_ids.npy": (np.dtype("uint16"), 90),
}
TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


class CorpusScaleAuditError(RuntimeError):
    """Report invalid authority, shard, array, or scale-gate evidence."""


def _reject_constant(value: str) -> None:
    raise CorpusScaleAuditError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CorpusScaleAuditError(f"Duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CorpusScaleAuditError(f"Input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, *, canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CorpusScaleAuditError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusScaleAuditError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or (canonical and raw != _canonical_json_bytes(value)):
        raise CorpusScaleAuditError(f"Input is not canonical JSON: {path}")
    return value, raw


def _load_self_hashed(
    path: Path,
    *,
    digest_field: str,
) -> tuple[dict[str, Any], str, str]:
    value, raw = _load_json(path)
    unsigned = dict(value)
    claimed = unsigned.pop(digest_field, None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise CorpusScaleAuditError(f"Self hash differs: {path}")
    return value, hashlib.sha256(raw).hexdigest(), observed


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CorpusScaleAuditError(f"{label} is not a lowercase SHA-256")
    return value


def _verify_protocol_inputs(
    *,
    source_root: Path,
    protocol: Mapping[str, Any],
) -> None:
    bindings = {
        "efficiency_protocol_file_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "large_model_protocol_file_sha256": "protocols/antibody_large_model_pilot_v1.json",
        "materialization_protocol_file_sha256": "protocols/oas_production_materialization_v3.json",
    }
    for key, relative in bindings.items():
        if _file_sha256(source_root / relative) != protocol["inputs"][key]:
            raise CorpusScaleAuditError(f"Protocol-bound input differs: {relative}")
    large_model, _ = _load_json(
        source_root / "protocols/antibody_large_model_pilot_v1.json",
        canonical=False,
    )
    efficiency, _ = _load_json(
        source_root / "protocols/antibody_efficiency_gate_v1.json",
        canonical=False,
    )
    threshold = protocol["acceptance"][
        "minimum_large_model_train_unique_residue_tokens"
    ]
    if (
        large_model.get("production_gate", {}).get(
            "minimum_admitted_unique_residue_tokens"
        )
        != threshold
        or efficiency.get("large_model_gate", {}).get(
            "minimum_admitted_unique_residue_tokens"
        )
        != threshold
    ):
        raise CorpusScaleAuditError("Large-model unique-token thresholds disagree")


def _array_histogram(
    array: np.ndarray,
    *,
    bin_count: int,
    minimum: int,
    maximum: int,
) -> tuple[list[int], int]:
    histogram = np.zeros(bin_count, dtype=np.uint64)
    total = 0
    for start in range(0, int(array.shape[0]), 1_048_576):
        chunk = np.asarray(array[start : start + 1_048_576])
        if chunk.size and (int(chunk.min()) < minimum or int(chunk.max()) > maximum):
            raise CorpusScaleAuditError("Composition array value is outside its domain")
        histogram += np.bincount(
            chunk.astype(np.int64, copy=False),
            minlength=bin_count,
        ).astype(np.uint64, copy=False)
        total += int(chunk.sum(dtype=np.uint64))
    return [int(value) for value in histogram], total


def _audit_shard(
    *,
    corpus_root: Path,
    bucket: int,
    split: str,
    reference: Mapping[str, Any],
    study_count: int,
    minimum_length: int,
    maximum_length: int,
) -> dict[str, Any]:
    expected_relative_root = f"reduce/bucket-{bucket:03d}/{split}"
    if reference.get("relative_root") != expected_relative_root:
        raise CorpusScaleAuditError("Shard relative root differs")
    root = corpus_root / expected_relative_root
    manifest, file_sha256, payload_sha256 = _load_self_hashed(
        root / "manifest.json",
        digest_field="manifest_sha256",
    )
    if (
        file_sha256 != reference.get("manifest_file_sha256")
        or payload_sha256 != reference.get("manifest_payload_sha256")
        or manifest.get("format_version") != SHARD_FORMAT
        or manifest.get("bucket") != bucket
        or manifest.get("split") != split
        or manifest.get("record_count") != reference.get("record_count")
        or manifest.get("test_access") != TEST_ACCESS
    ):
        raise CorpusScaleAuditError("Shard manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not set(USED_ARRAYS).issubset(files):
        raise CorpusScaleAuditError("Shard composition array inventory differs")
    arrays: dict[str, np.ndarray] = {}
    file_receipts = {}
    record_count = int(reference["record_count"])
    for name, (dtype, _) in USED_ARRAYS.items():
        path = root / name
        item = files[name]
        if (
            not isinstance(item, Mapping)
            or path.stat().st_size != item.get("bytes")
            or _file_sha256(path) != item.get("sha256")
        ):
            raise CorpusScaleAuditError("Shard composition file differs")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != dtype
            or tuple(array.shape) != (record_count,)
            or item.get("dtype") != str(dtype)
            or item.get("shape") != [record_count]
        ):
            raise CorpusScaleAuditError("Shard composition array schema differs")
        arrays[name] = array
        file_receipts[name] = {
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
        }

    lengths, residue_tokens = _array_histogram(
        arrays["sequence_lengths.npy"],
        bin_count=maximum_length + 1,
        minimum=minimum_length,
        maximum=maximum_length,
    )
    chain_types, _ = _array_histogram(
        arrays["chain_types.npy"], bin_count=4, minimum=1, maximum=3
    )
    status_flags, _ = _array_histogram(
        arrays["status_flags.npy"], bin_count=16, minimum=0, maximum=15
    )
    strata, _ = _array_histogram(
        arrays["strata.npy"], bin_count=4, minimum=0, maximum=3
    )
    studies, _ = _array_histogram(
        arrays["study_ids.npy"],
        bin_count=study_count,
        minimum=0,
        maximum=study_count - 1,
    )
    if (
        sum(lengths) != record_count
        or sum(chain_types) != record_count
        or sum(status_flags) != record_count
        or sum(strata) != record_count
        or sum(studies) != record_count
    ):
        raise CorpusScaleAuditError("Shard composition accounting differs")
    return {
        "bucket": bucket,
        "chain_type_histogram": chain_types,
        "file_receipts": file_receipts,
        "record_count": record_count,
        "residue_tokens": residue_tokens,
        "sequence_length_histogram": lengths,
        "split": split,
        "status_flag_histogram": status_flags,
        "stratum_histogram": strata,
        "study_histogram": studies,
    }


def _sum_histograms(items: list[Mapping[str, Any]], field: str) -> list[int]:
    width = len(items[0][field]) if items else 0
    if any(len(item[field]) != width for item in items):
        raise CorpusScaleAuditError(f"Histogram width differs: {field}")
    return [sum(int(item[field][index]) for item in items) for index in range(width)]


def audit_corpus_scale(
    *,
    source_root: Path,
    corpus_root: Path,
    protocol_path: Path,
    independent_audit_path: Path,
    independent_audit_file_sha256: str,
    independent_audit_payload_sha256: str,
    worker_count: int,
) -> dict[str, Any]:
    _require_digest(independent_audit_file_sha256, "Independent audit file digest")
    _require_digest(
        independent_audit_payload_sha256,
        "Independent audit payload digest",
    )
    if not 1 <= worker_count <= 8:
        raise CorpusScaleAuditError("Worker count is outside the frozen range")
    protocol, protocol_raw = _load_json(protocol_path, canonical=False)
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_pretraining": False,
            "scale_16_gpu": False,
            "scale_32_gpu": False,
            "scientific_claim": False,
            "systems_1_3b_pilot": False,
        }
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("measurement", {}).get("arrays") != sorted(USED_ARRAYS)
    ):
        raise CorpusScaleAuditError("Scale-audit protocol identity differs")
    _verify_protocol_inputs(source_root=source_root, protocol=protocol)

    manifest, manifest_file_sha256, manifest_payload_sha256 = _load_self_hashed(
        corpus_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
    )
    finalize, finalize_file_sha256, finalize_payload_sha256 = _load_self_hashed(
        corpus_root / "corpus_finalize_receipt.json",
        digest_field="receipt_sha256",
    )
    expected_inputs = protocol["inputs"]
    if (
        manifest_file_sha256 != expected_inputs["corpus_manifest_file_sha256"]
        or manifest_payload_sha256 != expected_inputs["corpus_manifest_payload_sha256"]
        or finalize_file_sha256 != expected_inputs["corpus_finalize_receipt_file_sha256"]
        or finalize_payload_sha256
        != expected_inputs["corpus_finalize_receipt_payload_sha256"]
        or manifest.get("format_version") != MANIFEST_FORMAT
        or finalize.get("manifest_file_sha256") != manifest_file_sha256
        or finalize.get("manifest_payload_sha256") != manifest_payload_sha256
        or manifest.get("test_access") != TEST_ACCESS
        or manifest.get("authorization")
        != {
            "large_scale_pretraining": False,
            "production_corpus_use": False,
            "scientific_claim": False,
        }
    ):
        raise CorpusScaleAuditError("Corpus manifest or finalization identity differs")

    independent, independent_raw = _load_json(independent_audit_path)
    unsigned = dict(independent)
    claimed_audit_payload = unsigned.pop("receipt_sha256", None)
    if (
        hashlib.sha256(independent_raw).hexdigest() != independent_audit_file_sha256
        or claimed_audit_payload != independent_audit_payload_sha256
        or hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        != claimed_audit_payload
        or independent.get("format_version") != INDEPENDENT_AUDIT_FORMAT
        or independent.get("authorization", {}).get("independent_replay") is not True
        or independent.get("inputs", {}).get("manifest_file_sha256")
        != manifest_file_sha256
        or independent.get("inputs", {}).get("manifest_payload_sha256")
        != manifest_payload_sha256
        or independent.get("counts") != manifest.get("counts")
        or independent.get("test_access") != TEST_ACCESS
    ):
        raise CorpusScaleAuditError("Independent corpus replay evidence differs")

    acceptance = protocol["acceptance"]
    buckets = manifest.get("buckets")
    if (
        not isinstance(buckets, list)
        or len(buckets) != acceptance["bucket_count"]
        or manifest.get("bucket_count") != acceptance["bucket_count"]
        or manifest.get("map_identity", {}).get("study_count")
        != acceptance["study_count"]
    ):
        raise CorpusScaleAuditError("Corpus bucket or study topology differs")
    tasks = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for expected_bucket, item in enumerate(buckets):
            if not isinstance(item, Mapping) or item.get("bucket") != expected_bucket:
                raise CorpusScaleAuditError("Corpus bucket order differs")
            shards = item.get("shards")
            if not isinstance(shards, Mapping) or set(shards) != {"train", "validation"}:
                raise CorpusScaleAuditError("Corpus shard inventory differs")
            for split in ("train", "validation"):
                future = pool.submit(
                    _audit_shard,
                    corpus_root=corpus_root,
                    bucket=expected_bucket,
                    split=split,
                    reference=shards[split],
                    study_count=int(acceptance["study_count"]),
                    minimum_length=int(acceptance["minimum_sequence_length"]),
                    maximum_length=int(acceptance["maximum_sequence_length"]),
                )
                futures[future] = (expected_bucket, split)
        for future in as_completed(futures):
            tasks.append(future.result())
    tasks.sort(key=lambda item: (int(item["bucket"]), str(item["split"])))

    summaries = {}
    for split in ("train", "validation"):
        selected = [item for item in tasks if item["split"] == split]
        record_count = sum(int(item["record_count"]) for item in selected)
        residue_tokens = sum(int(item["residue_tokens"]) for item in selected)
        summaries[split] = {
            "chain_type_histogram": _sum_histograms(selected, "chain_type_histogram"),
            "encoded_tokens_including_cls_sep": residue_tokens + 2 * record_count,
            "mean_sequence_length": residue_tokens / record_count,
            "record_count": record_count,
            "residue_tokens": residue_tokens,
            "sequence_length_histogram": _sum_histograms(
                selected, "sequence_length_histogram"
            ),
            "status_flag_histogram": _sum_histograms(
                selected, "status_flag_histogram"
            ),
            "stratum_histogram": _sum_histograms(selected, "stratum_histogram"),
            "study_histogram": _sum_histograms(selected, "study_histogram"),
        }
    if (
        summaries["train"]["record_count"] != acceptance["train_record_count"]
        or summaries["validation"]["record_count"]
        != acceptance["validation_record_count"]
    ):
        raise CorpusScaleAuditError("Corpus split record counts differ")
    threshold = int(acceptance["minimum_large_model_train_unique_residue_tokens"])
    observed = int(summaries["train"]["residue_tokens"])
    semantic_sha256 = hashlib.sha256(_canonical_json_bytes(tasks)).hexdigest()
    return {
        "audit_passed": True,
        "authorization": protocol["authorization"],
        "claim_boundary": protocol["claim_boundary"],
        "format_version": FORMAT_VERSION,
        "inputs": {
            "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "corpus_manifest_file_sha256": manifest_file_sha256,
            "corpus_manifest_payload_sha256": manifest_payload_sha256,
            "independent_audit_file_sha256": independent_audit_file_sha256,
            "independent_audit_payload_sha256": independent_audit_payload_sha256,
            "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        },
        "large_model_data_gate": {
            "observed_train_unique_residue_tokens": observed,
            "passed": observed >= threshold,
            "required_train_unique_residue_tokens": threshold,
        },
        "semantic_composition_sha256": semantic_sha256,
        "splits": summaries,
        "test_access": TEST_ACCESS,
        "verified_shard_count": len(tasks),
        "worker_count": worker_count,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = audit_corpus_scale(
        source_root=arguments.source_root,
        corpus_root=arguments.corpus_root,
        protocol_path=arguments.protocol,
        independent_audit_path=arguments.independent_audit,
        independent_audit_file_sha256=arguments.independent_audit_file_sha256,
        independent_audit_payload_sha256=arguments.independent_audit_payload_sha256,
        worker_count=arguments.worker_count,
    )
    _publish(arguments.output, result)


if __name__ == "__main__":
    main()
