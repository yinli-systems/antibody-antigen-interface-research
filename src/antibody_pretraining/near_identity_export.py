"""Authenticated input construction for OAS-to-AsEP near-identity searches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable, Mapping

from .downstream_overlap import DevelopmentChainReference, sequence_sha256
from .near_identity_overlap import NearIdentityOverlapError, reference_key
from .production_corpus import ProductionCorpus


EXPORT_FORMAT = "oas_near_identity_fasta_export_inventory_v1"


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class TargetShardExport:
    """Describe one already-authenticated dense OAS shard for native export."""

    split: str
    bucket: int
    global_offset: int
    record_count: int
    tokens_path: Path
    lengths_path: Path

    def validate(self) -> None:
        if self.split not in {"train", "validation"}:
            raise NearIdentityOverlapError("Target export split is invalid")
        if not 0 <= self.bucket < 256:
            raise NearIdentityOverlapError("Target export bucket is invalid")
        if self.global_offset < 0 or self.record_count < 0:
            raise NearIdentityOverlapError("Target export count is invalid")
        for array_path in (self.tokens_path, self.lengths_path):
            metadata = array_path.lstat()
            if (
                not array_path.is_absolute()
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or any(character in str(array_path) for character in "\t\r\n")
            ):
                raise NearIdentityOverlapError("Target export array path is unsafe")


def build_target_shard_inventory(
    *, train: ProductionCorpus, validation: ProductionCorpus
) -> tuple[TargetShardExport, ...]:
    """Bind both immutable corpus splits to a deterministic native inventory."""
    contracts = (train.contract, validation.contract)
    if (
        contracts[0].split != "train"
        or contracts[1].split != "validation"
        or not all(contract.array_file_hashes_verified is True for contract in contracts)
    ):
        raise NearIdentityOverlapError("Target corpora are not fully authenticated")
    anchor_fields = (
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "protocol_file_sha256",
        "independent_audit_file_sha256",
        "independent_audit_payload_sha256",
    )
    if any(
        getattr(contracts[0], field) != getattr(contracts[1], field)
        for field in anchor_fields
    ):
        raise NearIdentityOverlapError("Target corpus evidence anchors differ")
    output: list[TargetShardExport] = []
    for corpus in (train, validation):
        if len(corpus.offsets) != len(corpus.shards) + 1:
            raise NearIdentityOverlapError("Target corpus offsets differ")
        for index, shard in enumerate(corpus.shards):
            tokens = shard.arrays.get("tokens.npy")
            lengths = shard.arrays.get("lengths.npy")
            if tokens is None or lengths is None:
                raise NearIdentityOverlapError("Target corpus lacks export arrays")
            item = TargetShardExport(
                split=corpus.contract.split,
                bucket=int(shard.bucket),
                global_offset=int(corpus.offsets[index]),
                record_count=int(shard.record_count),
                tokens_path=tokens.path,
                lengths_path=lengths.path,
            )
            item.validate()
            output.append(item)
    expected = {
        split: list(range(256)) for split in ("train", "validation")
    }
    observed = {
        split: [item.bucket for item in output if item.split == split]
        for split in expected
    }
    if observed != expected:
        raise NearIdentityOverlapError("Target export bucket inventory differs")
    return tuple(output)


def write_target_inventory(path: Path, items: Iterable[TargetShardExport]) -> dict[str, Any]:
    """Write the exact tabular contract consumed by the C++ streaming exporter."""
    records = tuple(items)
    if not records:
        raise NearIdentityOverlapError("Target export inventory is empty")
    lines = ["split\tbucket\tglobal_offset\trecord_count\ttokens_path\tlengths_path\n"]
    counts = {"train": 0, "validation": 0}
    for item in records:
        item.validate()
        counts[item.split] += item.record_count
        lines.append(
            f"{item.split}\t{item.bucket}\t{item.global_offset}\t{item.record_count}"
            f"\t{item.tokens_path}\t{item.lengths_path}\n"
        )
    _exclusive_bytes(path, "".join(lines).encode("utf-8"))
    return {
        "format_version": EXPORT_FORMAT,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "shard_count": len(records),
        "record_counts": counts,
    }


def write_development_query_fasta(
    *,
    fasta_path: Path,
    manifest_path: Path,
    sequences: Iterable[tuple[DevelopmentChainReference, str]],
) -> dict[str, Any]:
    """Export label-free development chains and a digest-bound query manifest."""
    rows = tuple(sequences)
    if not rows:
        raise NearIdentityOverlapError("Development query export is empty")
    keyed: list[tuple[str, DevelopmentChainReference, str]] = []
    for reference, sequence in rows:
        reference.validate()
        canonical = sequence.strip().upper()
        if (
            len(canonical) != reference.sequence_length
            or sequence_sha256(canonical) != reference.sequence_sha256
        ):
            raise NearIdentityOverlapError("Development query sequence binding differs")
        keyed.append((reference_key(reference), reference, canonical))
    keyed.sort(key=lambda value: value[0])
    keys = [value[0] for value in keyed]
    if len(keys) != len(set(keys)):
        raise NearIdentityOverlapError("Development query key is duplicated")
    fasta = bytearray()
    records: list[dict[str, Any]] = []
    for key, reference, sequence in keyed:
        fasta.extend(f">{key}\n{sequence}\n".encode("ascii"))
        records.append(
            {
                "chain": reference.chain,
                "instance_id": reference.instance_id,
                "query_reference_key": key,
                "role": reference.role,
                "sequence_length": reference.sequence_length,
                "sequence_sha256": reference.sequence_sha256,
            }
        )
    _exclusive_bytes(fasta_path, bytes(fasta))
    manifest = {
        "format_version": "oas_asep_development_query_manifest_v1",
        "fasta_path": str(fasta_path.resolve()),
        "fasta_sha256": _sha256(fasta_path),
        "query_count": len(records),
        "records": records,
        "test_access": {
            "development_sequence_fields_accessed": True,
            "official_frozen_external_test_accessed": False,
            "task_labels_used": False,
        },
    }
    manifest["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    _exclusive_bytes(manifest_path, _canonical_json_bytes(manifest))
    return manifest


def publish_export_receipt(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one finite, self-hashed export receipt atomically and exclusively."""
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    result = dict(unsigned)
    result["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite export receipt: {path}")
        os.link(temporary, path)
        path.chmod(0o444)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "EXPORT_FORMAT",
    "TargetShardExport",
    "build_target_shard_inventory",
    "publish_export_receipt",
    "write_development_query_fasta",
    "write_target_inventory",
]
