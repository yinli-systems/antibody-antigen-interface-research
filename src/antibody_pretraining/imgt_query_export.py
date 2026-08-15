"""Authenticated AsEP-development query export for fixed-IMGT search.

The development cache is the only downstream authority accepted here.  Every
heavy/light sequence is digest-bound before ANARCII is invoked, and every
ANARCII result must reproduce the numbered domain exactly on the frozen
KA-Search 200-position vocabulary.  Unusual query positions fail closed; the
independent full-sequence MMseqs track remains responsible for gap-aware
coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Iterable, Mapping, Optional

from .downstream_overlap import DevelopmentChainReference, sequence_sha256
from .imgt_position import (
    CanonicalAlignment,
    IMGTPositionError,
    encode_numbered_residues,
)
from .near_identity_export import (
    publish_export_receipt,
    write_development_query_fasta,
)
from .near_identity_overlap import reference_key
from .oas_imgt_export import sequence_from_alignment
from .tokenizer import CHAIN_TYPE_TO_ID, ID_TO_CHAIN_TYPE


ANARCII_VERSION = "2.0.8"
ANARCII_SOURCE_COMMIT = "e0d8f192f5a861e03a50918f114d0f5735e42333"
ANARCII_WHEEL_SHA256 = (
    "6ac81caef04a9cad1af1ea320b35bdced620fe2bfdd0b8b17b98131495b584ca"
)
ANARCII_MODEL_SHA256 = (
    "03e9d07d13a3ec98a9d9fe57a0b2ff0853218aa2d7bdd50a25b6d83ff2bea2e0"
)
ANARCII_PARAMETERS_SHA256 = (
    "4d8e5fc9d8580f5e26a190a21853dd3f3770fbf83ce45a46a53d92b083063c39"
)
ASEP_QUERY_EXPORT_FORMAT = "oas_asep_development_imgt_query_export_v1"
IMGT_QUERY_MANIFEST_FORMAT = "oas_asep_development_imgt_query_manifest_v1"
ROLE_TO_ID = {"train": 0, "val": 1}
ID_TO_ROLE = {value: key for key, value in ROLE_TO_ID.items()}
IMGT_QUERY_RECORD = struct.Struct(">32sIBBB200s")
_RESIDUE = re.compile(r"[A-Z-]")


class IMGTQueryExportError(RuntimeError):
    """Report an incomplete or unauthenticated fixed-IMGT query export."""


@dataclass(frozen=True)
class IMGTQueryRecord:
    """One numbered development query in deterministic manifest order."""

    alignment: CanonicalAlignment
    chain_type: str
    domain_sequence_sha256: str
    query_ordinal: int
    role: str


@dataclass(frozen=True)
class NumberedDevelopmentQuery:
    """A fixed query record plus its provenance-bearing manifest fields."""

    binary: IMGTQueryRecord
    chain: str
    domain_sequence: str
    instance_id: str
    query_end: int
    query_reference_key: str
    query_start: int
    sequence: str
    sequence_sha256: str
    tool_score: float


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


def _regular_file(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata_value = candidate.lstat()
    except OSError as error:
        raise IMGTQueryExportError(f"Cannot stat {label}") from error
    if stat.S_ISLNK(metadata_value.st_mode) or not stat.S_ISREG(metadata_value.st_mode):
        raise IMGTQueryExportError(f"{label} is not a regular file")
    return candidate


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _digest_bytes(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except (TypeError, ValueError) as error:
        raise IMGTQueryExportError("Query sequence digest is invalid") from error
    if len(digest) != 32 or value != digest.hex():
        raise IMGTQueryExportError("Query sequence digest is not canonical SHA-256")
    return digest


def authenticate_anarcii_runtime(wheel: Path) -> dict[str, Any]:
    """Authenticate the isolated ANARCII 2.0.8 runtime and model assets."""
    wheel = _regular_file(wheel, "ANARCII wheel")
    if sys.version_info < (3, 11):
        raise IMGTQueryExportError("ANARCII 2.0.8 requires Python 3.11 or newer")
    if _sha256(wheel) != ANARCII_WHEEL_SHA256:
        raise IMGTQueryExportError("ANARCII wheel digest differs")
    try:
        distribution = metadata.distribution("anarcii")
    except metadata.PackageNotFoundError as error:
        raise IMGTQueryExportError("ANARCII runtime is unavailable") from error
    if distribution.version != ANARCII_VERSION:
        raise IMGTQueryExportError("ANARCII runtime version differs")
    root = Path(distribution.locate_file("")).absolute()
    assets = {
        "model": (
            root / "anarcii/models/antibody/antibody_4_2_128_512.pt",
            ANARCII_MODEL_SHA256,
        ),
        "parameters": (
            root / "anarcii/models/antibody/antibody_4_2_128_512.json",
            ANARCII_PARAMETERS_SHA256,
        ),
    }
    observed: dict[str, dict[str, str]] = {}
    for label, (path, expected) in assets.items():
        path = _regular_file(path, f"ANARCII {label}")
        actual = _sha256(path)
        if actual != expected:
            raise IMGTQueryExportError(f"ANARCII {label} digest differs")
        observed[label] = {"path": str(path), "sha256": actual}
    return {
        "assets": observed,
        "device": "cpu",
        "package": "anarcii",
        "package_version": ANARCII_VERSION,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "source_commit": ANARCII_SOURCE_COMMIT,
        "wheel_path": str(wheel),
        "wheel_sha256": ANARCII_WHEEL_SHA256,
    }


def _canonical_sequence(reference: DevelopmentChainReference, sequence: str) -> str:
    reference.validate()
    canonical = sequence.strip().upper()
    if (
        len(canonical) != reference.sequence_length
        or sequence_sha256(canonical) != reference.sequence_sha256
    ):
        raise IMGTQueryExportError("Development query sequence binding differs")
    return canonical


def _numbering(value: object) -> list[tuple[tuple[int, str], str]]:
    if type(value) is not list:
        raise IMGTQueryExportError("ANARCII numbering is not a list")
    output: list[tuple[tuple[int, str], str]] = []
    seen: set[tuple[int, str]] = set()
    core: set[int] = set()
    for item in value:
        if type(item) not in {tuple, list} or len(item) != 2:
            raise IMGTQueryExportError("ANARCII numbering entry is invalid")
        coordinate, residue = item
        if type(coordinate) not in {tuple, list} or len(coordinate) != 2:
            raise IMGTQueryExportError("ANARCII numbering coordinate is invalid")
        position, insertion = coordinate
        if (
            type(position) is not int
            or isinstance(position, bool)
            or not 1 <= position <= 128
            or type(insertion) is not str
            or (insertion.strip() and re.fullmatch(r"[A-Z]", insertion.strip()) is None)
            or type(residue) is not str
            or _RESIDUE.fullmatch(residue) is None
        ):
            raise IMGTQueryExportError("ANARCII numbering value is invalid")
        key = (position, insertion.strip())
        if key in seen:
            raise IMGTQueryExportError("ANARCII numbering coordinate is duplicated")
        seen.add(key)
        if not key[1]:
            core.add(position)
        output.append((key, residue))
    if core != set(range(1, 129)):
        raise IMGTQueryExportError("ANARCII core positions are incomplete")
    return output


def numbered_development_query(
    *,
    query_ordinal: int,
    reference: DevelopmentChainReference,
    sequence: str,
    result: object,
) -> NumberedDevelopmentQuery:
    """Validate one ANARCII result and construct a canonical query record."""
    canonical = _canonical_sequence(reference, sequence)
    if type(query_ordinal) is not int or isinstance(query_ordinal, bool) or query_ordinal < 0:
        raise IMGTQueryExportError("Query ordinal is invalid")
    if type(result) is not dict:
        raise IMGTQueryExportError("ANARCII result is not an object")
    if result.get("error") is not None:
        raise IMGTQueryExportError("ANARCII reported a query error")
    if result.get("scheme") != "imgt":
        raise IMGTQueryExportError("ANARCII query scheme differs")
    chain_type = result.get("chain_type")
    expected = {"heavy": {"H"}, "light": {"K", "L"}}[reference.chain]
    if chain_type not in expected:
        raise IMGTQueryExportError("ANARCII query chain type differs")
    score = result.get("score")
    if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
        raise IMGTQueryExportError("ANARCII query score is invalid")
    start = result.get("query_start")
    end = result.get("query_end")
    if not (
        type(start) is int
        and not isinstance(start, bool)
        and type(end) is int
        and not isinstance(end, bool)
        and 0 <= start <= end < len(canonical)
    ):
        raise IMGTQueryExportError("ANARCII query bounds are invalid")
    domain = canonical[start : end + 1]
    try:
        alignment = encode_numbered_residues(_numbering(result.get("numbering")))
    except IMGTPositionError as error:
        raise IMGTQueryExportError(f"ANARCII query position differs: {error}") from error
    if sequence_from_alignment(alignment) != domain:
        raise IMGTQueryExportError("ANARCII numbering does not reproduce query domain")
    normalized_chain = {"H": "heavy", "K": "kappa", "L": "lambda"}[chain_type]
    domain_digest = hashlib.sha256(domain.encode("ascii")).hexdigest()
    binary = IMGTQueryRecord(
        alignment=alignment,
        chain_type=normalized_chain,
        domain_sequence_sha256=domain_digest,
        query_ordinal=query_ordinal,
        role=reference.role,
    )
    pack_imgt_query(binary)
    return NumberedDevelopmentQuery(
        binary=binary,
        chain=reference.chain,
        domain_sequence=domain,
        instance_id=reference.instance_id,
        query_end=end,
        query_reference_key=reference_key(reference),
        query_start=start,
        sequence=canonical,
        sequence_sha256=reference.sequence_sha256,
        tool_score=float(score),
    )


def pack_imgt_query(record: IMGTQueryRecord) -> bytes:
    """Pack one query and prove its domain digest from fixed-position bytes."""
    if not 0 <= record.query_ordinal < 2**32:
        raise IMGTQueryExportError("Query ordinal is outside uint32")
    try:
        role_id = ROLE_TO_ID[record.role]
        chain_id = CHAIN_TYPE_TO_ID[record.chain_type]
    except KeyError as error:
        raise IMGTQueryExportError("Query enum is invalid") from error
    if chain_id not in {1, 2, 3}:
        raise IMGTQueryExportError("Query chain type is invalid")
    record.alignment.validate()
    domain = sequence_from_alignment(record.alignment).encode("ascii")
    digest = _digest_bytes(record.domain_sequence_sha256)
    if hashlib.sha256(domain).digest() != digest:
        raise IMGTQueryExportError("Query alignment domain digest differs")
    return IMGT_QUERY_RECORD.pack(
        digest,
        record.query_ordinal,
        role_id,
        chain_id,
        record.alignment.residue_count,
        record.alignment.encoded,
    )


def unpack_imgt_query(payload: bytes) -> IMGTQueryRecord:
    """Unpack one query record and reject any non-canonical byte sequence."""
    if len(payload) != IMGT_QUERY_RECORD.size:
        raise IMGTQueryExportError("Query record byte count differs")
    digest, ordinal, role_id, chain_id, residue_count, encoded = IMGT_QUERY_RECORD.unpack(
        payload
    )
    try:
        role = ID_TO_ROLE[role_id]
        chain_type = ID_TO_CHAIN_TYPE[chain_id]
    except KeyError as error:
        raise IMGTQueryExportError("Query record enum is invalid") from error
    record = IMGTQueryRecord(
        alignment=CanonicalAlignment(encoded=encoded, residue_count=residue_count),
        chain_type=chain_type,
        domain_sequence_sha256=digest.hex(),
        query_ordinal=ordinal,
        role=role,
    )
    if pack_imgt_query(record) != payload:
        raise IMGTQueryExportError("Query record is not canonical")
    return record


def export_numbered_development_queries(
    *,
    output_root: Path,
    sequences: Iterable[tuple[DevelopmentChainReference, str]],
    numbered: Mapping[str, object],
    runtime_provenance: Mapping[str, Any],
    expected_query_count: int = 3_106,
    source_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create a complete fixed-IMGT query bundle without overwriting evidence."""
    rows = tuple(sequences)
    if type(expected_query_count) is not int or expected_query_count <= 0:
        raise IMGTQueryExportError("Expected query count is invalid")
    keyed: list[tuple[str, DevelopmentChainReference, str]] = []
    for reference, sequence in rows:
        canonical = _canonical_sequence(reference, sequence)
        keyed.append((reference_key(reference), reference, canonical))
    keyed.sort(key=lambda value: value[0])
    keys = [value[0] for value in keyed]
    if len(keys) != expected_query_count or len(keys) != len(set(keys)):
        raise IMGTQueryExportError("Development query count or uniqueness differs")
    if type(numbered) is not dict or set(numbered) != set(keys):
        raise IMGTQueryExportError("ANARCII result identities differ")
    records = [
        numbered_development_query(
            query_ordinal=ordinal,
            reference=reference,
            sequence=sequence,
            result=numbered[key],
        )
        for ordinal, (key, reference, sequence) in enumerate(keyed)
    ]
    required_runtime = {
        "package": "anarcii",
        "package_version": ANARCII_VERSION,
        "source_commit": ANARCII_SOURCE_COMMIT,
        "wheel_sha256": ANARCII_WHEEL_SHA256,
    }
    if any(runtime_provenance.get(key) != value for key, value in required_runtime.items()):
        raise IMGTQueryExportError("ANARCII runtime provenance differs")

    output_root = Path(output_root).expanduser().absolute()
    output_root.mkdir(parents=True, exist_ok=False)
    fasta_path = output_root / "queries.fasta"
    sequence_manifest_path = output_root / "sequence_manifest.json"
    binary_path = output_root / "queries.imgt.bin"
    manifest_path = output_root / "imgt_query_manifest.json"
    receipt_path = output_root / "EXPORT_RECEIPT.json"
    sequence_manifest = write_development_query_fasta(
        fasta_path=fasta_path,
        manifest_path=sequence_manifest_path,
        sequences=((reference, sequence) for _, reference, sequence in keyed),
    )
    _exclusive_bytes(binary_path, b"".join(pack_imgt_query(row.binary) for row in records))
    manifest_records = []
    for row in records:
        manifest_records.append(
            {
                "chain": row.chain,
                "domain_sequence_length": len(row.domain_sequence),
                "domain_sequence_sha256": row.binary.domain_sequence_sha256,
                "imgt_residue_count": row.binary.alignment.residue_count,
                "instance_id": row.instance_id,
                "numbered_chain_type": row.binary.chain_type,
                "query_end_zero_based_inclusive": row.query_end,
                "query_ordinal": row.binary.query_ordinal,
                "query_reference_key": row.query_reference_key,
                "query_start_zero_based_inclusive": row.query_start,
                "role": row.binary.role,
                "sequence_length": len(row.sequence),
                "sequence_sha256": row.sequence_sha256,
                "tool_score": row.tool_score,
            }
        )
    manifest = {
        "binary_record_bytes": IMGT_QUERY_RECORD.size,
        "format_version": IMGT_QUERY_MANIFEST_FORMAT,
        "query_count": len(records),
        "records": manifest_records,
        "runtime": dict(runtime_provenance),
        "source": dict(source_provenance or {}),
        "test_access": {
            "development_sequence_fields_accessed": True,
            "frozen_official_external_test_accessed": False,
            "model_predictions_or_scores_accessed": False,
            "task_label_values_used": False,
        },
    }
    manifest["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    _exclusive_bytes(manifest_path, _canonical_json_bytes(manifest))
    outputs = {}
    for path in (fasta_path, sequence_manifest_path, binary_path, manifest_path):
        outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    receipt = {
        "authorization": {
            "fixed_imgt_search_authorized": False,
            "large_scale_pretraining_authorized": False,
            "reason": "exact fixed-IMGT and gap-aware near-identity tracks incomplete",
        },
        "format_version": ASEP_QUERY_EXPORT_FORMAT,
        "outputs": outputs,
        "query_count": len(records),
        "runtime": dict(runtime_provenance),
        "sequence_manifest_receipt_sha256": sequence_manifest["receipt_sha256"],
        "test_access": manifest["test_access"],
    }
    publish_export_receipt(receipt_path, receipt)
    return json.loads(receipt_path.read_text(encoding="ascii"))


if IMGT_QUERY_RECORD.size != 239:
    raise RuntimeError("Fixed IMGT query schema differs from the frozen width")


__all__ = [
    "ANARCII_MODEL_SHA256",
    "ANARCII_PARAMETERS_SHA256",
    "ANARCII_SOURCE_COMMIT",
    "ANARCII_VERSION",
    "ANARCII_WHEEL_SHA256",
    "ASEP_QUERY_EXPORT_FORMAT",
    "IMGT_QUERY_MANIFEST_FORMAT",
    "IMGT_QUERY_RECORD",
    "IMGTQueryExportError",
    "IMGTQueryRecord",
    "NumberedDevelopmentQuery",
    "authenticate_anarcii_runtime",
    "export_numbered_development_queries",
    "numbered_development_query",
    "pack_imgt_query",
    "unpack_imgt_query",
]
