"""Annotate AVIDa VHH construct boundaries under the frozen protocol."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from numbers import Real
from pathlib import Path
import platform
import re
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from interaction_first.avida_annotation_protocol import (
    load_avida_annotation_protocol,
)


SOURCE_ID = "avida_hil6_2024_11"
INPUT_STATUS = "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
ACCEPTED_STATUS = "imgt_heavy_domain_verified"
QUARANTINED_STATUS = "quarantined_vhh_boundary"
_ID = re.compile(r"^ab_sha256:[0-9a-f]{64}$")


class AvidaVhhAnnotationError(ValueError):
    """Raised when VHH annotation cannot produce authenticated evidence."""


def annotate_avida_vhh_boundaries(
    *,
    candidate_root: Path,
    protocol_path: Path,
    anarcii_wheel: Path,
    output_root: Path,
    batch_size: int = 512,
    chunk_size: int = 4_096,
    ncpu: int = 1,
    created_at_utc: str | None = None,
    numberer_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    """Run pinned ANARCII inference and preserve every accepted or failed row."""

    if type(batch_size) is not int or batch_size <= 0:
        raise AvidaVhhAnnotationError("batch_size must be a positive integer")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise AvidaVhhAnnotationError("chunk_size must be a positive integer")
    if type(ncpu) is not int or ncpu <= 0:
        raise AvidaVhhAnnotationError("ncpu must be a positive integer")
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    protocol, protocol_sha256 = load_avida_annotation_protocol(protocol_path)
    candidate_root = _directory(candidate_root, "candidate root")
    candidate_receipt_path = _regular_file(
        candidate_root / "CANDIDATE_RECEIPT.json",
        "candidate receipt",
    )
    candidate_receipt_sha256 = _file_sha256(candidate_receipt_path)
    if candidate_receipt_sha256 != protocol["input_binding"][
        "candidate_receipt_sha256"
    ]:
        raise AvidaVhhAnnotationError("candidate receipt digest differs from protocol")
    candidate_receipt = _strict_json(candidate_receipt_path, "candidate receipt")
    antibody_path = _authenticate_candidate(candidate_root, candidate_receipt)
    inputs = _load_antibodies(
        antibody_path,
        expected_count=protocol["input_binding"]["expected_antibodies"],
    )
    runtime = _authenticate_anarcii_runtime(
        protocol=protocol,
        wheel_path=anarcii_wheel,
    )
    output_root = Path(output_root).expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise AvidaVhhAnnotationError("annotation output root already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise AvidaVhhAnnotationError("annotation output parent is invalid")
    output_root.mkdir(mode=0o750)
    try:
        factory = numberer_factory or _default_numberer_factory
        numberer = factory(
            seq_type=protocol["vhh_boundary"]["algorithm"]["seq_type_argument"],
            mode=protocol["vhh_boundary"]["algorithm"]["mode"],
            cpu=True,
            batch_size=batch_size,
            ncpu=ncpu,
            verbose=False,
        )
        annotation_path = output_root / "VHH_BOUNDARY_ANNOTATIONS.parquet"
        reason_counts: Counter[str] = Counter()
        status_counts: Counter[str] = Counter()
        with annotation_path.open("xb") as handle:
            with pq.ParquetWriter(
                handle,
                _annotation_schema(),
                compression="zstd",
                compression_level=9,
                use_dictionary=True,
                write_statistics=True,
            ) as writer:
                for offset in range(0, len(inputs), chunk_size):
                    chunk = inputs[offset : offset + chunk_size]
                    sequences = [
                        (record["antibody_id"], record["construct_sequence"])
                        for record in chunk
                    ]
                    numbered = numberer.number(sequences)
                    expected_ids = {record["antibody_id"] for record in chunk}
                    if type(numbered) is not dict or set(numbered) != expected_ids:
                        raise AvidaVhhAnnotationError(
                            "ANARCII result identities differ from input"
                        )
                    annotations = [
                        _interpret_result(
                            antibody_id=record["antibody_id"],
                            construct_sequence=record["construct_sequence"],
                            construct_sequence_digest=record["sequence_digest"],
                            result=numbered[record["antibody_id"]],
                            protocol=protocol,
                        )
                        for record in chunk
                    ]
                    writer.write_table(
                        pa.Table.from_pylist(annotations, schema=_annotation_schema())
                    )
                    reason_counts.update(
                        reason
                        for row in annotations
                        for reason in row["quarantine_reasons"]
                    )
                    status_counts.update(
                        row["annotation_status"] for row in annotations
                    )
        counts = {
            "accepted": status_counts[ACCEPTED_STATUS],
            "input_antibodies": len(inputs),
            "quarantined": status_counts[QUARANTINED_STATUS],
            "quarantine_reasons": dict(sorted(reason_counts.items())),
        }
        if counts["accepted"] + counts["quarantined"] != len(inputs):
            raise AvidaVhhAnnotationError("annotation accounting differs")
        annotation_record = _file_record(annotation_path, output_root)
        audit = {
            "annotation_counts": counts,
            "annotator_source_sha256": _file_sha256(Path(__file__).resolve()),
            "candidate_receipt_sha256": candidate_receipt_sha256,
            "created_at_utc": timestamp,
            "format_version": "pairbench_avida_vhh_annotation_audit_v1",
            "input_antibody_table_sha256": _file_sha256(antibody_path),
            "normalization_eligible": False,
            "normalization_blockers": [
                "antigen_chain_boundary_annotation_pending",
                "independent_vhh_annotation_audit_pending",
            ],
            "protocol_sha256": protocol_sha256,
            "runtime": runtime,
            "execution": {
                "inference_batch_size": batch_size,
                "input_chunk_size": chunk_size,
                "ncpu": ncpu,
            },
            "source_id": SOURCE_ID,
            "table": annotation_record,
            "unknown_or_dropped_antibodies": 0,
        }
        audit_path = output_root / "AUDIT_SUMMARY.json"
        _write_create_only_json(audit_path, audit)
        receipt = {
            "annotation_counts": counts,
            "candidate_receipt_sha256": candidate_receipt_sha256,
            "created_at_utc": timestamp,
            "format_version": "pairbench_avida_vhh_annotation_receipt_v1",
            "normalization_eligible": False,
            "normalization_blockers": audit["normalization_blockers"],
            "outputs": sorted(
                [annotation_record, _file_record(audit_path, output_root)],
                key=lambda value: value["path"],
            ),
            "protocol_sha256": protocol_sha256,
            "source_id": SOURCE_ID,
            "status": "annotation_complete_with_explicit_quarantine",
        }
        _write_create_only_json(output_root / "ANNOTATION_RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        failure_path = output_root / "FAILURE_RECEIPT.json"
        if not failure_path.exists():
            _write_create_only_json(
                failure_path,
                {
                    "created_at_utc": timestamp,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                    "format_version": "pairbench_avida_vhh_annotation_failure_v1",
                    "protocol_sha256": protocol_sha256,
                    "source_id": SOURCE_ID,
                },
            )
        raise


def _interpret_result(
    *,
    antibody_id: str,
    construct_sequence: str,
    construct_sequence_digest: str,
    result: Any,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    reasons: list[str] = []
    acceptance = protocol["vhh_boundary"]["acceptance"]
    prefix_expected = acceptance["expected_construct_prefix"]
    suffix_expected = acceptance["expected_construct_suffix"]
    if not construct_sequence.startswith(prefix_expected):
        reasons.append("construct_prefix_differs")
    if not construct_sequence.endswith(suffix_expected):
        reasons.append("construct_suffix_differs")
    if type(result) is not dict:
        reasons.append("tool_result_not_object")
        result = {}
    tool_error = result.get("error")
    if tool_error is not None:
        if type(tool_error) is not str or not tool_error:
            reasons.append("tool_error_invalid")
            tool_error = repr(tool_error)
        else:
            reasons.append("tool_reported_error")
    score = result.get("score")
    if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
        reasons.append("tool_score_invalid")
        score = None
    else:
        score = float(score)
    query_start = result.get("query_start")
    query_end = result.get("query_end")
    bounds_valid = (
        type(query_start) is int
        and not isinstance(query_start, bool)
        and type(query_end) is int
        and not isinstance(query_end, bool)
        and 0 <= query_start <= query_end < len(construct_sequence)
    )
    if not bounds_valid:
        reasons.append("query_bounds_invalid")
        query_start = None
        query_end = None
    prefix_sequence = (
        construct_sequence[:query_start] if query_start is not None else None
    )
    suffix_sequence = (
        construct_sequence[query_end + 1 :] if query_end is not None else None
    )
    domain_sequence = (
        construct_sequence[query_start : query_end + 1]
        if query_start is not None and query_end is not None
        else None
    )
    if query_start is not None and query_start != acceptance[
        "expected_query_start_zero_based_inclusive"
    ]:
        reasons.append("query_start_differs")
    if prefix_sequence is not None and prefix_sequence != prefix_expected:
        reasons.append("query_prefix_differs")
    if suffix_sequence is not None and suffix_sequence != suffix_expected:
        reasons.append("query_suffix_differs")
    scheme = result.get("scheme")
    if scheme != acceptance["scheme"]:
        reasons.append("scheme_differs")
    chain_type = result.get("chain_type")
    if chain_type != acceptance["chain_type"]:
        reasons.append("chain_type_differs")
    numbering, numbering_residues = _canonical_numbering(result.get("numbering"), reasons)
    if domain_sequence is None or numbering_residues != domain_sequence:
        reasons.append("numbering_residues_differ")
    cdrs = protocol["vhh_boundary"]["imgt_cdr_ranges_one_based_inclusive"]
    cdr_sequences = {
        name: _numbering_region(numbering, limits[0], limits[1])
        if numbering is not None
        else None
        for name, limits in cdrs.items()
    }
    canonical_reasons = sorted(set(reasons))
    accepted = not canonical_reasons
    return {
        "annotation_status": ACCEPTED_STATUS if accepted else QUARANTINED_STATUS,
        "antibody_id": antibody_id,
        "cdr1_sequence": cdr_sequences["cdr1"],
        "cdr2_sequence": cdr_sequences["cdr2"],
        "cdr3_sequence": cdr_sequences["cdr3"],
        "chain_type": chain_type if type(chain_type) is str else None,
        "construct_sequence_digest": construct_sequence_digest,
        "domain_sequence": domain_sequence,
        "domain_sequence_digest": (
            hashlib.sha256(domain_sequence.encode()).hexdigest()
            if domain_sequence is not None
            else None
        ),
        "imgt_numbering": numbering,
        "prefix_sequence": prefix_sequence,
        "quarantine_reasons": canonical_reasons,
        "query_end_zero_based_inclusive": query_end,
        "query_start_zero_based_inclusive": query_start,
        "scheme": scheme if type(scheme) is str else None,
        "suffix_sequence": suffix_sequence,
        "tool_error": tool_error,
        "tool_score": score,
    }


def _canonical_numbering(
    value: Any,
    reasons: list[str],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if type(value) is not list:
        reasons.append("numbering_not_list")
        return None, None
    result: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    core_positions: set[int] = set()
    for entry in value:
        if type(entry) not in {tuple, list} or len(entry) != 2:
            reasons.append("numbering_entry_invalid")
            return None, None
        coordinate, residue = entry
        if type(coordinate) not in {tuple, list} or len(coordinate) != 2:
            reasons.append("numbering_coordinate_invalid")
            return None, None
        position, insertion = coordinate
        if (
            type(position) is not int
            or isinstance(position, bool)
            or not 1 <= position <= 128
            or type(insertion) is not str
            or (insertion.strip() and not re.fullmatch(r"[A-Z]", insertion.strip()))
            or type(residue) is not str
            or not re.fullmatch(r"[A-Z-]", residue)
        ):
            reasons.append("numbering_value_invalid")
            return None, None
        canonical_insertion = insertion.strip()
        key = (position, canonical_insertion)
        if key in keys:
            reasons.append("numbering_coordinate_duplicated")
            return None, None
        keys.add(key)
        if not canonical_insertion:
            core_positions.add(position)
        result.append(
            {
                "insertion": canonical_insertion,
                "position": position,
                "residue": residue,
            }
        )
    if core_positions != set(range(1, 129)):
        reasons.append("numbering_core_positions_incomplete")
    residues = "".join(record["residue"] for record in result if record["residue"] != "-")
    return result, residues


def _numbering_region(
    numbering: Sequence[Mapping[str, Any]],
    start: int,
    end: int,
) -> str:
    return "".join(
        record["residue"]
        for record in numbering
        if start <= record["position"] <= end and record["residue"] != "-"
    )


def _authenticate_candidate(
    candidate_root: Path,
    receipt: Mapping[str, Any],
) -> Path:
    if (
        receipt.get("format_version") != "pairbench_avida_candidate_receipt_v1"
        or receipt.get("source_id") != SOURCE_ID
        or receipt.get("status") != "candidate_complete_blocked"
        or receipt.get("registry_normalization_eligible") is not False
    ):
        raise AvidaVhhAnnotationError("candidate receipt authority differs")
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaVhhAnnotationError("candidate output inventory differs")
    by_path = {
        record.get("path"): record for record in records if type(record) is dict
    }
    record = by_path.get("entities_antibody.parquet")
    if type(record) is not dict or set(record) != {"path", "sha256", "size_bytes"}:
        raise AvidaVhhAnnotationError("candidate antibody record differs")
    path = _regular_file(
        candidate_root / "entities_antibody.parquet",
        "candidate antibody table",
    )
    if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record[
        "sha256"
    ]:
        raise AvidaVhhAnnotationError("candidate antibody table bytes differ")
    return path


def _load_antibodies(path: Path, *, expected_count: int) -> list[Mapping[str, str]]:
    table = pq.read_table(
        path,
        columns=[
            "annotation_status",
            "antibody_id",
            "construct_sequence",
            "sequence_digest",
            "source_id",
        ],
    )
    if table.num_rows != expected_count:
        raise AvidaVhhAnnotationError("candidate antibody count differs")
    records = table.to_pylist()
    identities: list[str] = []
    for record in records:
        sequence = record["construct_sequence"]
        identity = record["antibody_id"]
        if type(sequence) is not str or not sequence:
            raise AvidaVhhAnnotationError("candidate antibody sequence is invalid")
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        if (
            _ID.fullmatch(identity or "") is None
            or identity != f"ab_sha256:{digest}"
            or record["sequence_digest"] != digest
            or record["annotation_status"] != INPUT_STATUS
            or record["source_id"] != SOURCE_ID
        ):
            raise AvidaVhhAnnotationError("candidate antibody semantics differ")
        identities.append(identity)
    if identities != sorted(set(identities)):
        raise AvidaVhhAnnotationError("candidate antibodies are not canonical and unique")
    return records


def _authenticate_anarcii_runtime(
    *,
    protocol: Mapping[str, Any],
    wheel_path: Path,
) -> Mapping[str, Any]:
    algorithm = protocol["vhh_boundary"]["algorithm"]
    wheel_path = _regular_file(wheel_path, "ANARCII wheel")
    if _file_sha256(wheel_path) != algorithm["wheel_sha256"]:
        raise AvidaVhhAnnotationError("ANARCII wheel digest differs")
    try:
        distribution = metadata.distribution(algorithm["package"])
    except metadata.PackageNotFoundError as exc:
        raise AvidaVhhAnnotationError("pinned ANARCII package is not installed") from exc
    if distribution.version != algorithm["package_version"]:
        raise AvidaVhhAnnotationError("installed ANARCII version differs")
    installed_root = Path(distribution.locate_file("")).absolute()
    files = {
        "license": installed_root / "anarcii-2.0.8.dist-info/licenses/LICENCE",
        "parameters": installed_root / algorithm["model_parameters_path"],
        "weights": installed_root / algorithm["model_weights_path"],
    }
    expected = {
        "license": algorithm["license_file_sha256"],
        "parameters": algorithm["model_parameters_sha256"],
        "weights": algorithm["model_weights_sha256"],
    }
    for name, path in files.items():
        if _file_sha256(_regular_file(path, f"installed ANARCII {name}")) != expected[name]:
            raise AvidaVhhAnnotationError(f"installed ANARCII {name} digest differs")
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError as exc:
        raise AvidaVhhAnnotationError("torch runtime is unavailable") from exc
    return {
        "anarcii_version": distribution.version,
        "batch_device": "cpu",
        "model_parameters_sha256": expected["parameters"],
        "model_weights_sha256": expected["weights"],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "wheel_sha256": algorithm["wheel_sha256"],
    }


def _default_numberer_factory(**arguments: Any) -> Any:
    from anarcii import Anarcii

    return Anarcii(**arguments)


def _annotation_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("annotation_status", pa.string(), nullable=False),
            pa.field("antibody_id", pa.string(), nullable=False),
            pa.field("cdr1_sequence", pa.string()),
            pa.field("cdr2_sequence", pa.string()),
            pa.field("cdr3_sequence", pa.string()),
            pa.field("chain_type", pa.string()),
            pa.field("construct_sequence_digest", pa.string(), nullable=False),
            pa.field("domain_sequence", pa.string()),
            pa.field("domain_sequence_digest", pa.string()),
            pa.field(
                "imgt_numbering",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("insertion", pa.string(), nullable=False),
                            pa.field("position", pa.int16(), nullable=False),
                            pa.field("residue", pa.string(), nullable=False),
                        ]
                    )
                ),
            ),
            pa.field("prefix_sequence", pa.string()),
            pa.field("quarantine_reasons", pa.list_(pa.string()), nullable=False),
            pa.field("query_end_zero_based_inclusive", pa.int32()),
            pa.field("query_start_zero_based_inclusive", pa.int32()),
            pa.field("scheme", pa.string()),
            pa.field("suffix_sequence", pa.string()),
            pa.field("tool_error", pa.string()),
            pa.field("tool_score", pa.float64()),
        ]
    )


def _file_record(path: Path, root: Path) -> Mapping[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaVhhAnnotationError(f"non-finite JSON constant in {label}: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaVhhAnnotationError(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaVhhAnnotationError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise AvidaVhhAnnotationError(f"{label} must be an object")
    return value


def _directory(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaVhhAnnotationError(f"{label} is invalid")
    return result


def _regular_file(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaVhhAnnotationError(f"{label} is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _utc(value: str) -> None:
    if type(value) is not str or not value.endswith("Z"):
        raise AvidaVhhAnnotationError("annotation timestamp must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaVhhAnnotationError("annotation timestamp is invalid") from exc
