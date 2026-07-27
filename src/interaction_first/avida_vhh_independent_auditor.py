"""Independently replay and audit AVIDa VHH boundary annotations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from itertools import zip_longest
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import pyarrow.parquet as pq


SOURCE_ID = "avida_hil6_2024_11"
PROTOCOL_SHA256 = "63d88b7ff3f37c80d9478cad27cb11d9099258f83e15b49c6313684a2c2d0cca"
CANDIDATE_RECEIPT_SHA256 = "eceb95ab6e94e711fd3a7ad48e6bf42bf05f6b0d2a455a7921597ec2b0832dbe"
INPUT_TABLE_SHA256 = "a253648900624160577dc5122c3795b072ce48ea7eaacefb2912483033b6512a"
WHEEL_SHA256 = "6ac81caef04a9cad1af1ea320b35bdced620fe2bfdd0b8b17b98131495b584ca"
MODEL_SHA256 = "03e9d07d13a3ec98a9d9fe57a0b2ff0853218aa2d7bdd50a25b6d83ff2bea2e0"
PARAMETERS_SHA256 = "4d8e5fc9d8580f5e26a190a21853dd3f3770fbf83ce45a46a53d92b083063c39"
ACCEPTED_STATUS = "imgt_heavy_domain_verified"
QUARANTINED_STATUS = "quarantined_vhh_boundary"
PREFIX = "MKYLLPTAAAGLLLLAAQPAMA"
SUFFIX = "HHHHHH"
_ANTIBODY_ID = re.compile(r"^ab_sha256:[0-9a-f]{64}$")


class AvidaVhhIndependentAuditError(ValueError):
    """Raised when independently replayed VHH evidence disagrees."""


def audit_avida_vhh_annotations(
    *,
    candidate_root: Path,
    annotation_root: Path,
    protocol_path: Path,
    anarcii_wheel: Path,
    annotator_source: Path,
    output_receipt: Path,
    batch_size: int = 512,
    chunk_size: int = 4_096,
    ncpu: int = 4,
    audited_at_utc: str | None = None,
) -> Mapping[str, Any]:
    """Replay ANARCII from candidate bytes and compare every annotation field."""

    for value, label in (
        (batch_size, "batch_size"),
        (chunk_size, "chunk_size"),
        (ncpu, "ncpu"),
    ):
        if type(value) is not int or value <= 0:
            raise AvidaVhhIndependentAuditError(f"{label} must be positive")
    timestamp = audited_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _utc(timestamp)
    candidate_root = _directory(candidate_root, "candidate root")
    annotation_root = _directory(annotation_root, "annotation root")
    protocol_path = _regular_file(protocol_path, "annotation protocol")
    anarcii_wheel = _regular_file(anarcii_wheel, "ANARCII wheel")
    annotator_source = _regular_file(annotator_source, "annotator source")
    if _file_sha256(protocol_path) != PROTOCOL_SHA256:
        raise AvidaVhhIndependentAuditError("annotation protocol digest differs")
    _audit_protocol(_strict_json(protocol_path, "annotation protocol"))
    _audit_runtime(anarcii_wheel)
    candidate_receipt_path = _regular_file(
        candidate_root / "CANDIDATE_RECEIPT.json",
        "candidate receipt",
    )
    if _file_sha256(candidate_receipt_path) != CANDIDATE_RECEIPT_SHA256:
        raise AvidaVhhIndependentAuditError("candidate receipt digest differs")
    candidate_table = _candidate_table(
        candidate_root,
        _strict_json(candidate_receipt_path, "candidate receipt"),
    )
    annotation_receipt_path = _regular_file(
        annotation_root / "ANNOTATION_RECEIPT.json",
        "annotation receipt",
    )
    annotation_receipt = _strict_json(annotation_receipt_path, "annotation receipt")
    outputs = _audit_annotation_root(annotation_root, annotation_receipt)
    audit_summary = _strict_json(
        annotation_root / "AUDIT_SUMMARY.json",
        "annotation audit summary",
    )
    if audit_summary.get("annotator_source_sha256") != _file_sha256(annotator_source):
        raise AvidaVhhIndependentAuditError("annotator source digest differs")
    from anarcii import Anarcii

    numberer = Anarcii(
        seq_type="antibody",
        mode="accuracy",
        cpu=True,
        batch_size=batch_size,
        ncpu=ncpu,
        verbose=False,
    )
    annotation_table = annotation_root / "VHH_BOUNDARY_ANNOTATIONS.parquet"
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    replayed_rows = 0
    maximum_score_difference = 0.0
    candidate_batches = _parquet_batches(candidate_table, chunk_size)
    annotation_batches = _parquet_batches(annotation_table, chunk_size)
    sentinel = object()
    for candidate_batch, annotation_batch in zip_longest(
        candidate_batches,
        annotation_batches,
        fillvalue=sentinel,
    ):
        if candidate_batch is sentinel or annotation_batch is sentinel:
            raise AvidaVhhIndependentAuditError("VHH annotation batch counts differ")
        if len(candidate_batch) != len(annotation_batch):
            raise AvidaVhhIndependentAuditError("VHH annotation batch sizes differ")
        inputs = [_candidate_record(row) for row in candidate_batch]
        numbered = numberer.number(
            [(row["antibody_id"], row["construct_sequence"]) for row in inputs]
        )
        expected_ids = {row["antibody_id"] for row in inputs}
        if type(numbered) is not dict or set(numbered) != expected_ids:
            raise AvidaVhhIndependentAuditError("ANARCII replay identities differ")
        for candidate, observed in zip(inputs, annotation_batch, strict=True):
            expected = _expected_annotation(
                antibody_id=candidate["antibody_id"],
                construct=candidate["construct_sequence"],
                construct_digest=candidate["sequence_digest"],
                result=numbered[candidate["antibody_id"]],
            )
            score_difference = _compare_annotation(expected, observed)
            maximum_score_difference = max(maximum_score_difference, score_difference)
            replayed_rows += 1
            status_counts[expected["annotation_status"]] += 1
            reason_counts.update(expected["quarantine_reasons"])
    counts = {
        "accepted": status_counts[ACCEPTED_STATUS],
        "input_antibodies": replayed_rows,
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "quarantined": status_counts[QUARANTINED_STATUS],
    }
    if replayed_rows != 38_599 or sum(status_counts.values()) != replayed_rows:
        raise AvidaVhhIndependentAuditError("VHH replay accounting differs")
    if (
        annotation_receipt.get("annotation_counts") != counts
        or audit_summary.get("annotation_counts") != counts
        or annotation_receipt.get("candidate_receipt_sha256")
        != CANDIDATE_RECEIPT_SHA256
        or annotation_receipt.get("protocol_sha256") != PROTOCOL_SHA256
        or annotation_receipt.get("normalization_eligible") is not False
        or annotation_receipt.get("status")
        != "annotation_complete_with_explicit_quarantine"
        or audit_summary.get("unknown_or_dropped_antibodies") != 0
        or audit_summary.get("table") != outputs["VHH_BOUNDARY_ANNOTATIONS.parquet"]
        or audit_summary.get("execution")
        != {
            "inference_batch_size": batch_size,
            "input_chunk_size": chunk_size,
            "ncpu": ncpu,
        }
    ):
        raise AvidaVhhIndependentAuditError("VHH annotation receipt semantics differ")
    result = {
        "annotation_counts": counts,
        "annotation_receipt_sha256": _file_sha256(annotation_receipt_path),
        "audited_at_utc": timestamp,
        "auditor_source_sha256": _file_sha256(Path(__file__).resolve()),
        "candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA256,
        "format_version": "pairbench_avida_vhh_independent_replay_v1",
        "maximum_tool_score_absolute_difference": maximum_score_difference,
        "normalization_eligible": False,
        "normalization_blockers": [
            "tier2_boundary_policy_not_frozen",
            "unified_avida_normalization_receipt_pending",
        ],
        "protocol_sha256": PROTOCOL_SHA256,
        "replay": {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "device": "cpu",
            "ncpu": ncpu,
        },
        "semantic_comparison": "exact_all_fields_with_1e-12_tool_score_tolerance",
        "source_id": SOURCE_ID,
        "status": "pass_vhh_annotation_strict_tier1",
        "wheel_sha256": WHEEL_SHA256,
    }
    output_receipt = Path(output_receipt).expanduser().absolute()
    if not output_receipt.parent.is_dir() or output_receipt.parent.is_symlink():
        raise AvidaVhhIndependentAuditError("audit receipt parent is invalid")
    _write_create_only_json(output_receipt, result)
    return result


def _audit_protocol(payload: Mapping[str, Any]) -> None:
    boundary = payload.get("vhh_boundary")
    if type(boundary) is not dict:
        raise AvidaVhhIndependentAuditError("protocol VHH section differs")
    algorithm = boundary.get("algorithm")
    acceptance = boundary.get("acceptance")
    coordinates = boundary.get("coordinate_semantics")
    cdrs = boundary.get("imgt_cdr_ranges_one_based_inclusive")
    if (
        payload.get("format_version")
        != "pairscale_avida_hil6_annotation_protocol_v1"
        or payload.get("source_id") != SOURCE_ID
        or payload.get("score_free") is not True
        or type(algorithm) is not dict
        or algorithm.get("package") != "anarcii"
        or algorithm.get("package_version") != "2.0.8"
        or algorithm.get("seq_type_argument") != "antibody"
        or algorithm.get("rejected_seq_type_aliases") != ["vhh", "vnar"]
        or algorithm.get("mode") != "accuracy"
        or algorithm.get("wheel_sha256") != WHEEL_SHA256
        or algorithm.get("model_weights_sha256") != MODEL_SHA256
        or algorithm.get("model_parameters_sha256") != PARAMETERS_SHA256
        or type(acceptance) is not dict
        or acceptance.get("expected_construct_prefix") != PREFIX
        or acceptance.get("expected_construct_suffix") != SUFFIX
        or acceptance.get("expected_query_start_zero_based_inclusive") != 22
        or acceptance.get("chain_type") != "H"
        or acceptance.get("scheme") != "imgt"
        or coordinates
        != {
            "domain_python_slice": "construct_sequence[query_start:query_end+1]",
            "query_end": "zero_based_inclusive",
            "query_start": "zero_based_inclusive",
        }
        or cdrs != {"cdr1": [27, 38], "cdr2": [56, 65], "cdr3": [105, 117]}
    ):
        raise AvidaVhhIndependentAuditError("protocol VHH semantics differ")


def _audit_runtime(wheel: Path) -> None:
    if _file_sha256(wheel) != WHEEL_SHA256:
        raise AvidaVhhIndependentAuditError("ANARCII wheel digest differs")
    try:
        distribution = metadata.distribution("anarcii")
    except metadata.PackageNotFoundError as exc:
        raise AvidaVhhIndependentAuditError("ANARCII runtime is unavailable") from exc
    if distribution.version != "2.0.8":
        raise AvidaVhhIndependentAuditError("ANARCII runtime version differs")
    root = Path(distribution.locate_file("")).absolute()
    assets = {
        root / "anarcii/models/antibody/antibody_4_2_128_512.pt": MODEL_SHA256,
        root / "anarcii/models/antibody/antibody_4_2_128_512.json": PARAMETERS_SHA256,
    }
    for path, digest in assets.items():
        if _file_sha256(_regular_file(path, "ANARCII model asset")) != digest:
            raise AvidaVhhIndependentAuditError("ANARCII model asset digest differs")


def _candidate_table(root: Path, receipt: Mapping[str, Any]) -> Path:
    if (
        receipt.get("format_version") != "pairbench_avida_candidate_receipt_v1"
        or receipt.get("source_id") != SOURCE_ID
        or receipt.get("counts", {}).get("antibodies") != 38_599
    ):
        raise AvidaVhhIndependentAuditError("candidate receipt semantics differ")
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaVhhIndependentAuditError("candidate inventory differs")
    matches = [
        record
        for record in records
        if type(record) is dict and record.get("path") == "entities_antibody.parquet"
    ]
    if len(matches) != 1 or matches[0].get("sha256") != INPUT_TABLE_SHA256:
        raise AvidaVhhIndependentAuditError("candidate antibody record differs")
    path = _regular_file(root / "entities_antibody.parquet", "candidate antibody table")
    if (
        path.stat().st_size != matches[0].get("size_bytes")
        or _file_sha256(path) != INPUT_TABLE_SHA256
    ):
        raise AvidaVhhIndependentAuditError("candidate antibody table bytes differ")
    return path


def _candidate_record(row: Mapping[str, Any]) -> Mapping[str, str]:
    sequence = row.get("construct_sequence")
    if type(sequence) is not str:
        raise AvidaVhhIndependentAuditError("candidate VHH sequence differs")
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    identity = row.get("antibody_id")
    if (
        _ANTIBODY_ID.fullmatch(identity or "") is None
        or identity != f"ab_sha256:{digest}"
        or row.get("sequence_digest") != digest
        or row.get("annotation_status")
        != "raw_construct_untrimmed_signal_and_tag_boundaries_pending"
        or row.get("source_id") != SOURCE_ID
    ):
        raise AvidaVhhIndependentAuditError("candidate VHH row semantics differ")
    return {
        "antibody_id": identity,
        "construct_sequence": sequence,
        "sequence_digest": digest,
    }


def _expected_annotation(
    *,
    antibody_id: str,
    construct: str,
    construct_digest: str,
    result: Any,
) -> Mapping[str, Any]:
    reasons: list[str] = []
    if not construct.startswith(PREFIX):
        reasons.append("construct_prefix_differs")
    if not construct.endswith(SUFFIX):
        reasons.append("construct_suffix_differs")
    if type(result) is not dict:
        reasons.append("tool_result_not_object")
        result = {}
    error = result.get("error")
    if error is not None:
        if type(error) is not str or not error:
            reasons.append("tool_error_invalid")
            error = repr(error)
        else:
            reasons.append("tool_reported_error")
    score = result.get("score")
    if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
        reasons.append("tool_score_invalid")
        score = None
    else:
        score = float(score)
    start = result.get("query_start")
    end = result.get("query_end")
    if not (
        type(start) is int
        and not isinstance(start, bool)
        and type(end) is int
        and not isinstance(end, bool)
        and 0 <= start <= end < len(construct)
    ):
        reasons.append("query_bounds_invalid")
        start = None
        end = None
    prefix = construct[:start] if start is not None else None
    suffix = construct[end + 1 :] if end is not None else None
    domain = construct[start : end + 1] if start is not None and end is not None else None
    if start is not None and start != 22:
        reasons.append("query_start_differs")
    if prefix is not None and prefix != PREFIX:
        reasons.append("query_prefix_differs")
    if suffix is not None and suffix != SUFFIX:
        reasons.append("query_suffix_differs")
    scheme = result.get("scheme")
    chain = result.get("chain_type")
    if scheme != "imgt":
        reasons.append("scheme_differs")
    if chain != "H":
        reasons.append("chain_type_differs")
    numbering, residues = _canonical_numbering(result.get("numbering"), reasons)
    if domain is None or residues != domain:
        reasons.append("numbering_residues_differ")
    canonical_reasons = sorted(set(reasons))
    return {
        "annotation_status": ACCEPTED_STATUS if not canonical_reasons else QUARANTINED_STATUS,
        "antibody_id": antibody_id,
        "cdr1_sequence": _region(numbering, 27, 38),
        "cdr2_sequence": _region(numbering, 56, 65),
        "cdr3_sequence": _region(numbering, 105, 117),
        "chain_type": chain if type(chain) is str else None,
        "construct_sequence_digest": construct_digest,
        "domain_sequence": domain,
        "domain_sequence_digest": hashlib.sha256(domain.encode()).hexdigest() if domain else None,
        "imgt_numbering": numbering,
        "prefix_sequence": prefix,
        "quarantine_reasons": canonical_reasons,
        "query_end_zero_based_inclusive": end,
        "query_start_zero_based_inclusive": start,
        "scheme": scheme if type(scheme) is str else None,
        "suffix_sequence": suffix,
        "tool_error": error,
        "tool_score": score,
    }


def _canonical_numbering(
    value: Any,
    reasons: list[str],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if type(value) is not list:
        reasons.append("numbering_not_list")
        return None, None
    records: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    core: set[int] = set()
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
            or (insertion.strip() and re.fullmatch(r"[A-Z]", insertion.strip()) is None)
            or type(residue) is not str
            or re.fullmatch(r"[A-Z-]", residue) is None
        ):
            reasons.append("numbering_value_invalid")
            return None, None
        key = (position, insertion.strip())
        if key in keys:
            reasons.append("numbering_coordinate_duplicated")
            return None, None
        keys.add(key)
        if not key[1]:
            core.add(position)
        records.append(
            {"insertion": key[1], "position": position, "residue": residue}
        )
    if core != set(range(1, 129)):
        reasons.append("numbering_core_positions_incomplete")
    return records, "".join(
        record["residue"] for record in records if record["residue"] != "-"
    )


def _region(
    numbering: Sequence[Mapping[str, Any]] | None,
    start: int,
    end: int,
) -> str | None:
    if numbering is None:
        return None
    return "".join(
        record["residue"]
        for record in numbering
        if start <= record["position"] <= end and record["residue"] != "-"
    )


def _compare_annotation(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> float:
    expected_score = expected["tool_score"]
    observed_score = observed.get("tool_score")
    without_score = dict(observed)
    without_score.pop("tool_score", None)
    expected_without_score = dict(expected)
    expected_without_score.pop("tool_score")
    if without_score != expected_without_score:
        raise AvidaVhhIndependentAuditError(
            f"VHH annotation differs: {expected['antibody_id']}"
        )
    if expected_score is None or observed_score is None:
        if expected_score is not observed_score:
            raise AvidaVhhIndependentAuditError("VHH tool score nullability differs")
        return 0.0
    difference = abs(float(expected_score) - float(observed_score))
    if difference > 1e-12:
        raise AvidaVhhIndependentAuditError(
            f"VHH tool score differs: {expected['antibody_id']}"
        )
    return difference


def _audit_annotation_root(
    root: Path,
    receipt: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    records = receipt.get("outputs")
    if type(records) is not list:
        raise AvidaVhhIndependentAuditError("annotation inventory differs")
    by_path = {
        record.get("path"): record for record in records if type(record) is dict
    }
    expected = {"AUDIT_SUMMARY.json", "VHH_BOUNDARY_ANNOTATIONS.parquet"}
    if set(by_path) != expected:
        raise AvidaVhhIndependentAuditError("annotation output records differ")
    observed: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise AvidaVhhIndependentAuditError(
                "annotation root contains a non-regular entry"
            )
        observed.add(path.relative_to(root).as_posix())
    if observed != expected | {"ANNOTATION_RECEIPT.json"}:
        raise AvidaVhhIndependentAuditError("annotation root file set differs")
    for name, record in by_path.items():
        if set(record) != {"path", "sha256", "size_bytes"}:
            raise AvidaVhhIndependentAuditError("annotation output record differs")
        path = _regular_file(root / name, f"annotation output {name}")
        if path.stat().st_size != record["size_bytes"] or _file_sha256(path) != record[
            "sha256"
        ]:
            raise AvidaVhhIndependentAuditError("annotation output bytes differ")
    return by_path


def _parquet_batches(path: Path, batch_size: int) -> Iterator[list[Mapping[str, Any]]]:
    path = _regular_file(path, "Parquet table")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield batch.to_pylist()


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise AvidaVhhIndependentAuditError(
            f"non-finite JSON constant in {label}: {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AvidaVhhIndependentAuditError(f"duplicate JSON key in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AvidaVhhIndependentAuditError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise AvidaVhhIndependentAuditError(f"{label} must be an object")
    return value


def _directory(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_dir():
        raise AvidaVhhIndependentAuditError(f"{label} is invalid")
    return result


def _regular_file(path: Path, label: str) -> Path:
    result = Path(path).expanduser().absolute()
    if result.is_symlink() or not result.is_file():
        raise AvidaVhhIndependentAuditError(f"{label} is invalid")
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
        raise AvidaVhhIndependentAuditError("audit timestamp must use UTC Z form")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AvidaVhhIndependentAuditError("audit timestamp is invalid") from exc
