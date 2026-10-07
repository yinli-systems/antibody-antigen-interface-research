#!/usr/bin/env python3
"""Independently audit provisional PINDER-antibody S2 folds in four views.

The audit authenticates the provisional assignment and canonical sequence
manifest, reconstructs every retained/discarded ledger from source rows, and
exhaustively aligns every cross-fold unique-sequence pair in VH, non-empty VL,
concatenated-antibody, and antigen views.  Published crossing artifacts contain
digests and alignment metrics only, never raw sequences or system identifiers.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import math
import multiprocessing
import os
import platform
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence


FORMAT_VERSION = "pinder_antibody_s2_exact_four_view_nw_audit_v1"
EDGE_FORMAT_VERSION = "pinder_antibody_s2_exact_nw_crossing_edge_v1"
ASSIGNMENT_FORMAT_VERSION = "pinder_antibody_provisional_s2_fold_assignment_v1"
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
FOLD_INDICES = (0, 1, 2)
FOLD_PAIRS = ((0, 1), (0, 2), (1, 2))
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
IDENTITY_NUMERATOR = 4
IDENTITY_DENOMINATOR = 5
COVERAGE_NUMERATOR = 4
COVERAGE_DENOMINATOR = 5
GAP_OPEN_PENALTY = 10.0
GAP_EXTEND_PENALTY = 1.0
EXPECTED_BIOPYTHON_VERSION = "1.87"
EXPECTED_BLOSUM62_PAYLOAD_SHA256 = (
    "2c62ca4bb7a4c3ec5fe5faee64276841028a7e6638cf846c801dd9c5e64527e2"
)


class AuditError(RuntimeError):
    """Raised when an authenticated input or audit invariant fails."""


@dataclass(frozen=True)
class SourceRow:
    system_id: str
    pdb_id: str
    vh: str
    vl: str
    antigen: str

    @property
    def concatenated_antibody(self) -> str:
        return self.vh + self.vl

    @property
    def antibody_digest(self) -> str:
        return sequence_sha256(self.concatenated_antibody)

    @property
    def antigen_digest(self) -> str:
        return sequence_sha256(self.antigen)

    @property
    def exact_pair_id(self) -> str:
        return sequence_sha256(f"{self.antibody_digest}\0{self.antigen_digest}")


@dataclass(frozen=True)
class AlignmentMetrics:
    score: float
    identical_residue_pair_count: int
    aligned_residue_pair_count: int
    alignment_column_count: int
    aligned_region_identity: float
    left_reciprocal_coverage: float
    right_reciprocal_coverage: float
    minimum_reciprocal_coverage: float


@dataclass(frozen=True)
class StagedArtifact:
    temporary_path: Path
    target_path: Path
    sha256: str
    size_bytes: int
    edge_count: int


_WORKER_LEFT: tuple[tuple[str, str], ...] = ()
_WORKER_RIGHT: tuple[tuple[str, str], ...] = ()
_WORKER_VIEW = ""
_WORKER_LEFT_FOLD = -1
_WORKER_RIGHT_FOLD = -1
_WORKER_ALIGNER: object | None = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    encoded = "".join(f"{value}\n" for value in sorted(set(values))).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def verify_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise AuditError(f"{label} is not a regular file")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise AuditError(
            f"{label} SHA mismatch: expected={expected_sha256}, observed={observed}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuditError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise AuditError(f"non-finite JSON constant: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise AuditError("JSON root must be an object")

    def validate_finite(value: Any, label: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise AuditError(f"non-finite JSON number: {label}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                validate_finite(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                validate_finite(child, f"{label}[{index}]")

    validate_finite(payload, "root")
    return payload


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence and allow_empty:
        return ""
    if not sequence:
        raise AuditError(f"empty sequence: {label}")
    unexpected = set(sequence) - VALID_AMINO_ACIDS
    if unexpected:
        raise AuditError(f"noncanonical sequence {label}: {sorted(unexpected)}")
    return sequence


def load_source_rows(path: Path) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"INSTANCE", "pdb_id", "vh_seq", "vl_seq", "ag_seq"}
        if not required.issubset(reader.fieldnames or []):
            raise AuditError("canonical manifest schema mismatch")
        for raw in reader:
            system_id = raw["INSTANCE"]
            pdb_id = raw["pdb_id"].lower()
            if not system_id or not pdb_id:
                raise AuditError("empty system or PDB identifier")
            rows.append(
                SourceRow(
                    system_id=system_id,
                    pdb_id=pdb_id,
                    vh=normalize_sequence(raw["vh_seq"], f"{system_id}:VH"),
                    vl=normalize_sequence(
                        raw["vl_seq"], f"{system_id}:VL", allow_empty=True
                    ),
                    antigen=normalize_sequence(raw["ag_seq"], f"{system_id}:Ag"),
                )
            )
    if len(rows) != len({row.system_id for row in rows}):
        raise AuditError("canonical manifest has duplicate system identifiers")
    return sorted(rows, key=lambda row: row.system_id)


def validate_membership_ledger(
    values: Any,
    claimed_count: Any,
    claimed_sha256: Any,
    label: str,
) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise AuditError(f"{label} must be a string list")
    if values != sorted(values) or len(values) != len(set(values)):
        raise AuditError(f"{label} must be unique and sorted")
    if claimed_count != len(values):
        raise AuditError(f"{label} count mismatch")
    if claimed_sha256 != membership_sha256(values):
        raise AuditError(f"{label} membership SHA mismatch")
    return values


def validate_assignment(
    assignment: Mapping[str, Any],
    rows: Sequence[SourceRow],
    expected_scientific_sha256: str,
) -> tuple[dict[str, int], set[str], dict[str, Any]]:
    if assignment.get("format_version") != ASSIGNMENT_FORMAT_VERSION:
        raise AuditError("assignment format mismatch")
    if assignment.get("status") != (
        "PASS_PROVISIONAL_PDB_AND_EXACT_PAIR_INDIVISIBLE_S2_80_ASSIGNMENT"
    ):
        raise AuditError("assignment did not pass its provisional gate")
    if assignment.get("approval_status") != (
        "UNAPPROVED_PENDING_EXACT_FOUR_VIEW_NEEDLEMAN_WUNSCH_AUDIT"
    ):
        raise AuditError("assignment approval boundary mismatch")
    claimed_scientific = assignment.get("scientific_payload_sha256")
    scientific = dict(assignment)
    scientific.pop("scientific_payload_sha256", None)
    observed_scientific = canonical_payload_sha256(scientific)
    if (
        claimed_scientific != expected_scientific_sha256
        or observed_scientific != expected_scientific_sha256
    ):
        raise AuditError("assignment scientific payload SHA mismatch")
    if assignment.get("authorized_for_training") is not False:
        raise AuditError("provisional assignment was unexpectedly training-authorized")
    if assignment.get("authorized_for_model_selection") is not False:
        raise AuditError("provisional assignment was unexpectedly selection-authorized")
    checks = assignment.get("checks")
    if not isinstance(checks, Mapping):
        raise AuditError("assignment checks missing")
    if checks.get("validation_rows_read") is not False:
        raise AuditError("assignment read validation rows")
    if checks.get("test_rows_read") is not False:
        raise AuditError("assignment read test rows")
    input_counts = assignment.get("input_counts")
    if not isinstance(input_counts, Mapping):
        raise AuditError("assignment input counts missing")
    if input_counts.get("canonical_systems") != len(rows):
        raise AuditError("assignment canonical source count mismatch")

    source_by_id = {row.system_id: row for row in rows}
    all_ids = set(source_by_id)
    fold_by_system: dict[str, int] = {}
    retained_pair_sets: list[set[str]] = []
    retained_pdb_sets: list[set[str]] = []
    folds = assignment.get("retained_folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise AuditError("assignment must contain three retained folds")
    for expected_fold, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or fold.get("fold_index") != expected_fold:
            raise AuditError("retained fold ordering mismatch")
        system_ids = validate_membership_ledger(
            fold.get("retained_system_ids"),
            fold.get("retained_system_count"),
            fold.get("retained_system_membership_sha256"),
            f"fold {expected_fold} systems",
        )
        pair_ids = validate_membership_ledger(
            fold.get("retained_exact_pair_ids"),
            fold.get("retained_unique_exact_pair_count"),
            fold.get("retained_exact_pair_membership_sha256"),
            f"fold {expected_fold} exact pairs",
        )
        pdb_ids = validate_membership_ledger(
            fold.get("retained_pdb_ids"),
            fold.get("retained_unique_pdb_count"),
            fold.get("retained_pdb_membership_sha256"),
            f"fold {expected_fold} PDBs",
        )
        if not set(system_ids).issubset(all_ids):
            raise AuditError("retained fold contains an unknown system")
        for system_id in system_ids:
            if system_id in fold_by_system:
                raise AuditError("system retained in multiple folds")
            fold_by_system[system_id] = expected_fold
        observed_pairs = {source_by_id[value].exact_pair_id for value in system_ids}
        observed_pdbs = {source_by_id[value].pdb_id for value in system_ids}
        if observed_pairs != set(pair_ids) or observed_pdbs != set(pdb_ids):
            raise AuditError("retained pair/PDB ledger does not reconstruct from source")
        retained_pair_sets.append(set(pair_ids))
        retained_pdb_sets.append(set(pdb_ids))

    discarded = assignment.get("discarded_ledger")
    if not isinstance(discarded, Mapping):
        raise AuditError("discarded ledger missing")
    if discarded.get("discard_policy") != (
        "discard_complete_exact_pair_and_pdb_connected_component"
    ):
        raise AuditError("discard policy mismatch")
    discarded_ids = set(
        validate_membership_ledger(
            discarded.get("discarded_system_ids"),
            discarded.get("discarded_system_count"),
            discarded.get("discarded_system_membership_sha256"),
            "discarded systems",
        )
    )
    discarded_pairs = set(
        validate_membership_ledger(
            discarded.get("discarded_exact_pair_ids"),
            discarded.get("discarded_unique_exact_pair_count"),
            discarded.get("discarded_exact_pair_membership_sha256"),
            "discarded exact pairs",
        )
    )
    discarded_pdbs = set(
        validate_membership_ledger(
            discarded.get("discarded_pdb_ids"),
            discarded.get("discarded_unique_pdb_count"),
            discarded.get("discarded_pdb_membership_sha256"),
            "discarded PDBs",
        )
    )
    if not discarded_ids.issubset(all_ids):
        raise AuditError("discarded ledger contains an unknown system")
    if {source_by_id[value].exact_pair_id for value in discarded_ids} != discarded_pairs:
        raise AuditError("discarded exact-pair ledger does not reconstruct")
    if {source_by_id[value].pdb_id for value in discarded_ids} != discarded_pdbs:
        raise AuditError("discarded PDB ledger does not reconstruct")
    if set(fold_by_system) & discarded_ids:
        raise AuditError("retained and discarded systems overlap")
    if set(fold_by_system) | discarded_ids != all_ids:
        raise AuditError("retained/discarded systems are not exhaustive")

    buckets: dict[str, str] = {
        **{system_id: f"fold:{fold}" for system_id, fold in fold_by_system.items()},
        **{system_id: "discarded" for system_id in discarded_ids},
    }
    pair_buckets: defaultdict[str, set[str]] = defaultdict(set)
    pdb_buckets: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        pair_buckets[row.exact_pair_id].add(buckets[row.system_id])
        pdb_buckets[row.pdb_id].add(buckets[row.system_id])
    if any(len(values) != 1 for values in pair_buckets.values()):
        raise AuditError("an exact pair is split across retained/discarded buckets")
    if any(len(values) != 1 for values in pdb_buckets.values()):
        raise AuditError("a PDB is split across retained/discarded buckets")
    if any(
        retained_pair_sets[left] & retained_pair_sets[right]
        or retained_pdb_sets[left] & retained_pdb_sets[right]
        for left, right in FOLD_PAIRS
    ):
        raise AuditError("retained folds share an exact pair or PDB")

    return fold_by_system, discarded_ids, {
        "all_source_systems_accounted_once": True,
        "exact_pairs_indivisible_across_retained_and_discarded_buckets": True,
        "pdbs_indivisible_across_retained_and_discarded_buckets": True,
        "retained_system_count": len(fold_by_system),
        "discarded_system_count": len(discarded_ids),
    }


def build_fold_inventories(
    rows: Sequence[SourceRow], fold_by_system: Mapping[str, int]
) -> dict[str, tuple[tuple[tuple[str, str], ...], ...]]:
    inventory: dict[str, list[dict[str, str]]] = {
        view: [dict(), dict(), dict()] for view in REQUIRED_VIEWS
    }
    for row in rows:
        fold = fold_by_system.get(row.system_id)
        if fold is None:
            continue
        values = {
            "vh": row.vh,
            "vl": row.vl,
            "concatenated_antibody": row.concatenated_antibody,
            "antigen": row.antigen,
        }
        for view, sequence in values.items():
            if not sequence:
                continue
            inventory[view][fold][sequence_sha256(sequence)] = sequence
    result: dict[str, tuple[tuple[tuple[str, str], ...], ...]] = {}
    for view, fold_maps in inventory.items():
        folds = tuple(tuple(sorted(fold_map.items())) for fold_map in fold_maps)
        if any({digest for digest, _ in folds[left]} & {digest for digest, _ in folds[right]} for left, right in FOLD_PAIRS):
            raise AuditError(f"exact sequence overlap across folds in {view}")
        result[view] = folds
    return result


def metrics_from_alignment(left: str, right: str, alignment: Any) -> AlignmentMetrics:
    left_blocks, right_blocks = alignment.aligned
    paired = 0
    identical = 0
    for left_block, right_block in zip(left_blocks, right_blocks):
        left_start, left_end = map(int, left_block)
        right_start, right_end = map(int, right_block)
        width = left_end - left_start
        if width <= 0 or width != right_end - right_start:
            raise AuditError("invalid global-alignment residue blocks")
        paired += width
        identical += sum(
            a == b
            for a, b in zip(
                left[left_start:left_end], right[right_start:right_end]
            )
        )
    if paired <= 0:
        raise AuditError("global alignment contains no residue-residue column")
    left_coverage = paired / len(left)
    right_coverage = paired / len(right)
    score = float(alignment.score)
    if not math.isfinite(score):
        raise AuditError("alignment score is non-finite")
    return AlignmentMetrics(
        score=score,
        identical_residue_pair_count=identical,
        aligned_residue_pair_count=paired,
        alignment_column_count=len(left) + len(right) - paired,
        aligned_region_identity=identical / paired,
        left_reciprocal_coverage=left_coverage,
        right_reciprocal_coverage=right_coverage,
        minimum_reciprocal_coverage=min(left_coverage, right_coverage),
    )


def _blosum62_payload_sha256(matrix: Any) -> str:
    alphabet = str(matrix.alphabet)
    return canonical_payload_sha256(
        {
            "alphabet": alphabet,
            "scores": [
                [float(matrix[left, right]) for right in alphabet]
                for left in alphabet
            ],
        }
    )


def build_locked_aligner() -> object:
    try:
        import Bio
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as error:
        raise AuditError("Biopython is required") from error
    if Bio.__version__ != EXPECTED_BIOPYTHON_VERSION:
        raise AuditError(
            f"Biopython mismatch: expected {EXPECTED_BIOPYTHON_VERSION}, observed {Bio.__version__}"
        )
    matrix = substitution_matrices.load("BLOSUM62")
    if _blosum62_payload_sha256(matrix) != EXPECTED_BLOSUM62_PAYLOAD_SHA256:
        raise AuditError("BLOSUM62 payload mismatch")
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -GAP_OPEN_PENALTY
    aligner.extend_gap_score = -GAP_EXTEND_PENALTY
    return aligner


def is_safe_length_prune(left_length: int, right_length: int) -> bool:
    shorter = min(left_length, right_length)
    longer = max(left_length, right_length)
    return COVERAGE_DENOMINATOR * shorter < COVERAGE_NUMERATOR * longer


def meets_threshold(metrics: AlignmentMetrics, left_length: int, right_length: int) -> bool:
    paired = metrics.aligned_residue_pair_count
    identical = metrics.identical_residue_pair_count
    return (
        IDENTITY_DENOMINATOR * identical >= IDENTITY_NUMERATOR * paired
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * left_length
        and COVERAGE_DENOMINATOR * paired >= COVERAGE_NUMERATOR * right_length
    )


def _initialize_worker(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
    view: str,
    left_fold: int,
    right_fold: int,
) -> None:
    global _WORKER_LEFT, _WORKER_RIGHT, _WORKER_VIEW
    global _WORKER_LEFT_FOLD, _WORKER_RIGHT_FOLD, _WORKER_ALIGNER
    _WORKER_LEFT = left
    _WORKER_RIGHT = right
    _WORKER_VIEW = view
    _WORKER_LEFT_FOLD = left_fold
    _WORKER_RIGHT_FOLD = right_fold
    _WORKER_ALIGNER = build_locked_aligner()


def _audit_flat_range(bounds: tuple[int, int]) -> dict[str, Any]:
    if _WORKER_ALIGNER is None:
        raise AuditError("alignment worker not initialized")
    start, end = bounds
    right_count = len(_WORKER_RIGHT)
    aligned = 0
    pruned = 0
    crossings: list[dict[str, Any]] = []
    for flat_index in range(start, end):
        left_index, right_index = divmod(flat_index, right_count)
        left_digest, left_sequence = _WORKER_LEFT[left_index]
        right_digest, right_sequence = _WORKER_RIGHT[right_index]
        if is_safe_length_prune(len(left_sequence), len(right_sequence)):
            pruned += 1
            continue
        alignment = next(iter(_WORKER_ALIGNER.align(left_sequence, right_sequence)))
        metrics = metrics_from_alignment(left_sequence, right_sequence, alignment)
        aligned += 1
        if meets_threshold(metrics, len(left_sequence), len(right_sequence)):
            crossings.append(
                {
                    "format_version": EDGE_FORMAT_VERSION,
                    "sequence_view": _WORKER_VIEW,
                    "left_fold_index": _WORKER_LEFT_FOLD,
                    "right_fold_index": _WORKER_RIGHT_FOLD,
                    "left_sequence_sha256": left_digest,
                    "right_sequence_sha256": right_digest,
                    "left_length": len(left_sequence),
                    "right_length": len(right_sequence),
                    "alignment": asdict(metrics),
                    "locked_threshold_rule_met": True,
                    "raw_sequences_or_system_ids_included": False,
                }
            )
    return {
        "candidate_pair_count": end - start,
        "dynamic_programming_alignment_count": aligned,
        "provably_pruned_by_length_upper_bound_count": pruned,
        "crossing_rows": crossings,
    }


def _iter_partial_results(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
    view: str,
    left_fold: int,
    right_fold: int,
    workers: int,
    chunk_size: int,
) -> Iterator[dict[str, Any]]:
    total = len(left) * len(right)
    ranges = [
        (start, min(total, start + chunk_size))
        for start in range(0, total, chunk_size)
    ]
    initializer_args = (left, right, view, left_fold, right_fold)
    if workers == 1:
        _initialize_worker(*initializer_args)
        for bounds in ranges:
            yield _audit_flat_range(bounds)
        return
    context = multiprocessing.get_context("fork")
    with context.Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=initializer_args,
    ) as pool:
        yield from pool.imap(_audit_flat_range, ranges, chunksize=1)


def _write_jsonl(handle: BinaryIO, payload: Mapping[str, Any]) -> None:
    handle.write(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    )


def audit_view(
    view: str,
    folds: Sequence[Sequence[tuple[str, str]]],
    target_path: Path,
    workers: int,
    chunk_size: int,
) -> tuple[dict[str, Any], StagedArtifact]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    pair_summaries: list[dict[str, Any]] = []
    totals = {"candidate": 0, "aligned": 0, "pruned": 0, "crossing": 0}
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for left_fold, right_fold in FOLD_PAIRS:
                left = tuple(folds[left_fold])
                right = tuple(folds[right_fold])
                expected_candidates = len(left) * len(right)
                counts = {"candidate": 0, "aligned": 0, "pruned": 0, "crossing": 0}
                for partial in _iter_partial_results(
                    left,
                    right,
                    view,
                    left_fold,
                    right_fold,
                    workers,
                    chunk_size,
                ):
                    counts["candidate"] += partial["candidate_pair_count"]
                    counts["aligned"] += partial["dynamic_programming_alignment_count"]
                    counts["pruned"] += partial[
                        "provably_pruned_by_length_upper_bound_count"
                    ]
                    for row in partial["crossing_rows"]:
                        _write_jsonl(handle, row)
                        counts["crossing"] += 1
                if counts["candidate"] != expected_candidates:
                    raise AuditError("Cartesian-product enumeration incomplete")
                if counts["aligned"] + counts["pruned"] != expected_candidates:
                    raise AuditError("alignment/prune accounting incomplete")
                pair_summaries.append(
                    {
                        "left_fold_index": left_fold,
                        "right_fold_index": right_fold,
                        "left_unique_sequence_count": len(left),
                        "right_unique_sequence_count": len(right),
                        "cross_fold_unique_sequence_pair_count": counts["candidate"],
                        "dynamic_programming_alignment_count": counts["aligned"],
                        "provably_pruned_by_length_upper_bound_count": counts["pruned"],
                        "threshold_crossing_edge_count": counts["crossing"],
                    }
                )
                for key in totals:
                    totals[key] += counts[key]
                print(
                    f"completed view={view} folds={left_fold}-{right_fold} "
                    f"pairs={counts['candidate']} crossings={counts['crossing']}",
                    flush=True,
                )
            handle.flush()
            os.fsync(handle.fileno())
        artifact = StagedArtifact(
            temporary_path=temporary,
            target_path=target_path,
            sha256=sha256_file(temporary),
            size_bytes=temporary.stat().st_size,
            edge_count=totals["crossing"],
        )
        return {
            "sequence_view": view,
            "valid_zero_crossing_edges": totals["crossing"] == 0,
            "fold_pairs": pair_summaries,
            "total_cross_fold_unique_sequence_pair_count": totals["candidate"],
            "total_dynamic_programming_alignment_count": totals["aligned"],
            "total_provably_pruned_by_length_upper_bound_count": totals["pruned"],
            "total_threshold_crossing_edge_count": totals["crossing"],
        }, artifact
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stage_bytes(target_path: Path, payload: bytes) -> StagedArtifact:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".staged", dir=target_path.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return StagedArtifact(
        temporary_path=temporary,
        target_path=target_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        edge_count=0,
    )


def publish_no_clobber(artifact: StagedArtifact) -> None:
    if sha256_file(artifact.temporary_path) != artifact.sha256:
        raise AuditError("staged artifact changed before publication")
    try:
        os.link(artifact.temporary_path, artifact.target_path)
    except OSError as error:
        if error.errno == errno.EEXIST:
            raise FileExistsError(artifact.target_path) from error
        raise
    directory_descriptor = os.open(
        artifact.target_path.parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def runtime_provenance(workers: int, chunk_size: int) -> dict[str, Any]:
    import Bio
    from Bio.Align import _pairwisealigner

    backend = Path(_pairwisealigner.__file__).resolve(strict=True)
    return {
        "python": platform.python_version(),
        "biopython": Bio.__version__,
        "platform": platform.platform(),
        "worker_count": workers,
        "chunk_size": chunk_size,
        "multiprocessing_start_method": "fork" if workers > 1 else "single_process",
        "pairwise_aligner_binary": {
            "path": str(backend),
            "sha256": sha256_file(backend),
            "size_bytes": backend.stat().st_size,
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_targets = {
        view: output_dir / f"{view}.identity80_coverage80.crossing_edges.jsonl"
        for view in REQUIRED_VIEWS
    }
    result_target = output_dir / "PINDER_ANTIBODY_S2_EXACT_FOUR_VIEW_AUDIT_RESULT_V1.json"
    receipt_target = output_dir / "PINDER_ANTIBODY_S2_EXACT_FOUR_VIEW_AUDIT_RECEIPT_V1.json"
    targets = [*edge_targets.values(), result_target, receipt_target]
    if any(path.exists() for path in targets):
        raise FileExistsError("create-only audit output already exists")

    inputs = {
        "assignment": verify_file(
            arguments.assignment, arguments.assignment_sha256, "provisional assignment"
        ),
        "canonical_manifest": verify_file(
            arguments.canonical_manifest,
            arguments.canonical_manifest_sha256,
            "canonical manifest",
        ),
        "auditor_source": verify_file(
            Path(__file__), arguments.auditor_source_sha256, "auditor source"
        ),
    }
    assignment = strict_json(Path(inputs["assignment"]["path"]))
    rows = load_source_rows(Path(inputs["canonical_manifest"]["path"]))
    fold_by_system, discarded_ids, ledger_checks = validate_assignment(
        assignment, rows, arguments.assignment_scientific_sha256
    )
    inventories = build_fold_inventories(rows, fold_by_system)
    build_locked_aligner()
    runtime = runtime_provenance(arguments.workers, arguments.chunk_size)

    staged_edges: list[StagedArtifact] = []
    staged_result: StagedArtifact | None = None
    staged_receipt: StagedArtifact | None = None
    try:
        results: dict[str, dict[str, Any]] = {}
        for view in REQUIRED_VIEWS:
            result, artifact = audit_view(
                view,
                inventories[view],
                edge_targets[view],
                arguments.workers,
                arguments.chunk_size,
            )
            results[view] = result
            staged_edges.append(artifact)
        approved = all(
            results[view]["valid_zero_crossing_edges"] is True
            and results[view]["total_threshold_crossing_edge_count"] == 0
            for view in REQUIRED_VIEWS
        )
        total_candidates = sum(
            result["total_cross_fold_unique_sequence_pair_count"]
            for result in results.values()
        )
        total_aligned = sum(
            result["total_dynamic_programming_alignment_count"]
            for result in results.values()
        )
        total_pruned = sum(
            result["total_provably_pruned_by_length_upper_bound_count"]
            for result in results.values()
        )
        total_crossings = sum(
            result["total_threshold_crossing_edge_count"]
            for result in results.values()
        )
        if total_aligned + total_pruned != total_candidates:
            raise AuditError("four-view exhaustive accounting mismatch")
        for label, reference in inputs.items():
            if sha256_file(Path(reference["path"])) != reference["sha256"]:
                raise AuditError(f"{label} changed during audit")

        result_payload: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "status": (
                "PASS_EXACT_FOUR_VIEW_S2_80_ZERO_CROSSING_AUDIT"
                if approved
                else "FAIL_EXACT_FOUR_VIEW_S2_80_CROSSINGS_DETECTED"
            ),
            "approval_status": (
                "APPROVED_TRAIN_ONLY_S2_FOLDS"
                if approved
                else "NOT_APPROVED_REBUILD_SPLIT_REQUIRED"
            ),
            "authorized_for_training": approved,
            "authorized_for_model_selection": approved,
            "inputs": inputs,
            "assignment_scientific_payload_sha256": arguments.assignment_scientific_sha256,
            "ledger_checks": ledger_checks,
            "policy": {
                "algorithm": "exact global dynamic programming (Needleman-Wunsch family)",
                "backend": f"Biopython PairwiseAligner {EXPECTED_BIOPYTHON_VERSION}",
                "substitution_matrix": "BLOSUM62",
                "substitution_matrix_payload_sha256": EXPECTED_BLOSUM62_PAYLOAD_SHA256,
                "gap_open_penalty": GAP_OPEN_PENALTY,
                "gap_extend_penalty": GAP_EXTEND_PENALTY,
                "aligned_region_identity_minimum": 0.8,
                "reciprocal_coverage_minimum": 0.8,
                "threshold_equality_included": True,
                "required_sequence_views": list(REQUIRED_VIEWS),
                "missing_vl_policy": "exclude genuinely empty VL from VL inventory",
                "only_pruning_rule": "raw length ratio proves reciprocal coverage below 0.8",
            },
            "inventory": {
                view: {
                    "unique_sequence_count_by_fold": [
                        len(fold) for fold in inventories[view]
                    ],
                    "exact_cross_fold_overlap_count": 0,
                }
                for view in REQUIRED_VIEWS
            },
            "results": results,
            "exhaustive_accounting": {
                "total_cross_fold_unique_sequence_pair_count": total_candidates,
                "total_dynamic_programming_alignment_count": total_aligned,
                "total_provably_pruned_by_length_upper_bound_count": total_pruned,
                "total_threshold_crossing_edge_count": total_crossings,
                "alignments_plus_safe_prunes_equal_all_candidates": True,
            },
            "crossing_artifacts": {
                view: {
                    "path": str(edge_targets[view]),
                    "sha256": staged_edges[index].sha256,
                    "size_bytes": staged_edges[index].size_bytes,
                    "edge_count": staged_edges[index].edge_count,
                    "raw_sequences_or_system_ids_included": False,
                }
                for index, view in enumerate(REQUIRED_VIEWS)
            },
            "runtime": runtime,
            "data_firewall": {
                "source_pool": "PINDER_train_only",
                "validation_rows_read": False,
                "test_rows_read": False,
                "task_labels_read": False,
                "structures_read": False,
                "predictions_or_scores_read": False,
                "discarded_sequences_excluded_from_cross_fold_audit": len(discarded_ids),
            },
            "claim_boundary": (
                "This exact audit only authorizes train-only internal S2 folds for "
                "training and model selection. It is not a performance evaluation, "
                "temporal test, structural split, or PLM pretraining leakage audit."
            ),
        }
        result_payload["scientific_payload_sha256"] = canonical_payload_sha256(
            result_payload
        )
        staged_result = stage_bytes(result_target, canonical_json_bytes(result_payload))
        receipt_payload = {
            "format_version": "pinder_antibody_s2_exact_four_view_nw_audit_receipt_v1",
            "result": {
                "path": str(result_target),
                "sha256": staged_result.sha256,
                "size_bytes": staged_result.size_bytes,
                "scientific_payload_sha256": result_payload[
                    "scientific_payload_sha256"
                ],
            },
            "crossing_artifacts": result_payload["crossing_artifacts"],
            "all_outputs_create_only": True,
            "all_inputs_rehashed_before_publication": True,
        }
        staged_receipt = stage_bytes(
            receipt_target, canonical_json_bytes(receipt_payload)
        )
        for artifact in [*staged_edges, staged_result, staged_receipt]:
            publish_no_clobber(artifact)
        print(
            json.dumps(
                {
                    "status": result_payload["status"],
                    "authorized_for_training": approved,
                    "total_threshold_crossing_edge_count": total_crossings,
                    "result_sha256": staged_result.sha256,
                    "receipt_sha256": staged_receipt.sha256,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return result_payload
    finally:
        for artifact in [*staged_edges, staged_result, staged_receipt]:
            if artifact is not None:
                artifact.temporary_path.unlink(missing_ok=True)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", required=True)
    parser.add_argument("--assignment-scientific-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--auditor-source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=256)
    arguments = parser.parse_args(argv)
    if arguments.workers < 1 or arguments.chunk_size < 1:
        parser.error("workers and chunk size must be positive")
    for name in (
        "assignment_sha256",
        "assignment_scientific_sha256",
        "canonical_manifest_sha256",
        "auditor_source_sha256",
    ):
        value = getattr(arguments, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name} is not a lowercase SHA-256")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
