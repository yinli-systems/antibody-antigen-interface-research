#!/usr/bin/env python3
"""Run an exhaustive Needleman-Wunsch leakage audit on development splits.

The program reads sequence columns and split membership only. It establishes a
development-data barrier from a separate source manifest, refuses any audited
split that does not exactly cover that development pool, and never aligns rows
from sealed splits. Labels, contacts, predictions, and checkpoints are outside
the accepted input schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import multiprocessing
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from build_identity_isolated_development_split import (
    SequenceRecord,
    _eligible_ids,
    atomic_write_json,
    load_development_records,
    load_split_lists,
    normalize_sequence,
    sequence_digest,
    sha256_file,
)


FORMAT_VERSION = "sabdab2_development_needleman_wunsch_audit_v1"
DEFAULT_IDENTITY_THRESHOLD = 0.80
DEFAULT_COVERAGE_THRESHOLD = 0.80
DEFAULT_CALM_SCALED_THRESHOLD = 0.80
DEFAULT_GAP_OPEN = 10.0
DEFAULT_GAP_EXTEND = 1.0


@dataclass(frozen=True)
class AlignmentMetrics:
    """Statistics from one declared optimal global alignment."""

    score: float
    identical_residue_pair_count: int
    aligned_residue_pair_count: int
    alignment_column_count: int
    aligned_region_identity: float
    left_reciprocal_coverage: float
    right_reciprocal_coverage: float
    minimum_reciprocal_coverage: float
    calm_scaled_similarity: float
    full_alignment_column_identity: float


@dataclass(frozen=True)
class PairResult:
    """Digest-only result for one dynamically aligned cross-split pair."""

    train_digest: str
    validation_digest: str
    train_length: int
    validation_length: int
    metrics: AlignmentMetrics
    violates_identity_and_coverage_rule: bool
    violates_calm_scaled_rule: bool


@dataclass(frozen=True)
class AntibodyChainRecord:
    """Separate development-only variable heavy and light chain sequences."""

    instance_id: str
    heavy: str
    light: str


@dataclass(frozen=True)
class AuditPolicy:
    """Thresholds applied to every aligned sequence pair."""

    identity_threshold: float
    coverage_threshold: float
    calm_scaled_threshold: float
    gap_open: float
    gap_extend: float
    top_pair_count: int

    @property
    def safe_length_ratio_cutoff(self) -> float:
        """Return the weakest length bound needed to exclude both rules."""
        return min(self.coverage_threshold, self.calm_scaled_threshold)


_WORKER_TRAIN: tuple[tuple[str, str], ...] = ()
_WORKER_VALIDATION: tuple[tuple[str, str], ...] = ()
_WORKER_POLICY: AuditPolicy | None = None
_WORKER_ALIGNER: object | None = None


def _validate_probability(value: float, label: str) -> float:
    """Validate a closed-unit-interval threshold."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return value


def _canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible object with deterministic serialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _inventory_sha256(items: Sequence[tuple[str, str]]) -> str:
    """Hash an ordered digest-and-length inventory without exposing sequences."""
    rows = [f"{digest}\t{len(sequence)}" for digest, sequence in items]
    return hashlib.sha256(("\n".join(rows) + "\n").encode("ascii")).hexdigest()


def _unique_sequences(
    records: Mapping[str, SequenceRecord], identifiers: Iterable[str], role: str
) -> tuple[tuple[str, str], ...]:
    """Return unique development sequences ordered by their SHA-256 digests."""
    if role not in {"antibody", "antigen"}:
        raise ValueError(f"Unknown role: {role}")
    sequences = {
        records[identifier].antibody if role == "antibody" else records[identifier].antigen
        for identifier in identifiers
    }
    return tuple(sorted((sequence_digest(sequence), sequence) for sequence in sequences))


def load_development_chain_records(
    manifest_path: Path,
    eligible_ids: set[str],
    id_column: str,
    heavy_column: str,
    light_column: str,
) -> dict[str, AntibodyChainRecord]:
    """Read separate antibody chains for eligible development rows only."""
    records: dict[str, AntibodyChainRecord] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, heavy_column, light_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Sequence manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            identifier = str(row[id_column])
            if identifier not in eligible_ids:
                continue
            if identifier in records:
                raise ValueError(
                    f"Duplicate eligible instance {identifier!r} at row {row_number}"
                )
            heavy = normalize_sequence(row[heavy_column], f"{identifier}:{heavy_column}")
            light = normalize_sequence(
                row[light_column], f"{identifier}:{light_column}", allow_empty=True
            )
            records[identifier] = AntibodyChainRecord(identifier, heavy, light)
    missing_ids = eligible_ids - set(records)
    if missing_ids:
        raise ValueError(
            f"Sequence manifest is missing eligible IDs: {sorted(missing_ids)[:10]}"
        )
    return records


def _unique_chain_sequences(
    records: Mapping[str, AntibodyChainRecord],
    identifiers: Iterable[str],
    chain: str,
) -> tuple[tuple[str, str], ...]:
    """Return non-empty unique VH or VL sequences ordered by digest."""
    if chain not in {"vh", "vl"}:
        raise ValueError(f"Unknown antibody chain: {chain}")
    sequences = {
        records[identifier].heavy if chain == "vh" else records[identifier].light
        for identifier in identifiers
    }
    sequences.discard("")
    return tuple(sorted((sequence_digest(sequence), sequence) for sequence in sequences))


def _chain_presence_summary(
    records: Mapping[str, AntibodyChainRecord], identifiers: Sequence[str]
) -> dict[str, int]:
    """Count development instances with each antibody chain present."""
    return {
        "instance_count": len(identifiers),
        "instance_count_with_vh": sum(bool(records[item].heavy) for item in identifiers),
        "instance_count_with_vl": sum(bool(records[item].light) for item in identifiers),
        "instance_count_without_vl": sum(not records[item].light for item in identifiers),
    }


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write strict JSON Lines through an adjacent temporary file."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n"
            )
    os.replace(temporary, path)


def metrics_from_aligned_blocks(
    left: str,
    right: str,
    left_blocks: Sequence[Sequence[int]],
    right_blocks: Sequence[Sequence[int]],
    score: float,
) -> AlignmentMetrics:
    """Compute metrics from gap-free paired blocks of a global alignment.

    The aligned-region identity denominator is the number of alignment columns
    containing one residue from each sequence; all columns containing a gap are
    excluded from this denominator. Reciprocal coverage divides that same
    paired-column count by each full unaligned sequence length. The CALM-style
    scaled score is aligned-region identity times the smaller reciprocal
    coverage, which is algebraically equal to identical residues divided by the
    longer full sequence length.
    """
    if not left or not right:
        raise ValueError("Global alignment inputs must be non-empty")
    if len(left_blocks) != len(right_blocks):
        raise ValueError("Aligned block lists have different lengths")
    paired = 0
    identical = 0
    for left_block, right_block in zip(left_blocks, right_blocks):
        if len(left_block) != 2 or len(right_block) != 2:
            raise ValueError("Every aligned block must contain start and end coordinates")
        left_start, left_end = (int(value) for value in left_block)
        right_start, right_end = (int(value) for value in right_block)
        left_width = left_end - left_start
        right_width = right_end - right_start
        if left_width <= 0 or left_width != right_width:
            raise ValueError("Aligned residue blocks must have equal positive widths")
        paired += left_width
        identical += sum(
            left_residue == right_residue
            for left_residue, right_residue in zip(
                left[left_start:left_end], right[right_start:right_end]
            )
        )
    if paired <= 0:
        raise ValueError("The selected global alignment contains no residue-residue column")
    alignment_columns = len(left) + len(right) - paired
    identity = identical / paired
    left_coverage = paired / len(left)
    right_coverage = paired / len(right)
    minimum_coverage = min(left_coverage, right_coverage)
    scaled = identity * minimum_coverage
    algebraic_scaled = identical / max(len(left), len(right))
    if abs(scaled - algebraic_scaled) > 1e-12:
        raise AssertionError("CALM-style scaled-score identity failed")
    return AlignmentMetrics(
        score=float(score),
        identical_residue_pair_count=identical,
        aligned_residue_pair_count=paired,
        alignment_column_count=alignment_columns,
        aligned_region_identity=identity,
        left_reciprocal_coverage=left_coverage,
        right_reciprocal_coverage=right_coverage,
        minimum_reciprocal_coverage=minimum_coverage,
        calm_scaled_similarity=scaled,
        full_alignment_column_identity=identical / alignment_columns,
    )


def build_biopython_aligner(gap_open: float, gap_extend: float) -> object:
    """Construct the declared exact dynamic-programming alignment backend."""
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required for the production Needleman-Wunsch audit"
        ) from exc
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -abs(float(gap_open))
    aligner.extend_gap_score = -abs(float(gap_extend))
    if aligner.mode != "global":
        raise AssertionError("PairwiseAligner did not retain global mode")
    return aligner


def biopython_global_metrics(left: str, right: str, aligner: object) -> AlignmentMetrics:
    """Return metrics for the first deterministic optimal global traceback."""
    alignments = aligner.align(left, right)
    alignment = next(iter(alignments))
    aligned = alignment.aligned
    return metrics_from_aligned_blocks(
        left,
        right,
        aligned[0],
        aligned[1],
        float(alignment.score),
    )


def classify_pair(
    train_digest: str,
    train_sequence: str,
    validation_digest: str,
    validation_sequence: str,
    metrics: AlignmentMetrics,
    policy: AuditPolicy,
) -> PairResult:
    """Apply both declared leakage rules to one aligned pair."""
    tolerance = 1e-12
    primary = (
        metrics.aligned_region_identity + tolerance >= policy.identity_threshold
        and metrics.left_reciprocal_coverage + tolerance >= policy.coverage_threshold
        and metrics.right_reciprocal_coverage + tolerance >= policy.coverage_threshold
    )
    calm = metrics.calm_scaled_similarity + tolerance >= policy.calm_scaled_threshold
    return PairResult(
        train_digest=train_digest,
        validation_digest=validation_digest,
        train_length=len(train_sequence),
        validation_length=len(validation_sequence),
        metrics=metrics,
        violates_identity_and_coverage_rule=primary,
        violates_calm_scaled_rule=calm,
    )


def pair_length_upper_bound(left: str, right: str) -> float:
    """Return the maximum attainable minimum coverage or scaled similarity."""
    return min(len(left), len(right)) / max(len(left), len(right))


def _pair_rank(result: PairResult) -> tuple[object, ...]:
    """Return a stable ranking key for retaining the most similar pairs."""
    metrics = result.metrics
    return (
        metrics.calm_scaled_similarity,
        metrics.aligned_region_identity,
        metrics.minimum_reciprocal_coverage,
        metrics.identical_residue_pair_count,
        result.train_digest,
        result.validation_digest,
    )


def _pair_payload(result: PairResult) -> dict[str, object]:
    """Serialize a pair without raw sequences or instance identifiers."""
    payload = asdict(result)
    payload["metrics"] = asdict(result.metrics)
    return payload


def _worker_initialize(
    train: tuple[tuple[str, str], ...],
    validation: tuple[tuple[str, str], ...],
    policy: AuditPolicy,
) -> None:
    """Initialize immutable process-local alignment state."""
    global _WORKER_ALIGNER, _WORKER_POLICY, _WORKER_TRAIN, _WORKER_VALIDATION
    _WORKER_TRAIN = train
    _WORKER_VALIDATION = validation
    _WORKER_POLICY = policy
    _WORKER_ALIGNER = build_biopython_aligner(policy.gap_open, policy.gap_extend)


def _audit_flat_range(bounds: tuple[int, int]) -> dict[str, object]:
    """Audit one deterministic flat-index range of a Cartesian product."""
    if _WORKER_POLICY is None or _WORKER_ALIGNER is None:
        raise RuntimeError("Alignment worker was not initialized")
    start, end = bounds
    validation_count = len(_WORKER_VALIDATION)
    aligned_count = 0
    upper_bound_rejected_count = 0
    primary_count = 0
    calm_count = 0
    top_heap: list[tuple[tuple[object, ...], PairResult]] = []
    primary_examples: list[PairResult] = []
    calm_examples: list[PairResult] = []
    for flat_index in range(start, end):
        train_index, validation_index = divmod(flat_index, validation_count)
        train_digest, train_sequence = _WORKER_TRAIN[train_index]
        validation_digest, validation_sequence = _WORKER_VALIDATION[validation_index]
        if (
            pair_length_upper_bound(train_sequence, validation_sequence) + 1e-12
            < _WORKER_POLICY.safe_length_ratio_cutoff
        ):
            upper_bound_rejected_count += 1
            continue
        metrics = biopython_global_metrics(
            train_sequence, validation_sequence, _WORKER_ALIGNER
        )
        result = classify_pair(
            train_digest,
            train_sequence,
            validation_digest,
            validation_sequence,
            metrics,
            _WORKER_POLICY,
        )
        aligned_count += 1
        rank = _pair_rank(result)
        if len(top_heap) < _WORKER_POLICY.top_pair_count:
            heapq.heappush(top_heap, (rank, result))
        elif rank > top_heap[0][0]:
            heapq.heapreplace(top_heap, (rank, result))
        if result.violates_identity_and_coverage_rule:
            primary_count += 1
            primary_examples.append(result)
        if result.violates_calm_scaled_rule:
            calm_count += 1
            calm_examples.append(result)
    primary_examples.sort(key=_pair_rank, reverse=True)
    calm_examples.sort(key=_pair_rank, reverse=True)
    return {
        "aligned_count": aligned_count,
        "upper_bound_rejected_count": upper_bound_rejected_count,
        "primary_count": primary_count,
        "calm_count": calm_count,
        "top": [item[1] for item in top_heap],
        "primary_edges": primary_examples,
        "primary_examples": primary_examples[: _WORKER_POLICY.top_pair_count],
        "calm_examples": calm_examples[: _WORKER_POLICY.top_pair_count],
    }


def _retain_top(
    results: Iterable[PairResult], count: int
) -> list[PairResult]:
    """Return the highest ranked unique digest pairs."""
    unique: dict[tuple[str, str], PairResult] = {}
    for result in results:
        key = (result.train_digest, result.validation_digest)
        existing = unique.get(key)
        if existing is None or _pair_rank(result) > _pair_rank(existing):
            unique[key] = result
    return sorted(unique.values(), key=_pair_rank, reverse=True)[:count]


def audit_role(
    train: tuple[tuple[str, str], ...],
    validation: tuple[tuple[str, str], ...],
    policy: AuditPolicy,
    workers: int,
    chunk_size: int,
) -> dict[str, object]:
    """Exhaustively audit all unique train-by-validation sequence pairs."""
    if not train or not validation:
        raise ValueError("Both split sides must contain at least one unique sequence")
    total = len(train) * len(validation)
    ranges = [
        (start, min(total, start + chunk_size))
        for start in range(0, total, chunk_size)
    ]
    if workers == 1:
        _worker_initialize(train, validation, policy)
        partials = [_audit_flat_range(bounds) for bounds in ranges]
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(
            processes=workers,
            initializer=_worker_initialize,
            initargs=(train, validation, policy),
        ) as pool:
            partials = list(pool.imap(_audit_flat_range, ranges, chunksize=1))
    aligned_count = sum(int(partial["aligned_count"]) for partial in partials)
    rejected_count = sum(
        int(partial["upper_bound_rejected_count"]) for partial in partials
    )
    if aligned_count + rejected_count != total:
        raise AssertionError("The exhaustive Cartesian-product accounting is incomplete")
    primary_count = sum(int(partial["primary_count"]) for partial in partials)
    calm_count = sum(int(partial["calm_count"]) for partial in partials)
    top = _retain_top(
        (result for partial in partials for result in partial["top"]),
        policy.top_pair_count,
    )
    primary_examples = _retain_top(
        (result for partial in partials for result in partial["primary_examples"]),
        policy.top_pair_count,
    )
    primary_edges = _retain_top(
        (result for partial in partials for result in partial["primary_edges"]),
        primary_count,
    )
    if len(primary_edges) != primary_count:
        raise AssertionError("Primary crossing-edge aggregation lost or duplicated pairs")
    calm_examples = _retain_top(
        (result for partial in partials for result in partial["calm_examples"]),
        policy.top_pair_count,
    )
    train_digests = {digest for digest, _ in train}
    validation_digests = {digest for digest, _ in validation}
    exact_overlap_count = len(train_digests & validation_digests)
    valid_primary = primary_count == 0 and exact_overlap_count == 0
    valid_calm = calm_count == 0 and exact_overlap_count == 0
    return {
        "_primary_crossing_edges": primary_edges,
        "valid": valid_primary and valid_calm,
        "valid_identity_and_reciprocal_coverage_rule": valid_primary,
        "valid_calm_scaled_rule": valid_calm,
        "train_unique_sequence_count": len(train),
        "validation_unique_sequence_count": len(validation),
        "train_inventory_sha256": _inventory_sha256(train),
        "validation_inventory_sha256": _inventory_sha256(validation),
        "exact_sequence_cross_split_overlap_count": exact_overlap_count,
        "total_cross_split_unique_sequence_pair_count": total,
        "dynamic_programming_alignment_count": aligned_count,
        "provably_rejected_by_length_upper_bound_count": rejected_count,
        "identity_and_reciprocal_coverage_violation_count": primary_count,
        "calm_scaled_similarity_violation_count": calm_count,
        "top_pairs_by_calm_scaled_similarity": [_pair_payload(result) for result in top],
        "identity_and_reciprocal_coverage_violation_examples": [
            _pair_payload(result) for result in primary_examples
        ],
        "calm_scaled_similarity_violation_examples": [
            _pair_payload(result) for result in calm_examples
        ],
    }


def write_primary_edge_artifact(
    role_result: dict[str, object],
    output_dir: Path,
    split_name: str,
    sequence_role: str,
) -> dict[str, object]:
    """Persist every primary-rule crossing edge and attach its provenance."""
    raw_edges = role_result.pop("_primary_crossing_edges")
    if not isinstance(raw_edges, list):
        raise TypeError("Internal primary crossing edges are not a list")
    edge_path = output_dir / f"{split_name}.{sequence_role}.identity_coverage_edges.jsonl"
    rows = [_pair_payload(edge) for edge in raw_edges]
    atomic_write_jsonl(edge_path, rows)
    artifact = {
        "format": "JSON Lines; one digest-only cross-split edge per line",
        "path": str(edge_path.expanduser().resolve()),
        "sha256": sha256_file(edge_path),
        "edge_count": len(rows),
        "raw_sequences_or_instance_ids_included": False,
        "selection_rule": "identity threshold and both reciprocal coverage thresholds",
    }
    if artifact["edge_count"] != role_result[
        "identity_and_reciprocal_coverage_violation_count"
    ]:
        raise AssertionError("Crossing-edge artifact count differs from the audit count")
    role_result["identity_and_reciprocal_coverage_edge_artifact"] = artifact
    return artifact


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable NAME=PATH arguments with stable unique names."""
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected NAME=PATH, received: {value!r}")
        if name in result:
            raise ValueError(f"Duplicate audit split name: {name}")
        if not all(character.isalnum() or character in "_-" for character in name):
            raise ValueError(f"Unsafe audit split name: {name!r}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def establish_development_barrier(
    source_split_path: Path,
    pool_splits: Sequence[str],
    sealed_splits: Sequence[str],
) -> tuple[set[str], set[str], dict[str, int]]:
    """Resolve disjoint development and sealed identifiers from one authority."""
    if not sealed_splits:
        raise ValueError("At least one sealed split is required for the safety barrier")
    source_splits = load_split_lists(source_split_path)
    development_ids, sealed_counts = _eligible_ids(
        source_splits, pool_splits, sealed_splits
    )
    sealed_ids = {
        identifier for split_name in sealed_splits for identifier in source_splits[split_name]
    }
    if not sealed_ids:
        raise ValueError("The declared sealed pool is empty")
    if development_ids & sealed_ids:
        raise AssertionError("Development and sealed identifiers overlap")
    return development_ids, sealed_ids, sealed_counts


def validate_audited_split(
    path: Path,
    development_ids: set[str],
    sealed_ids: set[str],
    train_split: str,
    validation_split: str,
) -> tuple[list[str], list[str]]:
    """Require exact development coverage and reject every sealed identifier."""
    split_lists = load_split_lists(path)
    missing_names = {train_split, validation_split} - set(split_lists)
    if missing_names:
        raise ValueError(f"Audited split is missing names: {sorted(missing_names)}")
    train = split_lists[train_split]
    validation = split_lists[validation_split]
    audited_ids = set(train) | set(validation)
    sealed_overlap = audited_ids & sealed_ids
    if sealed_overlap:
        raise ValueError(
            f"Audited train/validation includes {len(sealed_overlap)} sealed identifiers"
        )
    if audited_ids != development_ids:
        missing = len(development_ids - audited_ids)
        unexpected = len(audited_ids - development_ids)
        raise ValueError(
            "Audited train/validation must exactly cover the authoritative development "
            f"pool (missing={missing}, unexpected={unexpected})"
        )
    return train, validation


def _dependency_versions() -> dict[str, str]:
    """Return exact runtime versions needed to reproduce alignments."""
    try:
        import Bio
    except ImportError as exc:
        raise RuntimeError("Biopython is required for the audit") from exc
    return {
        "python": platform.python_version(),
        "biopython": Bio.__version__,
        "platform": platform.platform(),
    }


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    """Run all requested development-only split audits and write one report."""
    policy = AuditPolicy(
        identity_threshold=_validate_probability(
            args.identity_threshold, "identity threshold"
        ),
        coverage_threshold=_validate_probability(
            args.coverage_threshold, "coverage threshold"
        ),
        calm_scaled_threshold=_validate_probability(
            args.calm_scaled_threshold, "CALM scaled threshold"
        ),
        gap_open=abs(float(args.gap_open)),
        gap_extend=abs(float(args.gap_extend)),
        top_pair_count=int(args.top_pair_count),
    )
    if policy.gap_open <= 0.0 or policy.gap_extend <= 0.0:
        raise ValueError("Gap penalties must be positive magnitudes")
    if policy.top_pair_count < 1:
        raise ValueError("top-pair-count must be positive")
    if args.workers < 1 or args.chunk_size < 1:
        raise ValueError("workers and chunk-size must be positive")
    development_ids, sealed_ids, sealed_counts = establish_development_barrier(
        args.development_source_split,
        args.pool_splits,
        args.sealed_splits,
    )
    records = load_development_records(
        args.manifest,
        development_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    chain_records = load_development_chain_records(
        args.manifest,
        development_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
    )
    named_paths = parse_named_paths(args.audit_split)
    edge_output_dir = (
        args.edge_output_dir.expanduser().resolve()
        if args.edge_output_dir is not None
        else args.output.expanduser().resolve().parent / "needleman_wunsch_edges"
    )
    results: dict[str, object] = {}
    chain_sensitivity: dict[str, object] = {}
    split_provenance: dict[str, object] = {}
    for name, path in named_paths.items():
        train_ids, validation_ids = validate_audited_split(
            path,
            development_ids,
            sealed_ids,
            args.train_split,
            args.validation_split,
        )
        role_results: dict[str, object] = {}
        for role in ("antibody", "antigen"):
            train_sequences = _unique_sequences(records, train_ids, role)
            validation_sequences = _unique_sequences(records, validation_ids, role)
            role_result = audit_role(
                train_sequences,
                validation_sequences,
                policy,
                workers=args.workers,
                chunk_size=args.chunk_size,
            )
            write_primary_edge_artifact(
                role_result,
                edge_output_dir,
                name,
                "concat_antibody" if role == "antibody" else "antigen",
            )
            role_results[role] = role_result
        split_valid = all(
            bool(role_results[role]["valid"]) for role in ("antibody", "antigen")
        )
        results[name] = {
            "valid": split_valid,
            "train_instance_count": len(train_ids),
            "validation_instance_count": len(validation_ids),
            "roles": role_results,
        }
        per_chain: dict[str, object] = {}
        for chain in ("vh", "vl"):
            train_chains = _unique_chain_sequences(chain_records, train_ids, chain)
            validation_chains = _unique_chain_sequences(
                chain_records, validation_ids, chain
            )
            chain_result = audit_role(
                train_chains,
                validation_chains,
                policy,
                workers=args.workers,
                chunk_size=args.chunk_size,
            )
            write_primary_edge_artifact(
                chain_result,
                edge_output_dir,
                name,
                chain,
            )
            per_chain[chain] = chain_result
        chain_sensitivity[name] = {
            "passes_identity_and_reciprocal_coverage_rule": all(
                bool(per_chain[chain]["valid_identity_and_reciprocal_coverage_rule"])
                for chain in ("vh", "vl")
            ),
            "train_chain_presence": _chain_presence_summary(chain_records, train_ids),
            "validation_chain_presence": _chain_presence_summary(
                chain_records, validation_ids
            ),
            "chains": per_chain,
        }
        split_provenance[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    source_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stage": "exhaustive_development_needleman_wunsch_leakage_audit",
        "intended_use": "development_only",
        "valid": all(bool(result["valid"]) for result in results.values()),
        "frozen_test_labels_or_predictions_read": False,
        "frozen_test_sequences_aligned": False,
        "sealed_identifiers_read_for_exclusion_only": True,
        "development_instance_count": len(development_ids),
        "sealed_source_splits": list(args.sealed_splits),
        "sealed_source_split_counts": sealed_counts,
        "audited_split_names": list(named_paths),
        "policy": {
            "algorithm": "exact global dynamic programming (Needleman-Wunsch family)",
            "backend": "Biopython PairwiseAligner",
            "substitution_matrix": "BLOSUM62",
            "gap_model": "affine, with penalized terminal gaps",
            "gap_open_penalty": policy.gap_open,
            "gap_extend_penalty": policy.gap_extend,
            "optimal_traceback_policy": (
                "first deterministic traceback returned by the pinned Biopython runtime"
            ),
            "aligned_region_identity_definition": (
                "identical residue-residue columns / all residue-residue columns; "
                "gap columns are excluded from this denominator"
            ),
            "reciprocal_coverage_definition": (
                "paired residue-residue columns / full raw length of the corresponding "
                "sequence"
            ),
            "calm_scaled_similarity_definition": (
                "aligned-region identity * min(train coverage, validation coverage), "
                "equivalently identical residue-residue columns / max(full lengths)"
            ),
            "identity_and_coverage_rule": (
                f"identity >= {policy.identity_threshold} and both reciprocal coverages "
                f">= {policy.coverage_threshold}"
            ),
            "calm_scaled_rule": (
                f"CALM-style scaled similarity >= {policy.calm_scaled_threshold}"
            ),
            "enumeration": (
                "all unique train x validation sequence pairs; pairs below the declared "
                "length-ratio bound are rejected without alignment because neither rule "
                "can mathematically be reached"
            ),
            "length_ratio_upper_bound_cutoff": policy.safe_length_ratio_cutoff,
        },
        "provenance": {
            "sequence_manifest": {
                "path": str(args.manifest.expanduser().resolve()),
                "sha256": sha256_file(args.manifest),
            },
            "development_source_split": {
                "path": str(args.development_source_split.expanduser().resolve()),
                "sha256": sha256_file(args.development_source_split),
            },
            "audited_splits": split_provenance,
            "audit_source": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "runtime": _dependency_versions(),
            "worker_count": args.workers,
            "chunk_size": args.chunk_size,
        },
        "results": results,
        "supplementary_sensitivity_analyses": {
            "antibody_chain_aware": {
                "included_in_primary_validity": False,
                "rationale": (
                    "Separate VH-to-VH and VL-to-VL audits can detect a highly similar "
                    "chain whose signal is diluted by heavy-light concatenation."
                ),
                "missing_light_chain_policy": (
                    "Instances without VL remain in the development pool but contribute "
                    "only to the VH audit; no missing-chain placeholder is aligned."
                ),
                "splits": chain_sensitivity,
            }
        },
        "limitations": [
            (
                "CALM reports Parasail Needleman-Wunsch and identity times coverage, but "
                "does not publish its substitution matrix, gap penalties, exact coverage "
                "denominator, or optimal-alignment tie policy; this is therefore a "
                "transparent CALM-inspired secondary audit, not a bitwise reproduction."
            ),
            (
                "A global-alignment identity threshold is one operational definition of "
                "sequence novelty and does not establish structural, epitope, functional, "
                "temporal, or prospective independence."
            ),
            (
                "Both audited partitions are development splits. Passing this audit does "
                "not convert validation into an external test set and does not authorize "
                "opening any frozen test data."
            ),
            (
                "Primary antibody identity follows the active heavy-light concatenation "
                "key. Separate VH and VL results are reported as a stricter supplementary "
                "sensitivity analysis and are deliberately not folded into primary validity."
            ),
        ],
    }
    payload["scientific_payload_sha256"] = _canonical_sha256(payload)
    atomic_write_json(args.output, payload)
    receipt = {
        "format_version": FORMAT_VERSION,
        "report_path": str(args.output.expanduser().resolve()),
        "report_sha256": sha256_file(args.output),
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
        "audit_source_sha256": payload["provenance"]["audit_source"]["sha256"],
    }
    receipt_path = args.output.with_name(f"{args.output.name}.sha256.json")
    atomic_write_json(receipt_path, receipt)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "receipt": str(receipt_path.expanduser().resolve()),
        "valid": payload["valid"],
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
        "split_validity": {
            name: result["valid"] for name, result in results.items()
        },
    }, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-source-split", type=Path, required=True)
    parser.add_argument(
        "--audit-split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for each development split to audit.",
    )
    parser.add_argument("--pool-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--sealed-splits", nargs="+", default=["test"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--id-column", default="INSTANCE")
    parser.add_argument("--heavy-column", default="vh_seq")
    parser.add_argument("--light-column", default="vl_seq")
    parser.add_argument("--antigen-column", default="ag_seq")
    parser.add_argument("--identity-threshold", type=float, default=DEFAULT_IDENTITY_THRESHOLD)
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD)
    parser.add_argument(
        "--calm-scaled-threshold", type=float, default=DEFAULT_CALM_SCALED_THRESHOLD
    )
    parser.add_argument("--gap-open", type=float, default=DEFAULT_GAP_OPEN)
    parser.add_argument("--gap-extend", type=float, default=DEFAULT_GAP_EXTEND)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-pair-count", type=int, default=20)
    parser.add_argument(
        "--edge-output-dir",
        type=Path,
        help="Directory for complete digest-only primary crossing-edge artifacts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the command-line audit."""
    args = build_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
