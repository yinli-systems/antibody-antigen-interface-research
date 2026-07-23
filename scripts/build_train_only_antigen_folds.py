#!/usr/bin/env python3
"""Build deterministic train-only antigen-cluster folds for model selection.

Only identifiers in the authority ``train`` split are eligible. Authority
``validation`` and ``test`` memberships are treated as sealed: their IDs are
used solely to enforce exclusion, and their sequence or label columns are
never parsed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FOLD_FORMAT_VERSION = "sabdab2_antigen_cluster_development_fold_v1"
INDEX_FORMAT_VERSION = "sabdab2_antigen_cluster_development_index_v1"
EXPECTED_AUTHORITY_SHA256 = (
    "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
)
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class PairRecord:
    """One source-train antibody-antigen instance."""

    instance_id: str
    antibody_digest: str
    antigen_digest: str

    @property
    def exact_pair(self) -> tuple[str, str]:
        """Return the exact sequence-pair identity."""
        return self.antibody_digest, self.antigen_digest


@dataclass(frozen=True)
class SimilarityHit:
    """One threshold-qualified undirected sequence-similarity edge."""

    left: str
    right: str
    identity: float
    left_coverage: float
    right_coverage: float
    source: str


@dataclass(frozen=True)
class Component:
    """One antigen-connected component and its balancing weights."""

    key: str
    antigen_digests: tuple[str, ...]
    instance_ids: tuple[str, ...]
    exact_pairs: tuple[tuple[str, str], ...]

    @property
    def exact_pair_count(self) -> int:
        """Return the number of deduplicated antibody-antigen sequence pairs."""
        return len(self.exact_pairs)

    @property
    def instance_count(self) -> int:
        """Return the number of source rows in the component."""
        return len(self.instance_ids)


class UnionFind:
    """Deterministic disjoint-set structure over sequence digests."""

    def __init__(self, values: Iterable[str]) -> None:
        identifiers = sorted(set(values))
        if not identifiers:
            raise ValueError("At least one antigen digest is required")
        self.parent = {value: value for value in identifiers}

    def find(self, value: str) -> str:
        """Return the canonical representative for one digest."""
        if value not in self.parent:
            raise KeyError(f"Unknown digest: {value}")
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        """Join two components while retaining a stable representative."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the builder command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--antigen-similarity-tsv", type=Path, required=True)
    parser.add_argument("--antibody-similarity-tsv", type=Path, default=None)
    parser.add_argument("--needleman-wunsch-antigen-edges", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument(
        "--expected-authority-sha256",
        default=EXPECTED_AUTHORITY_SHA256,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.fold_count != 3:
        parser.error("This protocol requires exactly three folds")
    for name in ("identity_threshold", "coverage_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")
    expected = str(args.expected_authority_sha256).lower()
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        parser.error("--expected-authority-sha256 must be a SHA-256 digest")
    args.expected_authority_sha256 = expected
    return args


def sha256_bytes(value: bytes) -> str:
    """Return a SHA-256 digest for bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize strict JSON deterministically."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_create_or_validate(path: Path, payload: bytes, resume: bool) -> None:
    """Create an immutable file or validate byte-identical resume output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise FileExistsError(f"Output already exists: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"Existing output differs from deterministic rebuild: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted membership without serializing it into provenance."""
    rows = "".join(f"{value}\n" for value in sorted(set(values)))
    return sha256_bytes(rows.encode("utf-8"))


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    """Normalize one sequence and reject unexpected residue symbols."""
    sequence = "".join(str(value).split()).upper()
    if not sequence and allow_empty:
        return sequence
    if not sequence:
        raise ValueError(f"Empty sequence for {label}")
    unexpected = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if unexpected:
        raise ValueError(f"Invalid residue symbols for {label}: {unexpected}")
    return sequence


def sequence_digest(sequence: str) -> str:
    """Return the canonical SHA-256 sequence identity."""
    return sha256_bytes(sequence.encode("ascii"))


def load_authority(
    path: Path,
    expected_sha256: str,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Load source memberships and establish the train-only authorization wall."""
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Authority split digest mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Authority manifest does not contain a splits object")
    required = {"train", "validation", "test"}
    if not required.issubset(raw_splits):
        raise ValueError(f"Authority split is missing memberships: {sorted(required - set(raw_splits))}")

    splits: dict[str, list[str]] = {}
    observed: set[str] = set()
    for name in sorted(required):
        raw_values = raw_splits[name]
        if not isinstance(raw_values, list):
            raise ValueError(f"Authority split {name!r} is not a list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Authority split {name!r} contains duplicates")
        overlap = observed.intersection(values)
        if overlap:
            raise ValueError("Authority split memberships overlap")
        observed.update(values)
        splits[name] = values

    allowed = set(splits["train"])
    sealed = set(splits["validation"]) | set(splits["test"])
    summary = {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "source_pool_split": "train",
        "allowed_instance_count": len(allowed),
        "allowed_membership_sha256": membership_sha256(allowed),
        "sealed_splits": ["validation", "test"],
        "sealed_instance_count": len(sealed),
        "sealed_membership_sha256": membership_sha256(sealed),
        "sealed_split_counts": {
            "validation": len(splits["validation"]),
            "test": len(splits["test"]),
        },
        "sealed_split_membership_sha256": {
            "validation": membership_sha256(splits["validation"]),
            "test": membership_sha256(splits["test"]),
        },
        "authority_intended_use": payload.get("intended_use"),
    }
    return allowed, sealed, summary


def _leading_csv_identifier(line: str) -> str:
    """Read the leading ID without parsing later sequence columns."""
    prefix = line.partition(",")[0].strip()
    if prefix.startswith('"'):
        values = next(csv.reader([prefix]))
        if len(values) != 1:
            raise ValueError("Unable to parse the leading canonical ID")
        return values[0]
    return prefix


def load_source_train_records(
    path: Path,
    allowed_ids: set[str],
) -> dict[str, PairRecord]:
    """Parse sequence fields only after an ID passes source-train authorization."""
    records: dict[str, PairRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("Canonical manifest is empty")
        header = next(csv.reader([header_line]))
        required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"Canonical manifest is missing columns: {missing}")
        if header[0] != "INSTANCE":
            raise ValueError("INSTANCE must be the first canonical-manifest column")
        index = {name: header.index(name) for name in required}

        for line_number, line in enumerate(handle, start=2):
            instance_id = _leading_csv_identifier(line)
            if instance_id not in allowed_ids:
                continue
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError(f"Malformed authorized canonical row at line {line_number}")
            if instance_id in records:
                raise ValueError(f"Duplicate authorized canonical row for {instance_id}")
            heavy = normalize_sequence(fields[index["vh_seq"]], f"{instance_id}:VH")
            light = normalize_sequence(
                fields[index["vl_seq"]], f"{instance_id}:VL", allow_empty=True
            )
            antigen = normalize_sequence(fields[index["ag_seq"]], f"{instance_id}:AG")
            records[instance_id] = PairRecord(
                instance_id=instance_id,
                antibody_digest=sequence_digest(heavy + light),
                antigen_digest=sequence_digest(antigen),
            )

    missing_ids = sorted(allowed_ids - set(records))
    if missing_ids:
        raise ValueError(
            f"Canonical manifest is missing {len(missing_ids)} authorized IDs; "
            f"first={missing_ids[:5]}"
        )
    return records


def normalize_fraction(value: str) -> float:
    """Normalize either a fraction or percentage to a unit interval."""
    result = float(value)
    if result > 1.0:
        result /= 100.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Similarity value is outside [0, 1]: {value}")
    return result


def load_similarity_edges(
    path: Path,
    allowed_digests: set[str],
    identity_threshold: float,
    coverage_threshold: float,
    source: str,
    require_complete_inventory: bool,
) -> tuple[list[SimilarityHit], dict[str, Any]]:
    """Load threshold edges after filtering both endpoints to the authorized pool."""
    hits: dict[tuple[str, str], SimilarityHit] = {}
    observed_allowed: set[str] = set()
    total_row_count = 0
    authorized_row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_row_count += 1
            leading = line.rstrip("\n").split("\t", 2)
            if len(leading) < 2:
                raise ValueError(f"Malformed similarity row at line {line_number}")
            query, target = leading[:2]
            if query not in allowed_digests or target not in allowed_digests:
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"Malformed authorized similarity row at line {line_number}")
            authorized_row_count += 1
            observed_allowed.update((query, target))
            identity, query_coverage, target_coverage = (
                normalize_fraction(value) for value in fields[2:]
            )
            if query == target:
                continue
            if (
                identity + 1e-12 < identity_threshold
                or query_coverage + 1e-12 < coverage_threshold
                or target_coverage + 1e-12 < coverage_threshold
            ):
                continue
            left, right = sorted((query, target))
            candidate = SimilarityHit(
                left=left,
                right=right,
                identity=identity,
                left_coverage=query_coverage if query == left else target_coverage,
                right_coverage=target_coverage if target == right else query_coverage,
                source=source,
            )
            previous = hits.get((left, right))
            candidate_rank = (
                candidate.identity,
                min(candidate.left_coverage, candidate.right_coverage),
                candidate.left_coverage + candidate.right_coverage,
            )
            previous_rank = (
                previous.identity,
                min(previous.left_coverage, previous.right_coverage),
                previous.left_coverage + previous.right_coverage,
            ) if previous is not None else None
            if previous_rank is None or candidate_rank > previous_rank:
                hits[(left, right)] = candidate

    missing = allowed_digests - observed_allowed
    if require_complete_inventory and missing:
        raise ValueError(
            f"Similarity evidence omitted {len(missing)} authorized sequence digests"
        )
    summary = {
        "provided": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "total_row_count": total_row_count,
        "authorized_row_count": authorized_row_count,
        "authorized_unique_digest_count": len(allowed_digests),
        "authorized_observed_digest_count": len(observed_allowed),
        "authorized_unobserved_digest_count": len(missing),
        "qualified_nonself_edge_count": len(hits),
        "identity_threshold": identity_threshold,
        "reciprocal_coverage_threshold": coverage_threshold,
        "filter_before_metric_parsing": True,
    }
    return sorted(hits.values(), key=lambda value: (value.left, value.right)), summary


def combine_edges(*collections: Sequence[SimilarityHit]) -> list[SimilarityHit]:
    """Merge undirected evidence sources while retaining the strongest edge."""
    combined: dict[tuple[str, str], SimilarityHit] = {}
    for edges in collections:
        for edge in edges:
            key = (edge.left, edge.right)
            previous = combined.get(key)
            candidate_rank = (
                edge.identity,
                min(edge.left_coverage, edge.right_coverage),
                edge.source,
            )
            previous_rank = (
                previous.identity,
                min(previous.left_coverage, previous.right_coverage),
                previous.source,
            ) if previous is not None else None
            if previous_rank is None or candidate_rank > previous_rank:
                combined[key] = edge
    return [combined[key] for key in sorted(combined)]


def build_antigen_components(
    records: Mapping[str, PairRecord],
    antigen_edges: Sequence[SimilarityHit],
) -> tuple[list[Component], dict[str, str]]:
    """Construct antigen-only components and aggregate exact-pair weights."""
    antigen_digests = {record.antigen_digest for record in records.values()}
    union_find = UnionFind(antigen_digests)
    for edge in antigen_edges:
        union_find.union(edge.left, edge.right)

    digests_by_root: dict[str, list[str]] = defaultdict(list)
    for digest in sorted(antigen_digests):
        digests_by_root[union_find.find(digest)].append(digest)
    component_key_by_digest: dict[str, str] = {}
    for digests in digests_by_root.values():
        key = membership_sha256(digests)
        for digest in digests:
            component_key_by_digest[digest] = key

    instances_by_component: dict[str, list[str]] = defaultdict(list)
    pairs_by_component: dict[str, set[tuple[str, str]]] = defaultdict(set)
    digests_by_component: dict[str, set[str]] = defaultdict(set)
    for instance_id, record in records.items():
        key = component_key_by_digest[record.antigen_digest]
        instances_by_component[key].append(instance_id)
        pairs_by_component[key].add(record.exact_pair)
        digests_by_component[key].add(record.antigen_digest)

    components = [
        Component(
            key=key,
            antigen_digests=tuple(sorted(digests_by_component[key])),
            instance_ids=tuple(sorted(instances_by_component[key])),
            exact_pairs=tuple(sorted(pairs_by_component[key])),
        )
        for key in sorted(instances_by_component)
    ]
    return components, component_key_by_digest


def assignment_objective(
    assignments: Mapping[str, int],
    components_by_key: Mapping[str, Component],
    fold_count: int,
) -> tuple[int, int, int, int]:
    """Return a lexicographic exact-pair-first balance objective."""
    pair_loads = [0] * fold_count
    instance_loads = [0] * fold_count
    for key, fold_index in assignments.items():
        component = components_by_key[key]
        pair_loads[fold_index] += component.exact_pair_count
        instance_loads[fold_index] += component.instance_count
    pair_range = max(pair_loads) - min(pair_loads)
    instance_range = max(instance_loads) - min(instance_loads)
    pair_square = sum(value * value for value in pair_loads)
    instance_square = sum(value * value for value in instance_loads)
    return pair_range, instance_range, pair_square, instance_square


def balance_components(
    components: Sequence[Component],
    fold_count: int,
    seed: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Balance components by deduplicated pairs first and instances second."""
    if len(components) < fold_count:
        raise ValueError("Fewer antigen components than requested folds")
    components_by_key = {component.key: component for component in components}

    def stable_tie(component: Component) -> str:
        return sha256_bytes(f"{seed}:{component.key}".encode("ascii"))

    ordered = sorted(
        components,
        key=lambda value: (
            -value.exact_pair_count,
            -value.instance_count,
            -len(value.antigen_digests),
            stable_tie(value),
        ),
    )
    pair_loads = [0] * fold_count
    instance_loads = [0] * fold_count
    component_loads = [0] * fold_count
    assignments: dict[str, int] = {}
    for component in ordered:
        fold_index = min(
            range(fold_count),
            key=lambda index: (
                pair_loads[index],
                instance_loads[index],
                component_loads[index],
                sha256_bytes(f"{seed}:fold:{index}".encode("ascii")),
            ),
        )
        assignments[component.key] = fold_index
        pair_loads[fold_index] += component.exact_pair_count
        instance_loads[fold_index] += component.instance_count
        component_loads[fold_index] += 1

    initial_objective = assignment_objective(assignments, components_by_key, fold_count)
    move_count = 0
    while True:
        current = assignment_objective(assignments, components_by_key, fold_count)
        component_counts = [0] * fold_count
        for fold_index in assignments.values():
            component_counts[fold_index] += 1
        best: tuple[tuple[int, int, int, int], str, int] | None = None
        for key in sorted(assignments):
            source_fold = assignments[key]
            if component_counts[source_fold] <= 1:
                continue
            for destination_fold in range(fold_count):
                if destination_fold == source_fold:
                    continue
                candidate = dict(assignments)
                candidate[key] = destination_fold
                objective = assignment_objective(candidate, components_by_key, fold_count)
                if objective >= current:
                    continue
                proposal = (objective, key, destination_fold)
                if best is None or proposal < best:
                    best = proposal
        if best is None:
            break
        _, key, destination_fold = best
        assignments[key] = destination_fold
        move_count += 1

    pair_loads = [0] * fold_count
    instance_loads = [0] * fold_count
    component_loads = [0] * fold_count
    for key, fold_index in assignments.items():
        component = components_by_key[key]
        pair_loads[fold_index] += component.exact_pair_count
        instance_loads[fold_index] += component.instance_count
        component_loads[fold_index] += 1
    summary = {
        "method": "deterministic_lpt_then_single_move_local_search",
        "primary_balance_unit": "unique_exact_antibody_antigen_pair_count",
        "secondary_balance_unit": "instance_count",
        "initial_objective": list(initial_objective),
        "final_objective": list(
            assignment_objective(assignments, components_by_key, fold_count)
        ),
        "local_improvement_move_count": move_count,
        "validation_exact_pair_counts": pair_loads,
        "validation_instance_counts": instance_loads,
        "validation_component_counts": component_loads,
        "exact_pair_count_range": max(pair_loads) - min(pair_loads),
        "instance_count_range": max(instance_loads) - min(instance_loads),
    }
    return assignments, summary


def crossing_hits(
    hits: Sequence[SimilarityHit],
    train_digests: set[str],
    validation_digests: set[str],
) -> list[SimilarityHit]:
    """Return evidence edges whose endpoints cross a fold boundary."""
    return [
        hit
        for hit in hits
        if (hit.left in train_digests and hit.right in validation_digests)
        or (hit.right in train_digests and hit.left in validation_digests)
    ]


def audit_fold(
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    allowed_ids: set[str],
    sealed_ids: set[str],
    records: Mapping[str, PairRecord],
    component_key_by_antigen: Mapping[str, str],
    antigen_edges: Sequence[SimilarityHit],
    antibody_edges: Sequence[SimilarityHit],
) -> dict[str, Any]:
    """Audit exact coverage, sealed exclusion, antigen isolation, and shortcuts."""
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Fold split lists contain duplicate IDs")
    train = set(train_ids)
    validation = set(validation_ids)
    sealed_crossing = (train | validation) & sealed_ids
    if sealed_crossing:
        raise PermissionError(
            f"Fold contains {len(sealed_crossing)} sealed authority identifiers"
        )
    overlap = train & validation
    missing = allowed_ids - (train | validation)
    extra = (train | validation) - allowed_ids
    if overlap or missing or extra:
        raise ValueError(
            "Fold does not exactly partition the authorized source train pool: "
            f"overlap={len(overlap)}, missing={len(missing)}, extra={len(extra)}"
        )

    train_antigens = {records[value].antigen_digest for value in train}
    validation_antigens = {records[value].antigen_digest for value in validation}
    train_antibodies = {records[value].antibody_digest for value in train}
    validation_antibodies = {records[value].antibody_digest for value in validation}
    train_pairs = {records[value].exact_pair for value in train}
    validation_pairs = {records[value].exact_pair for value in validation}
    train_components = {component_key_by_antigen[value] for value in train_antigens}
    validation_components = {
        component_key_by_antigen[value] for value in validation_antigens
    }

    antigen_crossing_edges = crossing_hits(
        antigen_edges,
        train_antigens,
        validation_antigens,
    )
    antibody_crossing_edges = crossing_hits(
        antibody_edges,
        train_antibodies,
        validation_antibodies,
    )
    exact_antibody_crossing = train_antibodies & validation_antibodies
    threshold_crossing_antibodies = {
        digest
        for edge in antibody_crossing_edges
        for digest in (edge.left, edge.right)
    }
    crossing_antibodies = exact_antibody_crossing | threshold_crossing_antibodies
    antibody_affected_ids = {
        value
        for value in train | validation
        if records[value].antibody_digest in crossing_antibodies
    }
    antigen_valid = (
        not (train_antigens & validation_antigens)
        and not (train_components & validation_components)
        and not antigen_crossing_edges
    )
    audit = {
        "valid": bool(antigen_valid and not (train_pairs & validation_pairs)),
        "coverage": {
            "mode": "exact",
            "expected_authorized_instance_count": len(allowed_ids),
            "observed_instance_count": len(train | validation),
            "train_validation_overlap_count": 0,
            "missing_authorized_id_count": 0,
            "extra_unauthorized_id_count": 0,
            "sealed_id_intersection_count": 0,
        },
        "effective_counts": {
            "train": {
                "instance_count": len(train),
                "unique_exact_pair_count": len(train_pairs),
                "duplicate_exact_pair_row_count": len(train) - len(train_pairs),
                "unique_antibody_count": len(train_antibodies),
                "unique_antigen_count": len(train_antigens),
            },
            "validation": {
                "instance_count": len(validation),
                "unique_exact_pair_count": len(validation_pairs),
                "duplicate_exact_pair_row_count": len(validation) - len(validation_pairs),
                "unique_antibody_count": len(validation_antibodies),
                "unique_antigen_count": len(validation_antigens),
            },
        },
        "exact_pair_crossing_count": len(train_pairs & validation_pairs),
        "antigen_isolation": {
            "valid": antigen_valid,
            "exact_sequence_crossing_digest_count": len(
                train_antigens & validation_antigens
            ),
            "component_crossing_count": len(train_components & validation_components),
            "threshold_qualified_crossing_edge_count": len(antigen_crossing_edges),
            "maximum_crossing_identity": max(
                (edge.identity for edge in antigen_crossing_edges),
                default=None,
            ),
        },
        "antibody_overlap": {
            "allowed_by_policy": True,
            "exact_sequence_crossing_digest_count": len(exact_antibody_crossing),
            "threshold_qualified_crossing_edge_count": len(antibody_crossing_edges),
            "affected_instance_count": len(antibody_affected_ids),
            "maximum_crossing_identity": max(
                (edge.identity for edge in antibody_crossing_edges),
                default=None,
            ),
        },
    }
    if not audit["valid"]:
        raise ValueError("Fold failed antigen-isolation audit")
    return audit


def absent_evidence_summary() -> dict[str, Any]:
    """Return an explicit record for an omitted optional evidence source."""
    return {
        "provided": False,
        "qualified_nonself_edge_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build, audit, and atomically publish the three train-only folds."""
    started = time.time()
    args.authority_split = args.authority_split.expanduser().resolve()
    args.canonical_manifest = args.canonical_manifest.expanduser().resolve()
    args.antigen_similarity_tsv = args.antigen_similarity_tsv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.antibody_similarity_tsv is not None:
        args.antibody_similarity_tsv = args.antibody_similarity_tsv.expanduser().resolve()
    if args.needleman_wunsch_antigen_edges is not None:
        args.needleman_wunsch_antigen_edges = (
            args.needleman_wunsch_antigen_edges.expanduser().resolve()
        )
    required_paths = [
        args.authority_split,
        args.canonical_manifest,
        args.antigen_similarity_tsv,
    ]
    optional_paths = [args.antibody_similarity_tsv, args.needleman_wunsch_antigen_edges]
    for path in required_paths + [value for value in optional_paths if value is not None]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required evidence file: {path}")

    allowed_ids, sealed_ids, authority_summary = load_authority(
        args.authority_split,
        args.expected_authority_sha256,
    )
    records = load_source_train_records(args.canonical_manifest, allowed_ids)
    allowed_antigens = {record.antigen_digest for record in records.values()}
    allowed_antibodies = {record.antibody_digest for record in records.values()}

    mmseqs_antigen_edges, antigen_evidence = load_similarity_edges(
        args.antigen_similarity_tsv,
        allowed_antigens,
        args.identity_threshold,
        args.coverage_threshold,
        source="exhaustive_mmseqs_antigen",
        require_complete_inventory=True,
    )
    if args.needleman_wunsch_antigen_edges is None:
        nw_antigen_edges: list[SimilarityHit] = []
        nw_evidence = absent_evidence_summary()
    else:
        nw_antigen_edges, nw_evidence = load_similarity_edges(
            args.needleman_wunsch_antigen_edges,
            allowed_antigens,
            args.identity_threshold,
            args.coverage_threshold,
            source="needleman_wunsch_antigen",
            require_complete_inventory=False,
        )
    antigen_edges = combine_edges(mmseqs_antigen_edges, nw_antigen_edges)

    if args.antibody_similarity_tsv is None:
        antibody_edges: list[SimilarityHit] = []
        antibody_evidence = absent_evidence_summary()
    else:
        antibody_edges, antibody_evidence = load_similarity_edges(
            args.antibody_similarity_tsv,
            allowed_antibodies,
            args.identity_threshold,
            args.coverage_threshold,
            source="exhaustive_mmseqs_antibody",
            require_complete_inventory=True,
        )

    components, component_key_by_antigen = build_antigen_components(
        records,
        antigen_edges,
    )
    assignments, balance_summary = balance_components(
        components,
        args.fold_count,
        args.seed,
    )
    component_by_key = {component.key: component for component in components}
    provenance = {
        "authority": authority_summary,
        "canonical_manifest": {
            "path": str(args.canonical_manifest),
            "sha256": sha256_file(args.canonical_manifest),
            "size_bytes": args.canonical_manifest.stat().st_size,
            "sequence_rows_parsed": "source_train_only",
        },
        "builder_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "similarity_evidence": {
            "antigen_mmseqs": antigen_evidence,
            "antigen_needleman_wunsch": nw_evidence,
            "antibody_mmseqs": antibody_evidence,
        },
        "thresholds": {
            "identity": args.identity_threshold,
            "reciprocal_coverage": args.coverage_threshold,
            "coverage_policy": "both_query_and_target",
        },
        "labels_contacts_predictions_or_checkpoints_read": False,
        "sealed_sequences_read": False,
    }

    fold_payloads: list[dict[str, Any]] = []
    for fold_index in range(args.fold_count):
        validation_components = {
            key for key, assigned_fold in assignments.items() if assigned_fold == fold_index
        }
        validation_ids = sorted(
            instance_id
            for key in validation_components
            for instance_id in component_by_key[key].instance_ids
        )
        validation_set = set(validation_ids)
        train_ids = sorted(allowed_ids - validation_set)
        audit = audit_fold(
            train_ids,
            validation_ids,
            allowed_ids,
            sealed_ids,
            records,
            component_key_by_antigen,
            antigen_edges,
            antibody_edges,
        )
        fold_payloads.append(
            {
                "format_version": FOLD_FORMAT_VERSION,
                "intended_use": "development_only",
                "stage": "train_only_antigen_cluster_model_selection",
                "fold_index": fold_index,
                "fold_count": args.fold_count,
                "seed": args.seed,
                "source_pool_split": "train",
                "coverage_mode": "exact",
                "isolation_policy": "antigen_exact",
                "similarity_isolation_policy": {
                    "role": "antigen",
                    "identity_threshold": args.identity_threshold,
                    "reciprocal_coverage_threshold": args.coverage_threshold,
                    "antibody_overlap_allowed": True,
                },
                "splits": {
                    "train": train_ids,
                    "validation": validation_ids,
                },
                "audit": audit,
                "provenance": provenance,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_entries: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_index, payload in enumerate(fold_payloads):
        path = args.output_dir / f"fold_{fold_index}.json"
        encoded = canonical_json_bytes(payload)
        atomic_create_or_validate(path, encoded, args.resume)
        fold_entries.append(
            {
                "fold_index": fold_index,
                "path": path.name,
                "sha256": sha256_bytes(encoded),
            }
        )
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "train_instance_count": len(payload["splits"]["train"]),
                "validation_instance_count": len(payload["splits"]["validation"]),
                "train_unique_exact_pair_count": payload["audit"]["effective_counts"][
                    "train"
                ]["unique_exact_pair_count"],
                "validation_unique_exact_pair_count": payload["audit"][
                    "effective_counts"
                ]["validation"]["unique_exact_pair_count"],
                "valid": payload["audit"]["valid"],
            }
        )

    component_sizes = sorted(
        (component.instance_count for component in components),
        reverse=True,
    )
    index_payload = {
        "format_version": INDEX_FORMAT_VERSION,
        "intended_use": "development_only",
        "stage": "train_only_antigen_cluster_model_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold_count": args.fold_count,
        "seed": args.seed,
        "source_pool_split": "train",
        "coverage_mode": "exact",
        "isolation_policy": "antigen_exact",
        "folds": fold_entries,
        "fold_summaries": fold_summaries,
        "balance": balance_summary,
        "components": {
            "component_count": len(components),
            "singleton_component_count": sum(value == 1 for value in component_sizes),
            "largest_component_instance_count": component_sizes[0],
            "largest_component_fraction": component_sizes[0] / len(records),
            "top_component_instance_counts": component_sizes[:20],
            "combined_antigen_edge_count": len(antigen_edges),
        },
        "source_train": {
            "instance_count": len(records),
            "unique_exact_pair_count": len(
                {record.exact_pair for record in records.values()}
            ),
            "unique_antibody_count": len(allowed_antibodies),
            "unique_antigen_count": len(allowed_antigens),
        },
        "authority": authority_summary,
        "provenance": provenance,
        "all_folds_valid": all(entry["valid"] for entry in fold_summaries),
        "elapsed_seconds": time.time() - started,
        "runtime": {
            "host": platform.node(),
            "python_version": platform.python_version(),
        },
    }
    index_path = args.output_dir / "index.json"
    if index_path.exists() and args.resume:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        stable_existing = dict(existing)
        stable_candidate = dict(index_payload)
        for value in (stable_existing, stable_candidate):
            value.pop("created_at_utc", None)
            value.pop("elapsed_seconds", None)
            value.pop("runtime", None)
        if stable_existing != stable_candidate:
            raise ValueError("Existing index differs from deterministic rebuild")
    elif index_path.exists():
        raise FileExistsError(f"Output already exists: {index_path}")
    else:
        atomic_create_or_validate(index_path, canonical_json_bytes(index_payload), False)
    print(json.dumps(index_payload, indent=2, sort_keys=True), flush=True)
    return index_payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fold builder without printing sequence content."""
    args = parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
