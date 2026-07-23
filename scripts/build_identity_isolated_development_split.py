#!/usr/bin/env python3
"""Build and audit a sequence-identity-isolated development split.

This utility deliberately reads sequence metadata and precomputed similarity hits
only. It never opens contact labels, pair labels, predictions, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FORMAT_VERSION = "sabdab2_identity_isolated_development_v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write strict JSON through an adjacent temporary file."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_fraction(value: str | float) -> float:
    """Accept either a fraction or a percentage and return a fraction."""
    result = float(value)
    if result > 1.0:
        result /= 100.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"Similarity fraction is outside [0, 1]: {value}")
    return result


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    """Normalize and validate one amino-acid sequence."""
    sequence = "".join(str(value).split()).upper()
    if not sequence and allow_empty:
        return sequence
    if not sequence or not sequence.isalpha() or not sequence.isascii():
        raise ValueError(f"Invalid amino-acid sequence for {label}")
    return sequence


def sequence_digest(sequence: str) -> str:
    """Return the stable digest used by the MMseqs2 FASTA export."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SequenceRecord:
    """One paired antibody-antigen sequence record."""

    instance_id: str
    antibody: str
    antigen: str

    @property
    def antibody_digest(self) -> str:
        return sequence_digest(self.antibody)

    @property
    def antigen_digest(self) -> str:
        return sequence_digest(self.antigen)


@dataclass(frozen=True)
class SimilarityHit:
    """One threshold-qualified MMseqs2 alignment."""

    query: str
    target: str
    identity: float
    query_coverage: float
    target_coverage: float


class UnionFind:
    """Deterministic disjoint-set data structure."""

    def __init__(self, identifiers: Iterable[str]) -> None:
        values = sorted(set(str(value) for value in identifiers))
        if not values:
            raise ValueError("At least one identifier is required")
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return the canonical representative for one identifier."""
        if value not in self.parent:
            raise KeyError(f"Unknown identifier: {value}")
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        """Join two sets while retaining a deterministic representative."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def load_split_lists(path: Path) -> dict[str, list[str]]:
    """Load disjoint split lists without imposing a project-specific format."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Split manifest does not contain a splits object")
    splits: dict[str, list[str]] = {}
    observed: set[str] = set()
    for raw_name, raw_values in raw_splits.items():
        name = str(raw_name)
        if not isinstance(raw_values, list):
            raise ValueError(f"Split {name!r} is not a list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Split {name!r} contains duplicate identifiers")
        overlap = observed.intersection(values)
        if overlap:
            raise ValueError(f"Identifiers occur in multiple splits: {sorted(overlap)[:5]}")
        observed.update(values)
        splits[name] = values
    return splits


def load_development_records(
    manifest_path: Path,
    eligible_ids: set[str],
    id_column: str,
    heavy_column: str,
    light_column: str,
    antigen_column: str,
) -> dict[str, SequenceRecord]:
    """Read only eligible development records from the sequence manifest."""
    records: dict[str, SequenceRecord] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {id_column, heavy_column, light_column, antigen_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Sequence manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            identifier = str(row[id_column])
            if identifier not in eligible_ids:
                continue
            if identifier in records:
                raise ValueError(f"Duplicate eligible instance {identifier!r} at row {row_number}")
            heavy = normalize_sequence(row[heavy_column], f"{identifier}:{heavy_column}")
            light = normalize_sequence(
                row[light_column], f"{identifier}:{light_column}", allow_empty=True
            )
            antigen = normalize_sequence(row[antigen_column], f"{identifier}:{antigen_column}")
            records[identifier] = SequenceRecord(identifier, heavy + light, antigen)
    missing_ids = eligible_ids - set(records)
    if missing_ids:
        raise ValueError(f"Sequence manifest is missing eligible IDs: {sorted(missing_ids)[:10]}")
    return records


def write_unique_fasta(path: Path, sequences: Iterable[str]) -> dict[str, str]:
    """Write one digest-keyed FASTA record per unique normalized sequence."""
    unique_sequences = sorted(set(sequences))
    if not unique_sequences:
        raise ValueError("Cannot export an empty FASTA file")
    sequence_by_digest = {sequence_digest(sequence): sequence for sequence in unique_sequences}
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for digest, sequence in sorted(sequence_by_digest.items()):
            handle.write(f">{digest}\n{sequence}\n")
    os.replace(temporary, path)
    return sequence_by_digest


def load_similarity_hits(
    path: Path,
    allowed_digests: set[str],
    identity_threshold: float,
    coverage_threshold: float,
) -> tuple[list[SimilarityHit], dict[str, object]]:
    """Load threshold-qualified hits involving development sequences only."""
    hits_by_pair: dict[tuple[str, str], SimilarityHit] = {}
    observed_development_digests: set[str] = set()
    parsed_rows = 0
    development_rows = 0
    malformed_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                malformed_rows += 1
                continue
            parsed_rows += 1
            query, target = fields[:2]
            if query not in allowed_digests or target not in allowed_digests:
                continue
            development_rows += 1
            observed_development_digests.update((query, target))
            identity, query_coverage, target_coverage = (
                normalize_fraction(value) for value in fields[2:]
            )
            if (
                query == target
                or identity + 1e-12 < identity_threshold
                or query_coverage + 1e-12 < coverage_threshold
                or target_coverage + 1e-12 < coverage_threshold
            ):
                continue
            key = tuple(sorted((query, target)))
            candidate = SimilarityHit(
                query=key[0],
                target=key[1],
                identity=identity,
                query_coverage=query_coverage,
                target_coverage=target_coverage,
            )
            existing = hits_by_pair.get(key)
            if existing is None or (
                candidate.identity,
                min(candidate.query_coverage, candidate.target_coverage),
            ) > (
                existing.identity,
                min(existing.query_coverage, existing.target_coverage),
            ):
                hits_by_pair[key] = candidate
    if malformed_rows:
        raise ValueError(f"Similarity artifact contains {malformed_rows} malformed rows")
    summary = {
        "path": str(path.expanduser().resolve()),
        "sha256": sha256_file(path),
        "parsed_row_count": parsed_rows,
        "development_row_count": development_rows,
        "qualified_nonself_digest_pair_count": len(hits_by_pair),
        "development_unique_digest_count": len(allowed_digests),
        "development_observed_digest_count": len(observed_development_digests),
        "development_unobserved_digest_count": len(allowed_digests - observed_development_digests),
        "development_unobserved_digest_sha256": hashlib.sha256(
            "\n".join(sorted(allowed_digests - observed_development_digests)).encode("ascii")
        ).hexdigest(),
    }
    return sorted(hits_by_pair.values(), key=lambda item: (item.query, item.target)), summary


def group_by_digest(
    records: Mapping[str, SequenceRecord], role: str
) -> dict[str, list[str]]:
    """Group development instances by an exact sequence digest."""
    if role not in {"antibody", "antigen"}:
        raise ValueError(f"Unknown sequence role: {role}")
    result: dict[str, list[str]] = defaultdict(list)
    for identifier, record in records.items():
        digest = record.antibody_digest if role == "antibody" else record.antigen_digest
        result[digest].append(identifier)
    return {digest: sorted(values) for digest, values in sorted(result.items())}


def build_components(
    records: Mapping[str, SequenceRecord],
    antibody_hits: Sequence[SimilarityHit],
    antigen_hits: Sequence[SimilarityHit],
) -> tuple[list[tuple[str, ...]], dict[str, object]]:
    """Build components induced by either antibody or antigen similarity."""
    union_find = UnionFind(records)
    role_groups = {
        "antibody": group_by_digest(records, "antibody"),
        "antigen": group_by_digest(records, "antigen"),
    }
    for groups in role_groups.values():
        for members in groups.values():
            for member in members[1:]:
                union_find.union(members[0], member)
    for role, hits in (("antibody", antibody_hits), ("antigen", antigen_hits)):
        groups = role_groups[role]
        for hit in hits:
            if hit.query not in groups or hit.target not in groups:
                raise ValueError(f"{role} hit references an unknown development digest")
            union_find.union(groups[hit.query][0], groups[hit.target][0])

    members: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(records):
        members[union_find.find(identifier)].append(identifier)
    components = sorted(
        (tuple(values) for values in members.values()),
        key=lambda values: (-len(values), values),
    )
    sizes = [len(component) for component in components]
    summary = {
        "component_count": len(components),
        "largest_component_size": max(sizes),
        "largest_component_fraction": max(sizes) / len(records),
        "singleton_component_count": sum(size == 1 for size in sizes),
        "top_component_sizes": sorted(sizes, reverse=True)[:20],
    }
    return components, summary


def choose_validation_components(
    components: Sequence[Sequence[str]],
    target_count: int,
    seed: int,
    minimum_validation_count: int,
    minimum_train_count: int,
) -> tuple[set[int], dict[str, object]]:
    """Choose a deterministic subset whose size is closest to the target."""
    if not components:
        raise ValueError("No components are available")
    total = sum(len(component) for component in components)
    if minimum_validation_count < 1 or minimum_train_count < 1:
        raise ValueError("Minimum split sizes must be positive")
    if minimum_validation_count + minimum_train_count > total:
        raise ValueError("Minimum split sizes exceed the development pool")

    ordered_indices = sorted(
        range(len(components)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{','.join(sorted(components[index]))}".encode("utf-8")
        ).hexdigest(),
    )
    backpointer: dict[int, tuple[int, int] | None] = {0: None}
    for component_index in ordered_indices:
        size = len(components[component_index])
        additions: dict[int, tuple[int, int]] = {}
        for previous_sum in sorted(backpointer, reverse=True):
            candidate_sum = previous_sum + size
            if candidate_sum not in backpointer and candidate_sum not in additions:
                additions[candidate_sum] = (previous_sum, component_index)
        backpointer.update(additions)

    feasible = [
        count
        for count in backpointer
        if count >= minimum_validation_count and total - count >= minimum_train_count
    ]
    if not feasible:
        raise ValueError("No component-isolated split satisfies the minimum sizes")
    selected_count = min(
        feasible,
        key=lambda count: (
            abs(count - target_count),
            count > target_count,
            count,
        ),
    )
    selected_indices: set[int] = set()
    cursor = selected_count
    while cursor:
        link = backpointer[cursor]
        if link is None:
            raise AssertionError("Broken subset-sum backpointer")
        cursor, component_index = link
        selected_indices.add(component_index)
    return selected_indices, {
        "target_validation_instance_count": target_count,
        "selected_validation_instance_count": selected_count,
        "absolute_target_error": abs(selected_count - target_count),
        "selected_validation_component_count": len(selected_indices),
        "optimization": "exact deterministic subset-sum over connected-component sizes",
    }


def assignments_from_components(
    components: Sequence[Sequence[str]], selected_validation: set[int]
) -> dict[str, str]:
    """Convert selected component indices into instance assignments."""
    assignments: dict[str, str] = {}
    for index, component in enumerate(components):
        split_name = "validation" if index in selected_validation else "train"
        for identifier in component:
            assignments[str(identifier)] = split_name
    return assignments


def _role_crossing_audit(
    digest_groups: Mapping[str, Sequence[str]],
    hits: Sequence[SimilarityHit],
    assignments: Mapping[str, str],
) -> dict[str, object]:
    """Count exact and aligned sequence relationships crossing splits."""
    exact_crossing_digests = []
    for digest, members in digest_groups.items():
        if len({assignments[member] for member in members}) > 1:
            exact_crossing_digests.append(digest)
    crossing_hits = []
    affected_instances: set[str] = set()
    for hit in hits:
        query_members = digest_groups[hit.query]
        target_members = digest_groups[hit.target]
        query_splits = {assignments[member] for member in query_members}
        target_splits = {assignments[member] for member in target_members}
        if any(left != right for left in query_splits for right in target_splits):
            crossing_hits.append(hit)
            affected_instances.update(query_members)
            affected_instances.update(target_members)
    return {
        "valid": not exact_crossing_digests and not crossing_hits,
        "exact_sequence_crossing_digest_count": len(exact_crossing_digests),
        "threshold_qualified_crossing_digest_pair_count": len(crossing_hits),
        "crossing_affected_instance_count": len(affected_instances),
        "maximum_crossing_identity": (
            max(hit.identity for hit in crossing_hits) if crossing_hits else None
        ),
        "crossing_digest_pair_examples": [
            {
                "query": hit.query,
                "target": hit.target,
                "identity": hit.identity,
                "query_coverage": hit.query_coverage,
                "target_coverage": hit.target_coverage,
            }
            for hit in crossing_hits[:20]
        ],
    }


def audit_assignments(
    records: Mapping[str, SequenceRecord],
    assignments: Mapping[str, str],
    antibody_hits: Sequence[SimilarityHit],
    antigen_hits: Sequence[SimilarityHit],
) -> dict[str, object]:
    """Verify double-remote isolation and summarize effective sample counts."""
    if set(records) != set(assignments):
        raise ValueError("Assignments must cover every development record exactly once")
    split_counts = Counter(assignments.values())
    if set(split_counts) != {"train", "validation"} or min(split_counts.values()) < 1:
        raise ValueError("Assignments must contain non-empty train and validation splits")

    role_groups = {
        "antibody": group_by_digest(records, "antibody"),
        "antigen": group_by_digest(records, "antigen"),
    }
    role_audits = {
        "antibody": _role_crossing_audit(role_groups["antibody"], antibody_hits, assignments),
        "antigen": _role_crossing_audit(role_groups["antigen"], antigen_hits, assignments),
    }
    effective_counts: dict[str, dict[str, int]] = {}
    sequence_sets: dict[str, dict[str, set[str] | set[tuple[str, str]]]] = {}
    for split_name in ("train", "validation"):
        selected = [record for key, record in records.items() if assignments[key] == split_name]
        antibodies = {record.antibody_digest for record in selected}
        antigens = {record.antigen_digest for record in selected}
        pairs = {(record.antibody_digest, record.antigen_digest) for record in selected}
        sequence_sets[split_name] = {
            "antibodies": antibodies,
            "antigens": antigens,
            "pairs": pairs,
        }
        effective_counts[split_name] = {
            "instance_count": len(selected),
            "unique_antibody_sequence_count": len(antibodies),
            "unique_antigen_sequence_count": len(antigens),
            "unique_exact_pair_count": len(pairs),
            "duplicate_exact_pair_row_count": len(selected) - len(pairs),
        }
    exact_overlap = {
        "antibody_sequence_count": len(
            sequence_sets["train"]["antibodies"]
            & sequence_sets["validation"]["antibodies"]
        ),
        "antigen_sequence_count": len(
            sequence_sets["train"]["antigens"]
            & sequence_sets["validation"]["antigens"]
        ),
        "exact_pair_count": len(
            sequence_sets["train"]["pairs"] & sequence_sets["validation"]["pairs"]
        ),
    }
    valid = all(audit["valid"] for audit in role_audits.values()) and not any(
        exact_overlap.values()
    )
    return {
        "valid": valid,
        "split_instance_counts": dict(sorted(split_counts.items())),
        "effective_counts": effective_counts,
        "exact_train_validation_overlap": exact_overlap,
        "roles": role_audits,
    }


def load_evidence(
    records: Mapping[str, SequenceRecord], args: argparse.Namespace
) -> tuple[list[SimilarityHit], list[SimilarityHit], dict[str, object]]:
    """Load both role-specific similarity artifacts with provenance checks."""
    identity_threshold = normalize_fraction(args.identity_threshold)
    coverage_threshold = normalize_fraction(args.coverage_threshold)
    artifact_identity = {
        "antibody": normalize_fraction(args.antibody_artifact_min_identity),
        "antigen": normalize_fraction(args.antigen_artifact_min_identity),
    }
    artifact_coverage = normalize_fraction(args.artifact_min_coverage)
    for role, minimum in artifact_identity.items():
        if minimum > identity_threshold + 1e-12:
            raise ValueError(
                f"{role} artifact was filtered above the requested identity threshold"
            )
    if artifact_coverage > coverage_threshold + 1e-12:
        raise ValueError("Similarity artifacts were filtered above the requested coverage")

    antibody_groups = group_by_digest(records, "antibody")
    antigen_groups = group_by_digest(records, "antigen")
    antibody_hits, antibody_summary = load_similarity_hits(
        args.antibody_hits,
        set(antibody_groups),
        identity_threshold,
        coverage_threshold,
    )
    antigen_hits, antigen_summary = load_similarity_hits(
        args.antigen_hits,
        set(antigen_groups),
        identity_threshold,
        coverage_threshold,
    )
    evidence = {
        "identity_definition": "MMseqs2 identical residues divided by alignment length",
        "coverage_policy": "both query and target coverage must meet the threshold",
        "identity_threshold": identity_threshold,
        "coverage_threshold": coverage_threshold,
        "artifact_declared_minimums": {
            "antibody_identity": artifact_identity["antibody"],
            "antigen_identity": artifact_identity["antigen"],
            "coverage": artifact_coverage,
        },
        "antibody": antibody_summary,
        "antigen": antigen_summary,
    }
    return antibody_hits, antigen_hits, evidence


def _eligible_ids(
    source_splits: Mapping[str, Sequence[str]],
    pool_split_names: Sequence[str],
    sealed_split_names: Sequence[str],
) -> tuple[set[str], dict[str, int]]:
    """Resolve development IDs while keeping sealed IDs out of all outputs."""
    overlap = set(pool_split_names) & set(sealed_split_names)
    if overlap:
        raise ValueError(f"Pool and sealed split names overlap: {sorted(overlap)}")
    unknown = (set(pool_split_names) | set(sealed_split_names)) - set(source_splits)
    if unknown:
        raise ValueError(f"Unknown source split names: {sorted(unknown)}")
    eligible = {identifier for name in pool_split_names for identifier in source_splits[name]}
    sealed_counts = {name: len(source_splits[name]) for name in sealed_split_names}
    if not eligible:
        raise ValueError("Development pool is empty")
    return eligible, sealed_counts


def common_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by the build and audit commands."""
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--antibody-hits", type=Path, required=True)
    parser.add_argument("--antigen-hits", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--antibody-artifact-min-identity", type=float, default=0.80)
    parser.add_argument("--antigen-artifact-min-identity", type=float, default=0.70)
    parser.add_argument("--artifact-min-coverage", type=float, default=0.80)
    parser.add_argument("--id-column", default="INSTANCE")
    parser.add_argument("--heavy-column", default="vh_seq")
    parser.add_argument("--light-column", default="vl_seq")
    parser.add_argument("--antigen-column", default="ag_seq")
    parser.add_argument("--output", type=Path, required=True)


def command_build(args: argparse.Namespace) -> None:
    """Build a deterministic train-validation split from an unsealed pool."""
    source_splits = load_split_lists(args.source_split)
    eligible_ids, sealed_counts = _eligible_ids(
        source_splits, args.pool_splits, args.sealed_splits
    )
    records = load_development_records(
        args.manifest,
        eligible_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    antibody_hits, antigen_hits, evidence = load_evidence(records, args)
    components, component_summary = build_components(records, antibody_hits, antigen_hits)
    target = round(len(records) * args.validation_fraction)
    selected, selection_summary = choose_validation_components(
        components,
        target_count=target,
        seed=args.seed,
        minimum_validation_count=args.minimum_validation_count,
        minimum_train_count=args.minimum_train_count,
    )
    assignments = assignments_from_components(components, selected)
    audit = audit_assignments(records, assignments, antibody_hits, antigen_hits)
    if not audit["valid"]:
        raise AssertionError("Generated split failed the double-remote leakage audit")
    split_lists = {
        name: sorted(identifier for identifier, value in assignments.items() if value == name)
        for name in ("train", "validation")
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": "build_identity_isolated_development_split",
        "intended_use": "development_only",
        "source_split_sha256": sha256_file(args.source_split),
        "sequence_manifest_sha256": sha256_file(args.manifest),
        "source_pool_splits": list(args.pool_splits),
        "sealed_source_splits": list(args.sealed_splits),
        "sealed_source_split_counts": sealed_counts,
        "frozen_test_labels_or_predictions_read": False,
        "seed": args.seed,
        "validation_fraction_target": args.validation_fraction,
        "splits": split_lists,
        "component_summary": component_summary,
        "selection_summary": selection_summary,
        "similarity_evidence": evidence,
        "audit": audit,
        "limitations": [
            "MMseqs2 is a sensitive heuristic search, not an exhaustive exact aligner.",
            "Unobserved low-complexity sequence digests require a masking-disabled rerun before a publication-grade leakage claim.",
            "This development split must not be described as a new prospective or external test set.",
            "Model selection must use validation only; every sealed test remains locked until the full protocol is frozen.",
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "component_summary": component_summary,
        "selection_summary": selection_summary,
        "audit": audit,
    }, indent=2, sort_keys=True))


def command_export_fasta(args: argparse.Namespace) -> None:
    """Export development-only sequence FASTAs for an independent search."""
    source_splits = load_split_lists(args.source_split)
    eligible_ids, sealed_counts = _eligible_ids(
        source_splits, args.pool_splits, args.sealed_splits
    )
    records = load_development_records(
        args.manifest,
        eligible_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    output_dir = args.output_dir.expanduser().resolve()
    antibody_path = output_dir / "development_antibody.fasta"
    antigen_path = output_dir / "development_antigen.fasta"
    antibodies = write_unique_fasta(antibody_path, (record.antibody for record in records.values()))
    antigens = write_unique_fasta(antigen_path, (record.antigen for record in records.values()))
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": "export_development_similarity_fastas",
        "intended_use": "development_only",
        "source_split_sha256": sha256_file(args.source_split),
        "sequence_manifest_sha256": sha256_file(args.manifest),
        "source_pool_splits": list(args.pool_splits),
        "sealed_source_splits": list(args.sealed_splits),
        "sealed_source_split_counts": sealed_counts,
        "frozen_test_labels_or_predictions_read": False,
        "development_instance_count": len(records),
        "antibody_unique_sequence_count": len(antibodies),
        "antigen_unique_sequence_count": len(antigens),
        "files": {
            "antibody": {
                "path": str(antibody_path),
                "sha256": sha256_file(antibody_path),
            },
            "antigen": {
                "path": str(antigen_path),
                "sha256": sha256_file(antigen_path),
            },
        },
    }
    atomic_write_json(output_dir / "fasta_export.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_audit(args: argparse.Namespace) -> None:
    """Audit two existing development splits without reading target labels."""
    source_splits = load_split_lists(args.split)
    eligible_ids, sealed_counts = _eligible_ids(
        source_splits,
        [args.train_split, args.validation_split],
        args.sealed_splits,
    )
    records = load_development_records(
        args.manifest,
        eligible_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    assignments = {
        identifier: split_name
        for split_name, source_name in (
            ("train", args.train_split),
            ("validation", args.validation_split),
        )
        for identifier in source_splits[source_name]
    }
    antibody_hits, antigen_hits, evidence = load_evidence(records, args)
    components, component_summary = build_components(records, antibody_hits, antigen_hits)
    audit = audit_assignments(records, assignments, antibody_hits, antigen_hits)
    payload = {
        "format_version": FORMAT_VERSION,
        "stage": "audit_identity_isolated_development_split",
        "intended_use": "development_only",
        "audited_split_sha256": sha256_file(args.split),
        "sequence_manifest_sha256": sha256_file(args.manifest),
        "audited_splits": [args.train_split, args.validation_split],
        "sealed_source_splits": list(args.sealed_splits),
        "sealed_source_split_counts": sealed_counts,
        "frozen_test_labels_or_predictions_read": False,
        "component_summary": component_summary,
        "similarity_evidence": evidence,
        "audit": audit,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "component_summary": component_summary,
        "audit": audit,
    }, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-fasta")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--source-split", type=Path, required=True)
    export.add_argument("--pool-splits", nargs="+", default=["train", "validation"])
    export.add_argument("--sealed-splits", nargs="+", default=["test"])
    export.add_argument("--id-column", default="INSTANCE")
    export.add_argument("--heavy-column", default="vh_seq")
    export.add_argument("--light-column", default="vl_seq")
    export.add_argument("--antigen-column", default="ag_seq")
    export.add_argument("--output-dir", type=Path, required=True)
    export.set_defaults(handler=command_export_fasta)

    build = subparsers.add_parser("build")
    common_parser_arguments(build)
    build.add_argument("--source-split", type=Path, required=True)
    build.add_argument("--pool-splits", nargs="+", default=["train", "validation"])
    build.add_argument("--sealed-splits", nargs="+", default=["test"])
    build.add_argument("--validation-fraction", type=float, default=0.10)
    build.add_argument("--minimum-validation-count", type=int, default=100)
    build.add_argument("--minimum-train-count", type=int, default=1000)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(handler=command_build)

    audit = subparsers.add_parser("audit")
    common_parser_arguments(audit)
    audit.add_argument("--split", type=Path, required=True)
    audit.add_argument("--train-split", default="train")
    audit.add_argument("--validation-split", default="validation")
    audit.add_argument("--sealed-splits", nargs="*", default=["test"])
    audit.set_defaults(handler=command_audit)
    return parser


def main() -> None:
    """Run the requested identity-isolation operation."""
    args = build_parser().parse_args()
    if hasattr(args, "validation_fraction") and not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("Validation fraction must be inside (0, 1)")
    args.handler(args)


if __name__ == "__main__":
    main()
