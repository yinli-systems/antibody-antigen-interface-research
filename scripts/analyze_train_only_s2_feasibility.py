#!/usr/bin/env python3
"""Audit whether whole-component train-only S2 folds are feasible.

This diagnostic reads sequence fields only for identifiers authorized by the
authority ``train`` split. It uses precomputed MMseqs2 similarity evidence to
cluster concatenated antibodies and antigens, then measures connected-component
sizes in the resulting bipartite interaction graph. It does not construct final
folds and does not replace the required exact Needleman--Wunsch audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT_VERSION = "sabdab2_train_only_s2_feasibility_v1"
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class UnionFind:
    """Deterministic disjoint-set structure."""

    def __init__(self, values: Iterable[str]) -> None:
        identifiers = sorted(set(values))
        if not identifiers:
            raise ValueError("UnionFind requires at least one identifier")
        self.parent = {value: value for value in identifiers}

    def find(self, value: str) -> str:
        """Return the canonical representative for one identifier."""
        if value not in self.parent:
            raise KeyError(f"Unknown identifier: {value}")
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        """Join two components with a stable lexical representative."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping deterministically."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted set membership without exposing its values."""
    rows = "".join(f"{value}\n" for value in sorted(set(values)))
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def sequence_sha256(sequence: str) -> str:
    """Hash one normalized amino-acid sequence."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def normalize_sequence(value: str, label: str, allow_empty: bool = False) -> str:
    """Normalize a sequence and reject unsupported residue symbols."""
    sequence = "".join(str(value).split()).upper()
    if not sequence and allow_empty:
        return sequence
    if not sequence:
        raise ValueError(f"Empty sequence for {label}")
    unexpected = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if unexpected:
        raise ValueError(f"Invalid residue symbols for {label}: {unexpected}")
    return sequence


def validate_digest(value: str, label: str) -> str:
    """Validate one lowercase SHA-256 command-line value."""
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def verify_file(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    """Physically verify one immutable input before it is parsed."""
    path = path.expanduser().resolve(strict=True)
    expected_sha256 = validate_digest(expected_sha256, label)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, "
            f"observed={observed_sha256}"
        )
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "size_bytes": path.stat().st_size,
    }


def strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate and non-finite values."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite value {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def load_authority(path: Path) -> tuple[set[str], set[str], dict[str, Any]]:
    """Load memberships and establish a strict train-only authorization wall."""
    payload = strict_json_object(path, "authority split")
    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Authority split must contain exactly train, validation, and test")
    normalized: dict[str, list[str]] = {}
    observed: set[str] = set()
    for name in ("train", "validation", "test"):
        raw_values = splits[name]
        if not isinstance(raw_values, list):
            raise ValueError(f"Authority split {name!r} is not a list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Authority split {name!r} contains duplicate identifiers")
        if observed.intersection(values):
            raise ValueError("Authority memberships overlap")
        observed.update(values)
        normalized[name] = values
    train = set(normalized["train"])
    sealed = set(normalized["validation"]) | set(normalized["test"])
    if not train:
        raise ValueError("Authority train is empty")
    return train, sealed, {
        "source_pool_split": "train",
        "authorized_instance_count": len(train),
        "authorized_membership_sha256": membership_sha256(train),
        "sealed_splits": ["validation", "test"],
        "sealed_instance_count": len(sealed),
        "sealed_membership_sha256": membership_sha256(sealed),
        "sealed_split_counts": {
            "validation": len(normalized["validation"]),
            "test": len(normalized["test"]),
        },
    }


def leading_csv_identifier(line: str) -> str:
    """Parse only the first CSV field before any sequence column is accessed."""
    prefix = line.partition(",")[0].strip()
    if not prefix:
        raise ValueError("Canonical manifest contains an empty leading identifier")
    if prefix.startswith('"'):
        values = next(csv.reader([prefix]))
        if len(values) != 1:
            raise ValueError("Unable to parse the leading canonical identifier")
        return str(values[0])
    return prefix


def load_authorized_pairs(
    path: Path,
    train_ids: set[str],
    sealed_ids: set[str],
) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    """Materialize sequence digests exclusively for authority-train rows."""
    records: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    skipped_before_sequence_parse = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("Canonical manifest is empty")
        header = next(csv.reader([header_line]))
        required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
        if header[0] != "INSTANCE" or not required.issubset(header):
            raise ValueError("Canonical manifest lacks the required leading ID or sequences")
        index = {name: header.index(name) for name in required}
        for line_number, line in enumerate(handle, start=2):
            instance_id = leading_csv_identifier(line)
            if instance_id not in train_ids:
                skipped_before_sequence_parse += 1
                continue
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError(f"Malformed authorized row at line {line_number}")
            if instance_id in seen:
                raise ValueError(f"Duplicate authorized row for {instance_id}")
            seen.add(instance_id)
            heavy = normalize_sequence(fields[index["vh_seq"]], f"{instance_id}:VH")
            light = normalize_sequence(
                fields[index["vl_seq"]], f"{instance_id}:VL", allow_empty=True
            )
            antigen = normalize_sequence(fields[index["ag_seq"]], f"{instance_id}:AG")
            records.append(
                (instance_id, sequence_sha256(heavy + light), sequence_sha256(antigen))
            )
    missing = train_ids - seen
    if missing:
        raise ValueError(f"Canonical manifest is missing {len(missing)} authority-train IDs")
    if seen.intersection(sealed_ids):
        raise PermissionError("A sealed identifier was materialized")
    return sorted(records), {
        "authorized_rows_materialized": len(records),
        "sealed_rows_materialized": 0,
        "non_authorized_rows_skipped_before_sequence_parse": skipped_before_sequence_parse,
        "sequence_columns_parsed_for": "authority_train_only",
    }


def normalize_fraction(value: str, label: str) -> float:
    """Normalize a fraction or percentage to the unit interval."""
    result = float(value)
    if result > 1.0:
        result /= 100.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]")
    return result


def load_similarity_relation(
    path: Path,
    authorized_digests: set[str],
    identity_threshold: float,
    coverage_threshold: float,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Load an authorized threshold relation from a five-column MMseqs2 TSV."""
    qualified: set[tuple[str, str]] = set()
    observed_inventory: set[str] = set()
    total_rows = 0
    authorized_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_rows += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed similarity row at line {line_number}")
            query, target = fields[:2]
            if query not in authorized_digests or target not in authorized_digests:
                continue
            if len(fields) != 5:
                raise ValueError(f"Malformed authorized similarity row at line {line_number}")
            authorized_rows += 1
            observed_inventory.update((query, target))
            identity = normalize_fraction(fields[2], "identity")
            query_coverage = normalize_fraction(fields[3], "query coverage")
            target_coverage = normalize_fraction(fields[4], "target coverage")
            if query == target:
                continue
            if (
                identity + 1e-12 >= identity_threshold
                and query_coverage + 1e-12 >= coverage_threshold
                and target_coverage + 1e-12 >= coverage_threshold
            ):
                qualified.add(tuple(sorted((query, target))))
    missing = authorized_digests - observed_inventory
    return qualified, {
        "total_row_count": total_rows,
        "authorized_row_count": authorized_rows,
        "authorized_unique_digest_count": len(authorized_digests),
        "authorized_observed_digest_count": len(observed_inventory),
        "authorized_unobserved_digest_count": len(missing),
        "qualified_unique_nonself_edge_count": len(qualified),
        "filter_before_metric_parsing": True,
    }


def cluster_entities(
    digests: set[str], edges: set[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Cluster authorized entity digests under the threshold relation."""
    union_find = UnionFind(digests)
    for left, right in sorted(edges):
        union_find.union(left, right)
    root_by_digest = {digest: union_find.find(digest) for digest in sorted(digests)}
    sizes = sorted(Counter(root_by_digest.values()).values(), reverse=True)
    return root_by_digest, {
        "entity_count": len(digests),
        "cluster_count": len(sizes),
        "singleton_cluster_count": sum(size == 1 for size in sizes),
        "largest_cluster_entity_count": sizes[0],
        "largest_cluster_entity_fraction": sizes[0] / len(digests),
        "top_cluster_entity_counts": sizes[:20],
    }


def bipartite_component_summary(
    records: Sequence[tuple[str, str, str]],
    antibody_cluster: Mapping[str, str],
    antigen_cluster: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize components after positive interactions join both entity axes."""
    antibody_nodes = {f"antibody:{value}" for value in antibody_cluster.values()}
    antigen_nodes = {f"antigen:{value}" for value in antigen_cluster.values()}
    union_find = UnionFind(antibody_nodes | antigen_nodes)
    row_edges: list[tuple[str, str]] = []
    exact_pairs: set[tuple[str, str]] = set()
    for _, antibody_digest, antigen_digest in records:
        antibody_node = f"antibody:{antibody_cluster[antibody_digest]}"
        antigen_node = f"antigen:{antigen_cluster[antigen_digest]}"
        union_find.union(antibody_node, antigen_node)
        row_edges.append((antibody_node, antigen_node))
        exact_pairs.add((antibody_digest, antigen_digest))

    rows_by_component: Counter[str] = Counter()
    pairs_by_component: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    antibody_nodes_by_component: defaultdict[str, set[str]] = defaultdict(set)
    antigen_nodes_by_component: defaultdict[str, set[str]] = defaultdict(set)
    for antibody_node, antigen_node in row_edges:
        root = union_find.find(antibody_node)
        rows_by_component[root] += 1
        antibody_nodes_by_component[root].add(antibody_node)
        antigen_nodes_by_component[root].add(antigen_node)
    for antibody_digest, antigen_digest in exact_pairs:
        antibody_node = f"antibody:{antibody_cluster[antibody_digest]}"
        root = union_find.find(antibody_node)
        pairs_by_component[root].add((antibody_digest, antigen_digest))

    ordered = sorted(
        rows_by_component,
        key=lambda root: (
            -rows_by_component[root],
            -len(pairs_by_component[root]),
            -len(antibody_nodes_by_component[root]),
            -len(antigen_nodes_by_component[root]),
            root,
        ),
    )
    top_components = [
        {
            "source_row_count": rows_by_component[root],
            "unique_exact_pair_count": len(pairs_by_component[root]),
            "antibody_cluster_node_count": len(antibody_nodes_by_component[root]),
            "antigen_cluster_node_count": len(antigen_nodes_by_component[root]),
        }
        for root in ordered[:20]
    ]
    largest = top_components[0]
    return {
        "component_count_with_interactions": len(ordered),
        "source_row_count": len(records),
        "unique_exact_pair_count": len(exact_pairs),
        "largest_component": {
            **largest,
            "source_row_fraction": largest["source_row_count"] / len(records),
            "unique_exact_pair_fraction": (
                largest["unique_exact_pair_count"] / len(exact_pairs)
            ),
        },
        "top_components": top_components,
        "whole_component_three_fold_feasible": (
            largest["source_row_count"] <= len(records) / 3
        ),
    }


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable JSON output without overwrite semantics."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--antibody-similarity-tsv", type=Path, required=True)
    parser.add_argument("--antibody-similarity-sha256", required=True)
    parser.add_argument("--antigen-similarity-tsv", type=Path, required=True)
    parser.add_argument("--antigen-similarity-sha256", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("identity_threshold", "coverage_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the train-only diagnostic and return its immutable payload."""
    input_references = {
        "authority_split": verify_file(
            args.authority_split, args.authority_sha256, "authority split"
        ),
        "canonical_sequence_manifest": verify_file(
            args.canonical_manifest,
            args.canonical_manifest_sha256,
            "canonical sequence manifest",
        ),
        "antibody_mmseqs2_similarity": verify_file(
            args.antibody_similarity_tsv,
            args.antibody_similarity_sha256,
            "antibody similarity TSV",
        ),
        "antigen_mmseqs2_similarity": verify_file(
            args.antigen_similarity_tsv,
            args.antigen_similarity_sha256,
            "antigen similarity TSV",
        ),
    }
    train_ids, sealed_ids, authority = load_authority(
        Path(input_references["authority_split"]["path"])
    )
    records, firewall = load_authorized_pairs(
        Path(input_references["canonical_sequence_manifest"]["path"]),
        train_ids,
        sealed_ids,
    )
    antibody_digests = {record[1] for record in records}
    antigen_digests = {record[2] for record in records}
    antibody_edges, antibody_similarity = load_similarity_relation(
        Path(input_references["antibody_mmseqs2_similarity"]["path"]),
        antibody_digests,
        args.identity_threshold,
        args.coverage_threshold,
    )
    antigen_edges, antigen_similarity = load_similarity_relation(
        Path(input_references["antigen_mmseqs2_similarity"]["path"]),
        antigen_digests,
        args.identity_threshold,
        args.coverage_threshold,
    )
    antibody_clusters, antibody_summary = cluster_entities(
        antibody_digests, antibody_edges
    )
    antigen_clusters, antigen_summary = cluster_entities(antigen_digests, antigen_edges)
    component_analysis = bipartite_component_summary(
        records, antibody_clusters, antigen_clusters
    )
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "train_only_s2_whole_component_feasibility_diagnostic",
        "intended_use": "train_only_split_design_diagnostic",
        "valid": True,
        "inputs": input_references,
        "parameters": {
            "identity_threshold": args.identity_threshold,
            "reciprocal_coverage_threshold": args.coverage_threshold,
            "sequence_views": ["concatenated_antibody", "antigen"],
            "similarity_backend": "precomputed_mmseqs2",
        },
        "authority": authority,
        "data_firewall": firewall,
        "similarity_evidence": {
            "antibody": antibody_similarity,
            "antigen": antigen_similarity,
        },
        "entity_clusters": {
            "antibody": antibody_summary,
            "antigen": antigen_summary,
        },
        "bipartite_component_analysis": component_analysis,
        "decision": {
            "whole_bipartite_component_assignment_rejected": not component_analysis[
                "whole_component_three_fold_feasible"
            ],
            "required_next_strategy": (
                "assign antibody and antigen clusters as separate S2 dimensions, "
                "then discard cross-fold interactions"
            ),
        },
        "limitations": [
            "This is an authority-train-only split-design diagnostic, not model evaluation.",
            "MMseqs2 relations are approximate and final folds require exhaustive exact Needleman-Wunsch auditing.",
            "Authorized sequences absent from an MMseqs2 inventory are treated as singleton clusters; adding missed similarity edges can only merge components, so the reported largest component is a lower bound.",
            "The diagnostic uses concatenated antibodies and antigens only; final S2 auditing must also cover VH and VL separately.",
            "No authority-validation or frozen-test sequence, label, embedding, prediction, checkpoint, or score was materialized.",
        ],
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    atomic_create_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entrypoint."""
    payload = run(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
