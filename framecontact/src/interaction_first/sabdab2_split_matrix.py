"""Label-free original, temporal, and antigen-fold SAbDab2 splits.

The split matrix is built from identifiers, deposition dates, exact sequence
digests, PDB assemblies, and an authenticated Foldseek all-vs-all table.  Task
labels and predictions are neither required nor read.
"""

from __future__ import annotations

import csv
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file
from .structural_novelty import FOLDSEEK_RECEIPT_FORMAT


SPLIT_MATRIX_FORMAT = "sabdab2_framecontact_split_matrix_v1"
ROLES = ("train", "validation", "test")
METADATA_CLUSTER_POLICIES = ("ignore", "antigen", "double_cold")
OUTPUT_COLUMNS = (
    "instance_id",
    "role",
    "antigen_structure_id",
    "antigen_sequence_sha256",
    "antibody_sequence_sha256",
    "pdb_assembly_id",
    "deposition_date",
    "antigen_cluster_id",
    "antibody_cluster_id",
    "source_split",
)


class SplitMatrixError(ValueError):
    """Raised when a split authority or isolation invariant is invalid."""


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def structure_ids_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or key not in reader.fieldnames:
                raise SplitMatrixError(f"{path} lacks key column {key!r}")
            result: dict[str, dict[str, str]] = {}
            for row in reader:
                identifier = str(row.get(key, "")).strip()
                if not identifier or identifier in result:
                    raise SplitMatrixError(f"Invalid or duplicate identifier {identifier!r}")
                result[identifier] = {
                    str(name): "" if value is None else str(value).strip()
                    for name, value in row.items()
                }
            return result
    except OSError as error:
        raise SplitMatrixError(f"Cannot read {path}") from error


def _load_prepared(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("pandas with a Parquet engine is required") from error
    required = {
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "paratope_split",
        "epitope_split",
    }
    frame = pd.read_parquet(path, columns=sorted(required))
    if not required.issubset(frame.columns):
        raise SplitMatrixError(
            f"Prepared authority lacks columns: {sorted(required - set(frame.columns))}"
        )
    if frame["instance_id"].isna().any() or frame["instance_id"].duplicated().any():
        raise SplitMatrixError("Prepared authority has null or duplicate instance IDs")
    if not (frame["paratope_split"] == frame["epitope_split"]).all():
        raise SplitMatrixError("Original paratope and epitope roles must agree")
    return {
        str(row["instance_id"]): row for row in frame.to_dict(orient="records")
    }


def _parse_date(value: str, identifier: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise SplitMatrixError(
            f"{identifier}: PDBdepo must be an ISO calendar date"
        ) from error


def _base_rows(
    metadata: dict[str, dict[str, str]], prepared: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    missing_metadata = set(prepared) - set(metadata)
    if missing_metadata:
        raise SplitMatrixError(
            "Every prepared instance must be present in the source metadata; "
            f"missing={len(missing_metadata)}"
        )
    rows = []
    for identifier in sorted(prepared):
        meta, source = metadata[identifier], prepared[identifier]
        required_meta = ("PDB_ID", "PDBdepo", "agclusters", "ab_cluster")
        missing = [name for name in required_meta if not meta.get(name)]
        if missing:
            raise SplitMatrixError(f"{identifier}: missing metadata {missing}")
        role = str(source["epitope_split"])
        if role not in ROLES:
            raise SplitMatrixError(f"{identifier}: invalid original role {role!r}")
        antibody_sha = str(source["ab_sequence_sha256"])
        antigen_sha = str(source["ag_sequence_sha256"])
        if not all(
            len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in (antibody_sha, antigen_sha)
        ):
            raise SplitMatrixError(f"{identifier}: invalid sequence SHA-256")
        rows.append(
            {
                "instance_id": identifier,
                "deposition_date": _parse_date(meta["PDBdepo"], identifier),
                "pdb_assembly_id": meta["PDB_ID"],
                "antigen_structure_id": identifier,
                "antigen_sequence_sha256": antigen_sha,
                "antibody_sequence_sha256": antibody_sha,
                "antigen_cluster_id": meta["agclusters"],
                "antibody_cluster_id": meta["ab_cluster"],
                "original_role": role,
            }
        )
    return rows


def _union_equal_columns(
    rows: list[dict[str, Any]], union_find: _UnionFind, columns: Iterable[str]
) -> None:
    for name in columns:
        first: dict[str, str] = {}
        for row in rows:
            value, identifier = str(row[name]), str(row["instance_id"])
            if value in first:
                union_find.union(first[value], identifier)
            else:
                first[value] = identifier


def _metadata_cluster_values(row: dict[str, Any], column: str) -> tuple[str, ...]:
    value = str(row[column]).strip()
    if column == "antigen_cluster_id":
        values = tuple(
            sorted({token.strip() for token in value.split("/") if token.strip()})
        )
        if not values:
            raise SplitMatrixError(
                f"{row['instance_id']}: antigen metadata cluster list is empty"
            )
        return values
    if not value:
        raise SplitMatrixError(f"{row['instance_id']}: {column} is empty")
    return (value,)


def _union_metadata_cluster_columns(
    rows: list[dict[str, Any]], union_find: _UnionFind, columns: Iterable[str]
) -> None:
    """Union structures sharing any SAbDab2 metadata cluster token."""

    for name in columns:
        first: dict[str, str] = {}
        for row in rows:
            identifier = str(row["instance_id"])
            for value in _metadata_cluster_values(row, name):
                if value in first:
                    union_find.union(first[value], identifier)
                else:
                    first[value] = identifier


def _union_equal_values(rows: list[dict[str, Any]], union_find: _UnionFind) -> None:
    _union_equal_columns(
        rows,
        union_find,
        (
            "pdb_assembly_id",
            "antigen_sequence_sha256",
            "antibody_sequence_sha256",
        ),
    )


def _components(rows: list[dict[str, Any]], union_find: _UnionFind) -> list[list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        identifier = str(row["instance_id"])
        result.setdefault(union_find.find(identifier), []).append(identifier)
    return [sorted(values) for values in result.values()]


def _role_counts(role_by_instance: dict[str, str]) -> dict[str, int]:
    return {role: sum(value == role for value in role_by_instance.values()) for role in ROLES}


def _cross_role_count(
    rows: list[dict[str, Any]], role_by_instance: dict[str, str], column: str
) -> int:
    roles: dict[str, set[str]] = {}
    for row in rows:
        roles.setdefault(str(row[column]), set()).add(role_by_instance[row["instance_id"]])
    return sum(len(values) > 1 for values in roles.values())


def _field_overlap_audit(
    rows: list[dict[str, Any]], role_by_instance: dict[str, str], column: str
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for value in _metadata_cluster_values(row, column):
            grouped.setdefault(value, []).append(row)
    crossing = {
        value: members
        for value, members in grouped.items()
        if len({role_by_instance[str(member["instance_id"])] for member in members}) > 1
    }
    role_set_counts: dict[str, int] = {}
    for members in crossing.values():
        role_set = "/".join(
            sorted({role_by_instance[str(member["instance_id"])] for member in members})
        )
        role_set_counts[role_set] = role_set_counts.get(role_set, 0) + 1
    crossing_identifiers = {
        str(member["instance_id"])
        for members in crossing.values()
        for member in members
    }
    return {
        "value_semantics": (
            "slash_separated_chain_cluster_tokens"
            if column == "antigen_cluster_id"
            else (
                "scalar_cluster_id"
                if column == "antibody_cluster_id"
                else "scalar_value"
            )
        ),
        "unique_value_count": len(grouped),
        "cross_role_value_count": len(crossing),
        "rows_in_cross_role_values": len(crossing_identifiers),
        "largest_cross_role_value_size": max(
            (len(members) for members in crossing.values()), default=0
        ),
        "cross_role_role_set_counts": dict(sorted(role_set_counts.items())),
    }


def metadata_cluster_isolation_audit(
    rows: list[dict[str, Any]], role_by_instance: dict[str, str]
) -> dict[str, Any]:
    """Report antigen/antibody metadata-cluster leakage without reading labels."""

    fields = {
        column: _field_overlap_audit(rows, role_by_instance, column)
        for column in ("antigen_cluster_id", "antibody_cluster_id")
    }
    return {
        "fields": fields,
        "valid": all(value["cross_role_value_count"] == 0 for value in fields.values()),
    }


def audit_split_manifest_shortcuts(
    path: Path, expected_sha256: str
) -> dict[str, Any]:
    """Audit exact and metadata-cluster cross-role shortcuts in a frozen CSV."""

    authenticated_sha256 = authenticate_file(path, expected_sha256, "split manifest")
    loaded = _load_csv(path, "instance_id")
    required = {
        "role",
        "pdb_assembly_id",
        "antigen_sequence_sha256",
        "antibody_sequence_sha256",
        "antigen_cluster_id",
        "antibody_cluster_id",
    }
    rows: list[dict[str, Any]] = []
    role_by_instance: dict[str, str] = {}
    for identifier, row in sorted(loaded.items()):
        missing = sorted(required - set(row))
        if missing:
            raise SplitMatrixError(f"Split manifest lacks columns: {missing}")
        role = str(row["role"])
        if role not in ROLES:
            raise SplitMatrixError(f"{identifier}: invalid role {role!r}")
        normalized = {**row, "instance_id": identifier}
        if any(not str(normalized[column]) for column in required - {"role"}):
            raise SplitMatrixError(f"{identifier}: empty split isolation field")
        rows.append(normalized)
        role_by_instance[identifier] = role
    exact_fields = {
        column: _field_overlap_audit(rows, role_by_instance, column)
        for column in (
            "pdb_assembly_id",
            "antigen_sequence_sha256",
            "antibody_sequence_sha256",
        )
    }
    metadata = metadata_cluster_isolation_audit(rows, role_by_instance)
    exact_valid = all(
        value["cross_role_value_count"] == 0 for value in exact_fields.values()
    )
    return {
        "format_version": "sabdab2_split_shortcut_audit_v1",
        "split_manifest": str(path),
        "split_manifest_sha256": authenticated_sha256,
        "instance_count": len(rows),
        "role_counts": _role_counts(role_by_instance),
        "exact_isolation": {"fields": exact_fields, "valid": exact_valid},
        "metadata_cluster_isolation": metadata,
        "double_cold_valid": exact_valid and metadata["valid"],
        "task_labels_read": False,
        "predictions_read": False,
        "claim_boundary": (
            "Label-free split shortcut audit only; no model, generalization, or "
            "benchmark-improvement claim."
        ),
    }


def _isolation_audit(
    rows: list[dict[str, Any]], role_by_instance: dict[str, str]
) -> dict[str, Any]:
    counts = _role_counts(role_by_instance)
    overlaps = {
        name: _cross_role_count(rows, role_by_instance, name)
        for name in (
            "pdb_assembly_id",
            "antigen_sequence_sha256",
            "antibody_sequence_sha256",
        )
    }
    return {
        "role_counts": counts,
        "cross_role_overlap_counts": overlaps,
        "valid": all(value > 0 for value in counts.values())
        and all(value == 0 for value in overlaps.values()),
    }


def _temporal_roles(
    rows: list[dict[str, Any]], *, train_cutoff: date, validation_cutoff: date
) -> tuple[dict[str, str], dict[str, Any]]:
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_cutoff must be before validation_cutoff")
    union_find = _UnionFind(row["instance_id"] for row in rows)
    _union_equal_values(rows, union_find)
    date_by_id = {row["instance_id"]: row["deposition_date"] for row in rows}
    roles: dict[str, str] = {}
    crossing_components = 0
    for component in _components(rows, union_find):
        dates = [date_by_id[identifier] for identifier in component]
        maximum = max(dates)
        if maximum <= train_cutoff:
            role = "train"
        elif maximum <= validation_cutoff:
            role = "validation"
        else:
            role = "test"
        if min(dates) <= train_cutoff < maximum or min(dates) <= validation_cutoff < maximum:
            crossing_components += 1
        roles.update({identifier: role for identifier in component})
    audit = _isolation_audit(rows, roles)
    audit.update(
        {
            "train_cutoff": train_cutoff.isoformat(),
            "validation_cutoff": validation_cutoff.isoformat(),
            "components_promoted_to_later_period": crossing_components,
            "assignment_rule": "component_role_from_latest_deposition_date",
        }
    )
    if not audit["valid"]:
        raise SplitMatrixError("Temporal split failed exact-sequence/PDB isolation")
    return roles, audit


def symmetric_foldseek_score(query_tm: float, target_tm: float, rule: str) -> float:
    """Reduce directional Foldseek TM scores to one symmetric edge score."""
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (query_tm, target_tm)):
        raise SplitMatrixError("Foldseek TM scores must lie in [0, 1]")
    if rule == "reciprocal_min":
        return min(query_tm, target_tm)
    if rule == "geometric_mean":
        return math.sqrt(query_tm * target_tm)
    if rule == "reciprocal_max":
        return max(query_tm, target_tm)
    raise ValueError("Foldseek score rule must be reciprocal_min, geometric_mean, or reciprocal_max")


def _read_foldseek_edges(
    path: Path,
    known_ids: set[str],
    tm_threshold: float,
    score_rule: str = "reciprocal_max",
) -> tuple[list[tuple[str, str]], int]:
    if not math.isfinite(tm_threshold) or not 0 < tm_threshold <= 1:
        raise ValueError("tm_threshold must lie in (0, 1]")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"query", "target", "qtmscore", "ttmscore"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise SplitMatrixError("Foldseek table lacks reciprocal TM-score columns")
            edges: set[tuple[str, str]] = set()
            row_count = 0
            for row in reader:
                row_count += 1
                query, target = str(row["query"]), str(row["target"])
                if query not in known_ids or target not in known_ids:
                    raise SplitMatrixError("Foldseek table references an unknown structure")
                try:
                    score = symmetric_foldseek_score(
                        float(row["qtmscore"]),
                        float(row["ttmscore"]),
                        score_rule,
                    )
                except ValueError as error:
                    raise SplitMatrixError("Foldseek TM scores must be numeric") from error
                if query != target and score >= tm_threshold:
                    edges.add(tuple(sorted((query, target))))
            return sorted(edges), row_count
    except OSError as error:
        raise SplitMatrixError(f"Cannot read {path}") from error


def _greedy_component_roles(
    components: list[list[str]], *, seed: int, fractions: tuple[float, float, float]
) -> dict[str, str]:
    if len(components) < 3 or len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise SplitMatrixError("At least three components and positive role fractions are required")
    total_fraction = sum(fractions)
    normalized = tuple(value / total_fraction for value in fractions)
    total = sum(len(component) for component in components)
    targets = {role: total * fraction for role, fraction in zip(ROLES, normalized)}
    counts = {role: 0 for role in ROLES}
    ordered = sorted(
        components,
        key=lambda values: (
            -len(values),
            hashlib.sha256(f"{seed}\0{'|'.join(values)}".encode()).hexdigest(),
        ),
    )
    roles: dict[str, str] = {}
    for component in ordered:
        choices = []
        for role in ROLES:
            proposed = counts[role] + len(component)
            overflow = max(0.0, proposed - targets[role]) / max(1.0, targets[role])
            fill = proposed / max(1.0, targets[role])
            choices.append((overflow > 0, overflow, fill, ROLES.index(role), role))
        role = min(choices)[-1]
        counts[role] += len(component)
        roles.update({identifier: role for identifier in component})
    if any(counts[role] == 0 for role in ROLES):
        raise SplitMatrixError("Component assignment produced an empty role")
    return roles


def _antigen_fold_roles(
    rows: list[dict[str, Any]],
    *,
    edges: list[tuple[str, str]],
    reported_alignment_rows: int,
    tm_threshold: float,
    seed: int,
    fractions: tuple[float, float, float],
    metadata_cluster_policy: str = "ignore",
) -> tuple[dict[str, str], dict[str, Any]]:
    if metadata_cluster_policy not in METADATA_CLUSTER_POLICIES:
        raise ValueError(
            "metadata_cluster_policy must be ignore, antigen, or double_cold"
        )
    identifiers = [row["instance_id"] for row in rows]
    structural_union_find = _UnionFind(identifiers)
    for query, target in edges:
        structural_union_find.union(query, target)
    structural_components = _components(rows, structural_union_find)

    policy_union_find = _UnionFind(identifiers)
    _union_equal_values(rows, policy_union_find)
    for query, target in edges:
        policy_union_find.union(query, target)
    if metadata_cluster_policy in {"antigen", "double_cold"}:
        _union_metadata_cluster_columns(
            rows, policy_union_find, ("antigen_cluster_id",)
        )
    if metadata_cluster_policy == "double_cold":
        _union_metadata_cluster_columns(
            rows, policy_union_find, ("antibody_cluster_id",)
        )
    policy_components = _components(rows, policy_union_find)
    roles = _greedy_component_roles(policy_components, seed=seed, fractions=fractions)
    cross_role_edges = sum(roles[left] != roles[right] for left, right in edges)
    audit = _isolation_audit(rows, roles)
    cluster_audit = metadata_cluster_isolation_audit(rows, roles)
    enforced_cluster_columns = {
        "ignore": (),
        "antigen": ("antigen_cluster_id",),
        "double_cold": ("antigen_cluster_id", "antibody_cluster_id"),
    }[metadata_cluster_policy]
    enforced_clusters_valid = all(
        cluster_audit["fields"][column]["cross_role_value_count"] == 0
        for column in enforced_cluster_columns
    )
    audit.update(
        {
            "tm_threshold": tm_threshold,
            "reported_alignment_rows": reported_alignment_rows,
            "hard_edge_count": len(edges),
            "hard_edge_cross_role_count": cross_role_edges,
            # Retain the legacy fields while making their policy semantics
            # explicit. Exact sequence/PDB equality may merge otherwise
            # separate structural components.
            "component_count": len(policy_components),
            "component_definition": (
                "foldseek_hard_edges_plus_exact_pdb_antigen_sequence_"
                "and_antibody_sequence_equivalence"
                f"_plus_metadata_cluster_policy_{metadata_cluster_policy}"
            ),
            "metadata_cluster_policy": metadata_cluster_policy,
            "enforced_metadata_cluster_columns": list(enforced_cluster_columns),
            "metadata_cluster_isolation": cluster_audit,
            "policy_component_count": len(policy_components),
            "policy_largest_component_size": max(map(len, policy_components)),
            "structural_component_count": len(structural_components),
            "structural_largest_component_size": max(
                map(len, structural_components)
            ),
            "largest_component_size": max(map(len, policy_components)),
            "target_fractions": dict(zip(ROLES, fractions)),
            "assignment_seed": seed,
        }
    )
    audit["valid"] = (
        audit["valid"] and cross_role_edges == 0 and enforced_clusters_valid
    )
    if not audit["valid"]:
        raise SplitMatrixError("Antigen-fold split failed a hard isolation constraint")
    return roles, audit


def _write_manifest(
    path: Path,
    rows: list[dict[str, Any]],
    roles: dict[str, str],
    source_split: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instance_id": row["instance_id"],
                    "role": roles[row["instance_id"]],
                    "antigen_structure_id": row["antigen_structure_id"],
                    "antigen_sequence_sha256": row["antigen_sequence_sha256"],
                    "antibody_sequence_sha256": row["antibody_sequence_sha256"],
                    "pdb_assembly_id": row["pdb_assembly_id"],
                    "deposition_date": row["deposition_date"].isoformat(),
                    "antigen_cluster_id": row["antigen_cluster_id"],
                    "antibody_cluster_id": row["antibody_cluster_id"],
                    "source_split": source_split,
                }
            )


def build_split_matrix(
    *,
    metadata_path: Path,
    metadata_sha256: str,
    prepared_path: Path,
    prepared_sha256: str,
    foldseek_alignments_path: Path,
    foldseek_alignments_sha256: str,
    foldseek_receipt_path: Path,
    foldseek_receipt_sha256: str,
    output_root: Path,
    train_cutoff: date = date(2022, 12, 31),
    validation_cutoff: date = date(2024, 6, 30),
    tm_threshold: float = 0.50,
    foldseek_score_rule: str = "reciprocal_max",
    assignment_seed: int = 20_260_730,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    metadata_cluster_policy: str = "ignore",
) -> dict[str, Any]:
    """Build all three frozen manifests atomically without reading task labels."""
    authenticated = {
        "metadata_sha256": authenticate_file(metadata_path, metadata_sha256, "metadata"),
        "prepared_sha256": authenticate_file(prepared_path, prepared_sha256, "prepared split"),
        "foldseek_alignments_sha256": authenticate_file(
            foldseek_alignments_path, foldseek_alignments_sha256, "Foldseek alignments"
        ),
        "foldseek_receipt_sha256": authenticate_file(
            foldseek_receipt_path, foldseek_receipt_sha256, "Foldseek receipt"
        ),
    }
    metadata = _load_csv(metadata_path, "INSTANCE")
    prepared = _load_prepared(prepared_path)
    rows = _base_rows(metadata, prepared)
    identifiers = {row["instance_id"] for row in rows}
    try:
        foldseek_receipt = json.loads(foldseek_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitMatrixError("Cannot parse Foldseek receipt") from error
    if (
        foldseek_receipt.get("schema_version") != FOLDSEEK_RECEIPT_FORMAT
        or foldseek_receipt.get("query_target_same_database") is not True
        or int(foldseek_receipt.get("alignment_type", -1)) != 1
        or foldseek_receipt.get("test_labels_read") is not False
        or int(foldseek_receipt.get("structure_count", -1)) != len(identifiers)
        or foldseek_receipt.get("complete_ordered_cartesian_grid") is not True
        or int(foldseek_receipt.get("expected_ordered_pair_count", -1))
        != len(identifiers) ** 2
        or int(foldseek_receipt.get("raw_row_count", -1)) != len(identifiers) ** 2
        or foldseek_receipt.get("structure_ids_sha256")
        != structure_ids_sha256(identifiers)
    ):
        raise SplitMatrixError("Foldseek receipt does not authenticate this all-vs-all cohort")
    edges, alignment_rows = _read_foldseek_edges(
        foldseek_alignments_path,
        identifiers,
        tm_threshold,
        score_rule=foldseek_score_rule,
    )
    original_roles = {row["instance_id"]: row["original_role"] for row in rows}
    temporal_roles, temporal_audit = _temporal_roles(
        rows, train_cutoff=train_cutoff, validation_cutoff=validation_cutoff
    )
    antigen_fold_roles, antigen_fold_audit = _antigen_fold_roles(
        rows,
        edges=edges,
        reported_alignment_rows=alignment_rows,
        tm_threshold=tm_threshold,
        seed=assignment_seed,
        fractions=fractions,
        metadata_cluster_policy=metadata_cluster_policy,
    )
    antigen_fold_audit["foldseek_score_rule"] = foldseek_score_rule
    original_audit = _isolation_audit(rows, original_roles)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite split matrix: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        roles = {
            "original": original_roles,
            "temporal": temporal_roles,
            "antigen_fold": antigen_fold_roles,
        }
        for name, values in roles.items():
            _write_manifest(temporary / f"{name}.csv", rows, values, name)
        receipt = {
            "format_version": SPLIT_MATRIX_FORMAT,
            "authority": {
                **authenticated,
                "task_labels_read": False,
                "predictions_read": False,
                "foldseek_version": foldseek_receipt.get("foldseek_version"),
            },
            "cohort": {
                "instance_count": len(rows),
                "instance_ids_sha256": structure_ids_sha256(identifiers),
                "source_metadata_instance_count": len(metadata),
                "source_metadata_instances_outside_prepared_cohort": (
                    len(metadata) - len(prepared)
                ),
                "prepared_is_complete_metadata_subset": True,
            },
            "manifests": {
                name: {
                    "path": f"{name}.csv",
                    "sha256": sha256_file(temporary / f"{name}.csv"),
                }
                for name in roles
            },
            "audits": {
                "original": original_audit,
                "temporal": temporal_audit,
                "antigen_fold": antigen_fold_audit,
            },
            "claim_boundary": (
                "Label-free split construction and leakage audit only; no model, "
                "generalization, or benchmark-improvement claim."
            ),
        }
        (temporary / "SPLIT_MATRIX_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "SPLIT_MATRIX_FORMAT",
    "SplitMatrixError",
    "METADATA_CLUSTER_POLICIES",
    "audit_split_manifest_shortcuts",
    "build_split_matrix",
    "metadata_cluster_isolation_audit",
    "symmetric_foldseek_score",
    "structure_ids_sha256",
]
