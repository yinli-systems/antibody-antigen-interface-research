"""Fail-closed Foldseek/TM-score audit for antigen-structure splits."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any

from .sabdab2_structure import authenticate_file, canonical_json_bytes


STRUCTURAL_AUDIT_FORMAT = "framecontact_antigen_structure_audit_v2"
FOLDSEEK_RECEIPT_FORMAT = "framecontact_foldseek_all_vs_all_receipt_v1"
ROLES = ("train", "validation", "test")
MANIFEST_COLUMNS = {
    "instance_id",
    "role",
    "antigen_structure_id",
    "antigen_sequence_sha256",
    "antibody_sequence_sha256",
    "pdb_assembly_id",
}
ALIGNMENT_COLUMNS = {"query", "target", "qtmscore", "ttmscore"}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StructuralNoveltyError(ValueError):
    """Raised when split provenance or structural isolation is invalid."""


class _UnionFind:
    def __init__(self, values: set[str]) -> None:
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


def _read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise StructuralNoveltyError(f"{path} has no header")
            return [
                {str(key): str(value).strip() for key, value in row.items()}
                for row in reader
            ]
    except OSError as error:
        raise StructuralNoveltyError(f"Cannot read {path}") from error


def _roles_for_value(rows: list[dict[str, str]], column: str) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    for row in rows:
        roles.setdefault(row[column], set()).add(row["role"])
    return roles


def _cross_role_values(rows: list[dict[str, str]], column: str) -> list[str]:
    return sorted(
        value for value, roles in _roles_for_value(rows, column).items() if len(roles) > 1
    )


def _structure_id_digest(structure_ids: set[str]) -> str:
    payload = "".join(f"{identifier}\n" for identifier in sorted(structure_ids)).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def symmetric_tm_score(query_tm: float, target_tm: float, rule: str) -> float:
    """Reduce two length-normalized TM scores under an explicit symmetric rule."""
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (query_tm, target_tm)):
        raise StructuralNoveltyError("TM scores must lie in [0, 1]")
    if rule == "reciprocal_min":
        return min(query_tm, target_tm)
    if rule == "geometric_mean":
        return math.sqrt(query_tm * target_tm)
    if rule == "reciprocal_max":
        return max(query_tm, target_tm)
    raise ValueError("TM score rule must be reciprocal_min, geometric_mean, or reciprocal_max")


def audit_antigen_structure_split(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    alignments_path: Path,
    alignments_sha256: str,
    foldseek_receipt_path: Path,
    foldseek_receipt_sha256: str,
    tm_threshold: float = 0.50,
    foldseek_score_rule: str = "reciprocal_max",
    boundary_width: float = 0.02,
    boundary_pair_limit: int = 1_000,
) -> dict[str, Any]:
    """Prove no structural hard edge or exact identifier crosses split roles."""
    if not math.isfinite(tm_threshold) or not 0 < tm_threshold <= 1:
        raise ValueError("tm_threshold must lie in (0, 1]")
    symmetric_tm_score(0.0, 0.0, foldseek_score_rule)
    if not math.isfinite(boundary_width) or boundary_width < 0:
        raise ValueError("boundary_width must be nonnegative")
    if (
        isinstance(boundary_pair_limit, bool)
        or not isinstance(boundary_pair_limit, int)
        or boundary_pair_limit <= 0
    ):
        raise ValueError("boundary_pair_limit must be a positive integer")
    authenticated_manifest = authenticate_file(
        manifest_path, manifest_sha256, "structural split manifest"
    )
    authenticated_alignments = authenticate_file(
        alignments_path, alignments_sha256, "Foldseek alignments"
    )
    authenticated_receipt = authenticate_file(
        foldseek_receipt_path, foldseek_receipt_sha256, "Foldseek receipt"
    )
    rows = _read_csv(manifest_path)
    if not rows or not MANIFEST_COLUMNS.issubset(rows[0]):
        raise StructuralNoveltyError(
            f"Manifest is missing columns: {sorted(MANIFEST_COLUMNS - set(rows[0]) if rows else MANIFEST_COLUMNS)}"
        )
    instance_ids = [row["instance_id"] for row in rows]
    if any(not value for value in instance_ids) or len(set(instance_ids)) != len(instance_ids):
        raise StructuralNoveltyError("Instance identifiers must be nonempty and unique")
    role_counts = {role: sum(row["role"] == role for row in rows) for role in ROLES}
    if any(count == 0 for count in role_counts.values()) or any(
        row["role"] not in ROLES for row in rows
    ):
        raise StructuralNoveltyError("Manifest must have nonempty train/validation/test roles")
    for column in ("antigen_sequence_sha256", "antibody_sequence_sha256"):
        if any(not _SHA256.fullmatch(row[column]) for row in rows):
            raise StructuralNoveltyError(f"{column} must contain lowercase SHA-256 digests")
    for column in ("antigen_structure_id", "pdb_assembly_id"):
        if any(not row[column] for row in rows):
            raise StructuralNoveltyError(f"{column} must be nonempty")

    structure_roles = _roles_for_value(rows, "antigen_structure_id")
    structure_ids = set(structure_roles)
    try:
        foldseek_receipt = json.loads(foldseek_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructuralNoveltyError("Cannot parse Foldseek receipt") from error
    if (
        foldseek_receipt.get("schema_version") != FOLDSEEK_RECEIPT_FORMAT
        or foldseek_receipt.get("query_target_same_database") is not True
        or int(foldseek_receipt.get("alignment_type", -1)) != 1
        or foldseek_receipt.get("test_labels_read") is not False
        or int(foldseek_receipt.get("structure_count", -1)) != len(structure_ids)
        or foldseek_receipt.get("structure_ids_sha256")
        != _structure_id_digest(structure_ids)
        or foldseek_receipt.get("complete_ordered_cartesian_grid") is not True
        or int(foldseek_receipt.get("expected_ordered_pair_count", -1))
        != len(structure_ids) ** 2
        or int(foldseek_receipt.get("raw_row_count", -1))
        != len(structure_ids) ** 2
    ):
        raise StructuralNoveltyError("Foldseek receipt does not prove this all-vs-all search")
    output_columns = set(foldseek_receipt.get("format_output", []))
    if not ALIGNMENT_COLUMNS.issubset(output_columns):
        raise StructuralNoveltyError("Foldseek receipt lacks reciprocal TM-score columns")

    union_find = _UnionFind(structure_ids)
    hard_edges: set[tuple[str, str]] = set()
    boundary_counts = {"cross_role": 0, "within_role": 0}
    boundary_heaps: dict[str, list[tuple[float, str, str, float]]] = {
        "cross_role": [],
        "within_role": [],
    }
    max_reported_cross_role_score = 0.0
    alignment_row_count = 0
    try:
        with alignments_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or not ALIGNMENT_COLUMNS.issubset(
                reader.fieldnames
            ):
                raise StructuralNoveltyError("Alignment table lacks required columns")
            for row in reader:
                alignment_row_count += 1
                query, target = str(row["query"]), str(row["target"])
                if query not in structure_ids or target not in structure_ids:
                    raise StructuralNoveltyError(
                        f"Alignment references unknown structure: {query!r}, {target!r}"
                    )
                try:
                    query_tm = float(row["qtmscore"])
                    target_tm = float(row["ttmscore"])
                except ValueError as error:
                    raise StructuralNoveltyError("TM scores must be numeric") from error
                if query == target:
                    continue
                pair = tuple(sorted((query, target)))
                score = symmetric_tm_score(query_tm, target_tm, foldseek_score_rule)
                if structure_roles[query] != structure_roles[target]:
                    max_reported_cross_role_score = max(
                        max_reported_cross_role_score, score
                    )
                # Boundary sampling uses the canonical query orientation so a
                # complete ordered pair is represented exactly once. Hard-edge
                # auditing below intentionally accepts either query direction.
                if query < target:
                    distance = abs(score - tm_threshold)
                    if distance <= boundary_width:
                        stratum = (
                            "cross_role"
                            if structure_roles[query] != structure_roles[target]
                            else "within_role"
                        )
                        boundary_counts[stratum] += 1
                        boundary_heap = boundary_heaps[stratum]
                        candidate = (-distance, query, target, score)
                        if len(boundary_heap) < boundary_pair_limit:
                            heapq.heappush(boundary_heap, candidate)
                        elif distance < -boundary_heap[0][0]:
                            heapq.heapreplace(boundary_heap, candidate)
                if score >= tm_threshold:
                    hard_edges.add(pair)
    except OSError as error:
        raise StructuralNoveltyError(f"Cannot read {alignments_path}") from error
    if alignment_row_count != len(structure_ids) ** 2:
        raise StructuralNoveltyError(
            "Alignment table row count differs from the authenticated Cartesian grid"
        )
    for query, target in hard_edges:
        union_find.union(query, target)
    cross_role_hard_edges = sorted(
        pair
        for pair in hard_edges
        if structure_roles[pair[0]] != structure_roles[pair[1]]
    )
    hard_edge_count = len(hard_edges)
    cross_role_hard_edge_count = len(cross_role_hard_edges)

    components: dict[str, list[str]] = {}
    for structure_id in structure_ids:
        components.setdefault(union_find.find(structure_id), []).append(structure_id)
    cross_role_components = []
    for members in components.values():
        roles = sorted({role for member in members for role in structure_roles[member]})
        if len(roles) > 1:
            cross_role_components.append(
                {"members": sorted(members), "roles": roles}
            )
    exact_antigen_overlap = _cross_role_values(rows, "antigen_sequence_sha256")
    exact_antibody_overlap = _cross_role_values(rows, "antibody_sequence_sha256")
    assembly_overlap = _cross_role_values(rows, "pdb_assembly_id")
    same_structure_overlap = sorted(
        identifier for identifier, roles in structure_roles.items() if len(roles) > 1
    )
    violations = {
        "cross_role_hard_edge_count": cross_role_hard_edge_count,
        "cross_role_component_count": len(cross_role_components),
        "same_structure_cross_role_count": len(same_structure_overlap),
        "exact_antigen_sequence_cross_role_count": len(exact_antigen_overlap),
        "exact_antibody_sequence_cross_role_count": len(exact_antibody_overlap),
        "pdb_assembly_cross_role_count": len(assembly_overlap),
    }
    valid = all(count == 0 for count in violations.values())
    return {
        "format_version": STRUCTURAL_AUDIT_FORMAT,
        "valid": valid,
        "authority": {
            "manifest_sha256": authenticated_manifest,
            "alignments_sha256": authenticated_alignments,
            "foldseek_receipt_sha256": authenticated_receipt,
            "foldseek_version": foldseek_receipt.get("foldseek_version"),
            "query_target_same_database": True,
            "alignment_type": 1,
            "test_labels_read": False,
        },
        "thresholds": {
            "hard_edge_symmetric_tm_score": tm_threshold,
            "foldseek_score_rule": foldseek_score_rule,
            "spot_check_boundary_width": boundary_width,
        },
        "counts": {
            "instances": len(rows),
            "structures": len(structure_ids),
            "roles": role_counts,
            "reported_alignment_rows": alignment_row_count,
            "hard_edges": hard_edge_count,
            "structural_components": len(components),
            "boundary_pair_count": sum(boundary_counts.values()),
            "cross_role_boundary_pair_count": boundary_counts["cross_role"],
            "within_role_boundary_pair_count": boundary_counts["within_role"],
            "boundary_pair_limit_per_stratum": boundary_pair_limit,
        },
        "violations": violations,
        "maximum_reported_cross_role_symmetric_tm_score": max_reported_cross_role_score,
        "boundary_spot_check_pairs": [
            {
                "query": query,
                "target": target,
                "symmetric_tm_score": score,
                "pair_stratum": stratum,
            }
            for stratum in ("cross_role", "within_role")
            for _, query, target, score in sorted(
                boundary_heaps[stratum],
                key=lambda value: (-value[0], value[1], value[2]),
            )
        ],
        "boundary_spot_check_pairs_truncated": any(
            boundary_counts[stratum] > boundary_pair_limit
            for stratum in boundary_counts
        ),
        "boundary_spot_check_truncation_by_stratum": {
            stratum: boundary_counts[stratum] > boundary_pair_limit
            for stratum in boundary_counts
        },
        "violation_examples": {
            "cross_role_hard_edges": [
                list(pair) for pair in cross_role_hard_edges[:20]
            ],
            "cross_role_components": cross_role_components[:20],
            "same_structures": same_structure_overlap[:20],
            "exact_antigen_sequences": exact_antigen_overlap[:20],
            "exact_antibody_sequences": exact_antibody_overlap[:20],
            "pdb_assemblies": assembly_overlap[:20],
        },
        "claim_boundary": (
            "Split-isolation audit only; it does not establish model generalization or "
            "performance. Boundary pairs still require recorded TM-align/US-align spot checks."
        ),
    }


def write_structural_audit(payload: dict[str, Any], output_path: Path) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite structural audit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload))


__all__ = [
    "FOLDSEEK_RECEIPT_FORMAT",
    "STRUCTURAL_AUDIT_FORMAT",
    "StructuralNoveltyError",
    "audit_antigen_structure_split",
    "symmetric_tm_score",
    "write_structural_audit",
]
