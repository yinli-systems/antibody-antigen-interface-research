"""Leakage-controlled connected-component splitting utilities."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SimilarityEdge:
    """One evidence-backed relationship that must not cross data splits."""

    left: str
    right: str
    source: str
    identity: float | None = None
    query_coverage: float | None = None
    target_coverage: float | None = None


class UnionFind:
    """Deterministic disjoint-set data structure with path compression."""

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
        """Join two sets while keeping the lexicographically smallest root."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def build_components(
    instance_ids: Iterable[str],
    similarity_edges: Iterable[SimilarityEdge] = (),
    equivalence_groups: Iterable[Iterable[str]] = (),
) -> list[tuple[str, ...]]:
    """Build components induced by every configured leakage relationship."""
    identifiers = sorted(set(str(value) for value in instance_ids))
    union_find = UnionFind(identifiers)
    known = set(identifiers)
    for edge in similarity_edges:
        if edge.left not in known or edge.right not in known:
            raise ValueError(f"Similarity edge references an unknown instance: {edge}")
        if edge.left != edge.right:
            union_find.union(edge.left, edge.right)
    for raw_group in equivalence_groups:
        group = sorted(set(str(value) for value in raw_group))
        unknown = set(group) - known
        if unknown:
            raise ValueError(f"Equivalence group references unknown instances: {sorted(unknown)}")
        for member in group[1:]:
            union_find.union(group[0], member)

    members: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        members[union_find.find(identifier)].append(identifier)
    return sorted(
        (tuple(sorted(component)) for component in members.values()),
        key=lambda component: (-len(component), component),
    )


def _normalized_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if len(fractions) < 2:
        raise ValueError("At least two split fractions are required")
    normalized = {str(name): float(value) for name, value in fractions.items()}
    if any(not name or value <= 0.0 for name, value in normalized.items()):
        raise ValueError("Split names must be non-empty and fractions must be positive")
    total = sum(normalized.values())
    return {name: value / total for name, value in normalized.items()}


def deterministic_component_split(
    components: Sequence[Sequence[str]],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole components with a deterministic size-balancing heuristic."""
    normalized = _normalized_fractions(fractions)
    canonical = [tuple(sorted(set(str(value) for value in component))) for component in components]
    if not canonical or any(not component for component in canonical):
        raise ValueError("Components must be non-empty")
    flattened = [identifier for component in canonical for identifier in component]
    if len(flattened) != len(set(flattened)):
        raise ValueError("An identifier occurs in more than one component")
    if len(canonical) < len(normalized):
        raise ValueError("There are fewer components than requested splits")

    total = len(flattened)
    targets = {name: total * fraction for name, fraction in normalized.items()}
    counts = {name: 0 for name in normalized}
    assigned_components = {name: 0 for name in normalized}
    assignments: dict[str, str] = {}

    ordered = sorted(
        canonical,
        key=lambda component: (
            -len(component),
            hashlib.sha256(f"{seed}:{','.join(component)}".encode("utf-8")).hexdigest(),
        ),
    )
    for index, component in enumerate(ordered):
        remaining = len(ordered) - index
        empty_splits = [name for name, count in assigned_components.items() if count == 0]
        candidates = empty_splits if remaining == len(empty_splits) else list(normalized)

        def objective(split_name: str) -> tuple[float, float, float, str]:
            projected = dict(counts)
            projected[split_name] += len(component)
            global_relative_error = sum(
                abs(projected[name] - targets[name]) / max(targets[name], 1.0)
                for name in normalized
            )
            proposed_fill_ratio = projected[split_name] / max(targets[split_name], 1.0)
            overfill = max(proposed_fill_ratio - 1.0, 0.0)
            current_fill_ratio = counts[split_name] / max(targets[split_name], 1.0)
            tie = hashlib.sha256(
                f"{seed}:{split_name}:{','.join(component)}".encode("utf-8")
            ).hexdigest()
            return global_relative_error, overfill, current_fill_ratio, tie

        selected = min(candidates, key=objective)
        for identifier in component:
            assignments[identifier] = selected
        counts[selected] += len(component)
        assigned_components[selected] += 1

    if any(value == 0 for value in assigned_components.values()):
        raise AssertionError("Component splitter produced an empty split")
    return assignments


def audit_component_split(
    components: Sequence[Sequence[str]],
    assignments: Mapping[str, str],
) -> dict[str, object]:
    """Verify component isolation and summarize split sizes."""
    split_counts: dict[str, int] = defaultdict(int)
    component_counts: dict[str, int] = defaultdict(int)
    violations: list[dict[str, object]] = []
    expected: set[str] = set()
    for component in components:
        members = [str(value) for value in component]
        expected.update(members)
        observed = {assignments.get(member) for member in members}
        if None in observed or len(observed) != 1:
            violations.append({"members": members, "observed_splits": sorted(str(x) for x in observed)})
            continue
        split_name = str(next(iter(observed)))
        split_counts[split_name] += len(members)
        component_counts[split_name] += 1
    extra = set(assignments) - expected
    missing = expected - set(assignments)
    return {
        "valid": not violations and not extra and not missing,
        "instance_count": len(expected),
        "component_count": len(components),
        "split_instance_counts": dict(sorted(split_counts.items())),
        "split_component_counts": dict(sorted(component_counts.items())),
        "violation_count": len(violations),
        "violations": violations[:20],
        "missing_instance_ids": sorted(missing),
        "extra_instance_ids": sorted(extra),
    }
