from __future__ import annotations

import sys
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
sys.path.insert(0, str(SOURCE))

import audit_pinder_antibody_s2_exact_four_view_v1 as audit
import build_pinder_antibody_s2_folds_v1 as folds
import build_pinder_antibody_s2_train_validation_vertex_separator_v1 as separator
import audit_pinder_antibody_s2_train_validation_exact_v1 as exact_audit


def test_assignment_format_matches_independent_auditor() -> None:
    assert separator.ASSIGNMENT_FORMAT_VERSION == exact_audit.ASSIGNMENT_FORMAT_VERSION


def test_chain_view_edges_are_lifted_to_concatenated_antibodies() -> None:
    rows = [
        audit.SourceRow("s1", "p1", "AAAA", "CCCC", "GGGG"),
        audit.SourceRow("s2", "p2", "AAAA", "DDDD", "HHHH"),
        audit.SourceRow("s3", "p3", "AAAT", "EEEE", "IIII"),
    ]
    vh_a = audit.sequence_sha256("AAAA")
    vh_b = audit.sequence_sha256("AAAT")
    edges = {
        "vh": {tuple(sorted((vh_a, vh_b)))},
        "vl": set(),
        "concatenated_antibody": set(),
        "antigen": set(),
    }
    antibody_edges, antigen_edges, evidence = separator.build_direct_forbidden_edges(
        rows, edges
    )
    antibodies = {row.antibody_digest for row in rows}
    assert antigen_edges == set()
    assert len(antibody_edges) == 3
    assert set().union(*(set(edge) for edge in antibody_edges)) == antibodies
    assert evidence["transitive_closure_used_as_a_cross_split_constraint"] is False


def test_separator_solver_retains_balanced_double_cold_split() -> None:
    rows = [
        folds.SourceRow(
            system_id=f"s{number}",
            pdb_id=f"p{number}",
            topology="test",
            antibody_digest=f"a{number}",
            antigen_digest=f"g{number}",
        )
        for number in range(8)
    ]
    antibody_roots = {row.antibody_digest: row.antibody_digest for row in rows}
    antigen_roots = {row.antigen_digest: row.antigen_digest for row in rows}
    components, pair_by_id, _ = folds.interaction_components(
        rows, antibody_roots, antigen_roots
    )
    solution, result, antibodies, antigens = separator.solve_separator(
        rows,
        components,
        pair_by_id,
        antibody_edges=set(),
        antigen_edges=set(),
        minimum_fraction=0.15,
        maximum_fraction=0.25,
        time_limit_seconds=10,
    )
    assert solution.shape == (separator.VariableIndex(8, 8, 8).count,)
    assert len(antibodies) == len(antigens) == 8
    assert result["optimal_retained_unique_exact_pair_count"] == 8
    assert result["retained_validation_unique_exact_pair_count"] == 2


def test_separator_can_discard_similarity_bridge() -> None:
    rows = [
        folds.SourceRow("s0", "p0", "test", "a0", "g0"),
        folds.SourceRow("s1", "p1", "test", "a1", "g1"),
        folds.SourceRow("s2", "p2", "test", "a2", "g2"),
        folds.SourceRow("s3", "p3", "test", "a3", "g3"),
        folds.SourceRow("s4", "p4", "test", "a4", "g4"),
        folds.SourceRow("s5", "p5", "test", "a5", "g5"),
        folds.SourceRow("s6", "p6", "test", "a6", "g6"),
        folds.SourceRow("s7", "p7", "test", "a7", "g7"),
    ]
    antibody_roots = {row.antibody_digest: row.antibody_digest for row in rows}
    antigen_roots = {row.antigen_digest: row.antigen_digest for row in rows}
    components, pair_by_id, _ = folds.interaction_components(
        rows, antibody_roots, antigen_roots
    )
    solution, result, antibodies, _ = separator.solve_separator(
        rows,
        components,
        pair_by_id,
        antibody_edges={("a0", "a1"), ("a1", "a2")},
        antigen_edges=set(),
        minimum_fraction=0.15,
        maximum_fraction=0.25,
        time_limit_seconds=10,
    )
    index = separator.VariableIndex(8, 8, 8)
    buckets = {
        value: (
            "train"
            if solution[index.antibody_train(number)] > 0.5
            else "validation"
            if solution[index.antibody_validation(number)] > 0.5
            else "discarded"
        )
        for number, value in enumerate(antibodies)
    }
    assert result["retained_validation_unique_exact_pair_count"] == 2
    assert not any(
        {buckets[left], buckets[right]} == {"train", "validation"}
        for left, right in (("a0", "a1"), ("a1", "a2"))
    )
