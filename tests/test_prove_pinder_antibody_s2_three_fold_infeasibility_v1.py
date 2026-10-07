from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SOURCE_DIR = (
    Path(__file__).parents[1]
    / "remote_tools"
    / "abf_paircontact_v1"
    / "pinder_scale_v1"
)
AUDIT_SOURCE = SOURCE_DIR / "audit_pinder_antibody_s2_exact_four_view_v1.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_pinder_antibody_s2_exact_four_view_v1", AUDIT_SOURCE
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
sys.modules[AUDIT_SPEC.name] = AUDIT
AUDIT_SPEC.loader.exec_module(AUDIT)

SOURCE = SOURCE_DIR / "prove_pinder_antibody_s2_three_fold_infeasibility_v1.py"
SPEC = importlib.util.spec_from_file_location("pinder_s2_infeasibility", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(sequence: str) -> str:
    return AUDIT.sequence_sha256(sequence)


def test_chain_edges_are_lifted_to_concatenated_antibody_entities():
    rows = [
        AUDIT.SourceRow("s1", "1aaa", "AAAA", "CCCC", "MMMM"),
        AUDIT.SourceRow("s2", "2bbb", "AAAT", "DDDD", "NNNN"),
        AUDIT.SourceRow("s3", "3ccc", "EEEE", "FFFF", "PPPP"),
    ]
    parent_antibody = {row.antibody_digest: row.antibody_digest for row in rows}
    parent_antigen = {row.antigen_digest: row.antigen_digest for row in rows}
    edges = {
        "vh": {(digest("AAAA"), digest("AAAT"))},
        "vl": set(),
        "concatenated_antibody": set(),
        "antigen": set(),
    }
    antibody_roots, antigen_roots, pairs, evidence = MODULE.build_augmented_components(
        rows, parent_antibody, parent_antigen, edges
    )
    assert antibody_roots[rows[0].antibody_digest] == antibody_roots[
        rows[1].antibody_digest
    ]
    assert antibody_roots[rows[0].antibody_digest] != antibody_roots[
        rows[2].antibody_digest
    ]
    assert len(set(antigen_roots.values())) == 3
    assert len(pairs) == 3
    assert evidence["ingested_crossing_edge_count"]["vh"] == 1


def test_exact_same_vh_is_indivisible_even_without_a_crossing_edge():
    rows = [
        AUDIT.SourceRow("s1", "1aaa", "AAAA", "CCCC", "MMMM"),
        AUDIT.SourceRow("s2", "2bbb", "AAAA", "DDDD", "NNNN"),
    ]
    parent_antibody = {row.antibody_digest: row.antibody_digest for row in rows}
    parent_antigen = {row.antigen_digest: row.antigen_digest for row in rows}
    edges = {view: set() for view in MODULE.REQUIRED_VIEWS}
    antibody_roots, _, _, _ = MODULE.build_augmented_components(
        rows, parent_antibody, parent_antigen, edges
    )
    assert len(set(antibody_roots.values())) == 1


def test_axis_summary_reports_pair_degree_not_row_multiplicity():
    rows = [
        AUDIT.SourceRow("s1", "1aaa", "AAAA", "CCCC", "MMMM"),
        AUDIT.SourceRow("s2", "2bbb", "AAAA", "CCCC", "MMMM"),
        AUDIT.SourceRow("s3", "3ccc", "AAAA", "CCCC", "NNNN"),
    ]
    root = rows[0].antibody_digest
    roots = {row.antibody_digest: root for row in rows}
    pairs = {(row.antibody_digest, row.antigen_digest) for row in rows}
    summary, _, degree = MODULE.summarize_axis(roots, pairs, "antibody")
    assert degree == 2
    assert summary["largest_pair_degree_component"]["unique_exact_pair_degree"] == 2
