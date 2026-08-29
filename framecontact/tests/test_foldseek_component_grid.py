"""Contracts for label-free Foldseek graph-connectivity auditing."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_foldseek_component_grid.py"
    spec = importlib.util.spec_from_file_location("foldseek_component_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_symmetric_score_rules_distinguish_partial_from_global_similarity() -> None:
    module = _module()
    assert module.symmetric_score(0.9, 0.3, "reciprocal_max") == pytest.approx(0.9)
    assert module.symmetric_score(0.9, 0.3, "reciprocal_min") == pytest.approx(0.3)
    assert module.symmetric_score(0.9, 0.3, "geometric_mean") == pytest.approx(
        (0.27) ** 0.5
    )


def test_component_grid_uses_each_unordered_edge_once(tmp_path: Path) -> None:
    module = _module()
    ids = [f"case_{index}" for index in range(6)]
    rows = [
        {
            "instance_id": identifier,
            "pdb_assembly_id": identifier,
            "antigen_sequence_sha256": f"{index + 1:064x}",
            "antibody_sequence_sha256": f"{index + 101:064x}",
        }
        for index, identifier in enumerate(ids)
    ]
    path = tmp_path / "foldseek.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query", "target", "qtmscore", "ttmscore"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for query in ids:
            for target in ids:
                qtm = ttm = 1.0 if query == target else 0.1
                if {query, target} == {"case_0", "case_1"}:
                    qtm, ttm = 0.9, 0.3
                writer.writerow(
                    {
                        "query": query,
                        "target": target,
                        "qtmscore": qtm,
                        "ttmscore": ttm,
                    }
                )
    grid, ordered_rows = module.component_grid(
        rows,
        path,
        configurations=(("reciprocal_min", 0.5), ("reciprocal_max", 0.5)),
    )
    assert ordered_rows == 36
    by_rule = {row["score_rule"]: row for row in grid}
    assert by_rule["reciprocal_min"]["hard_undirected_edge_count"] == 0
    assert by_rule["reciprocal_max"]["hard_undirected_edge_count"] == 1
    assert by_rule["reciprocal_min"]["component_count"] == 6
    assert by_rule["reciprocal_max"]["component_count"] == 5
