from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sorted_endpoint_ids_are_enforced(tmp_path: Path) -> None:
    module = load("prepare_string_cluster_inputs", "scripts/prepare_string_cluster_inputs.py")
    path = tmp_path / "ids.txt"
    path.write_text("a\nb\n", encoding="utf-8")
    assert module.read_sorted_ids(path) == {"a", "b"}
    path.write_text("b\na\n", encoding="utf-8")
    try:
        module.read_sorted_ids(path)
    except ValueError as error:
        assert "strictly sorted" in str(error)
    else:
        raise AssertionError("unsorted endpoint inventory was accepted")


def test_cluster_pair_collapse_keeps_strongest_and_removes_self(monkeypatch) -> None:
    module = load("collapse_string_cluster_pairs", "scripts/collapse_string_cluster_pairs.py")
    rows = [
        ["a", "b", "0", "10", "0", "5", "0", "0", "0", "800"],
        ["a2", "b", "0", "20", "0", "5", "0", "0", "0", "900"],
        ["a", "a2", "0", "99", "0", "99", "0", "0", "0", "999"],
        ["b", "c", "0", "15", "0", "5", "0", "0", "0", "850"],
    ]
    monkeypatch.setattr(module, "direct_rows", lambda _: iter(rows))
    selected, counts = module.collapse(
        Path("unused"), {"a": "A", "a2": "A", "b": "B", "c": "C"}
    )
    assert set(selected) == {("A", "B"), ("B", "C")}
    assert selected[("A", "B")][1][0:2] == ("a2", "b")
    assert counts == {
        "direct_edges": 4,
        "duplicate_cluster_pair_edges": 1,
        "missing_endpoint_memberships": 0,
        "self_cluster_edges_removed": 1,
        "unique_nonself_cluster_pairs": 2,
    }


def test_cluster_pair_collapse_fails_on_missing_membership(monkeypatch) -> None:
    module = load("collapse_string_cluster_pairs_missing", "scripts/collapse_string_cluster_pairs.py")
    monkeypatch.setattr(
        module,
        "direct_rows",
        lambda _: iter([["a", "missing", "0", "1", "0", "1", "0", "0", "0", "700"]]),
    )
    try:
        module.collapse(Path("unused"), {"a": "A"})
    except ValueError as error:
        assert "missing from MMseqs" in str(error)
    else:
        raise AssertionError("missing MMseqs membership was accepted")
