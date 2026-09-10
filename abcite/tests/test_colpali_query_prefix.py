from types import SimpleNamespace

from abcite_core.colpali_encoder import ensure_query_prefix


def test_ensure_query_prefix_fills_blank() -> None:
    proc = SimpleNamespace(query_prefix=None)
    assert ensure_query_prefix(proc) == "Query: "
    assert proc.query_prefix == "Query: "


def test_ensure_query_prefix_keeps_existing() -> None:
    proc = SimpleNamespace(query_prefix="Question: ")
    assert ensure_query_prefix(proc) == "Question: "
    assert proc.query_prefix == "Question: "


def test_ensure_query_prefix_fills_empty_string() -> None:
    proc = SimpleNamespace(query_prefix="")
    assert ensure_query_prefix(proc) == "Query: "
