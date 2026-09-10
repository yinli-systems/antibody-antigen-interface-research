from types import SimpleNamespace

from abcite_core.colpali_encoder import encoder_cache_name, ensure_query_prefix


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


def test_encoder_cache_name_does_not_reuse_colqwen2() -> None:
    assert encoder_cache_name("vidore/colpali-v1.3", "colpali") == "colpali-v1.3"
    assert encoder_cache_name("/tmp/colpali-v1.3-local", "colpali") == "colpali-v1.3"
    assert encoder_cache_name("vidore/colqwen2-v1.0", "colqwen2") == "colqwen2"
    assert encoder_cache_name("/tmp/colqwen2.5-local", "colqwen2.5") == "colqwen2.5"
    assert encoder_cache_name("vidore/colqwen2-v1.0") != encoder_cache_name("vidore/colpali-v1.3")
    assert encoder_cache_name("vidore/colqwen2.5-v0.2") != encoder_cache_name("vidore/colqwen2-v1.0")
