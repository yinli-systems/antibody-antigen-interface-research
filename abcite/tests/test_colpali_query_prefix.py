from pathlib import Path
from types import SimpleNamespace

from abcite_core.colpali_encoder import encoder_cache_name, ensure_query_prefix

ROOT = Path(__file__).resolve().parents[1]


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


def test_prefers_colqwen25_over_colqwen2_when_weights_present(monkeypatch) -> None:
    from abcite_core.colpali_encoder import preferred_visual_encoder_kind

    monkeypatch.delenv("ABCITE_COLPALI", raising=False)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen25_weights_present", lambda: True)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen2_weights_present", lambda: True)
    assert preferred_visual_encoder_kind() == "colqwen2.5"


def test_falls_back_to_colqwen2_without_25_weights(monkeypatch) -> None:
    from abcite_core.colpali_encoder import preferred_visual_encoder_kind

    monkeypatch.delenv("ABCITE_COLPALI", raising=False)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen25_weights_present", lambda: False)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen2_weights_present", lambda: True)
    assert preferred_visual_encoder_kind() == "colqwen2"


def test_colpali_off_skips_late_even_if_25_cached(monkeypatch) -> None:
    from abcite_core.colpali_encoder import preferred_visual_encoder_kind

    monkeypatch.setenv("ABCITE_COLPALI", "0")
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen25_weights_present", lambda: True)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen2_weights_present", lambda: True)
    assert preferred_visual_encoder_kind() != "colqwen2.5"
    assert preferred_visual_encoder_kind() != "colqwen2"


def test_no_late_weights_falls_back_clip_or_hash(monkeypatch) -> None:
    from abcite_core.colpali_encoder import preferred_visual_encoder_kind

    monkeypatch.delenv("ABCITE_COLPALI", raising=False)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen25_weights_present", lambda: False)
    monkeypatch.setattr("abcite_core.colpali_encoder.colqwen2_weights_present", lambda: False)
    monkeypatch.setattr(
        "abcite_core.clip_encoder.clip_forced_off",
        lambda: True,
    )
    assert preferred_visual_encoder_kind() == "local-image"


def test_cpu_ci_does_not_load_colqwen25(monkeypatch) -> None:
    monkeypatch.setenv("ABCITE_COLPALI", "auto")
    monkeypatch.setenv("ABCITE_CLIP", "0")
    monkeypatch.setattr("abcite_core.colpali_encoder.late_interaction_load_ok", lambda: False)
    called: list[int] = []

    def _boom():
        called.append(1)
        raise AssertionError("CI must not load ColQwen2.5 without CUDA / ABCITE_COLPALI=1")

    monkeypatch.setattr("abcite_core.colpali_encoder.try_load_colpali", _boom)
    from abcite_core.clip_encoder import build_vision_encoder

    enc = build_vision_encoder()
    assert called == []
    assert enc.name in {"local-image", "tiny-clip"}


def test_vidore_25_receipts_do_not_clobber_colqwen2() -> None:
    from evals.run_vidore_official import _suite_receipt, _task_receipt

    ds = "vidore/arxivqa_test_subsampled"
    assert _suite_receipt("colqwen2").name == "vidore_official_latest.json"
    assert "colqwen2_5" in _suite_receipt("colqwen2.5").name
    assert _task_receipt(ds, "colqwen2").name == "vidore_official_arxivqa.json"
    assert "colqwen2_5" in _task_receipt(ds, "colqwen2.5").name
    assert _task_receipt(ds, "colqwen2.5") != _task_receipt(ds, "colqwen2")


def test_text_rag_stays_bm25_default() -> None:
    from abcite_core.retrieval import HybridIndex
    from abcite_core.vision import LocalImageEncoder
    from ingest.chunking import load_corpus

    idx = HybridIndex(load_corpus(ROOT / "corpus"), vision=LocalImageEncoder())
    hits = idx.search("Does trastuzumab bind the HER2 domain IV epitope?", top_k=5)
    assert hits
    assert any(h.bm25 > 0 for h in hits)


def test_vidore_sota_flag_is_leaderboard_not_checkpoint() -> None:
    from evals.run_vidore_official import LEADERBOARD_SOTA_V1, PUBLISHED_COLQWEN25_V1

    assert LEADERBOARD_SOTA_V1 > PUBLISHED_COLQWEN25_V1
    assert 0.8947 < LEADERBOARD_SOTA_V1
    assert abs(0.8947 - PUBLISHED_COLQWEN25_V1) <= 0.005
    receipt = (ROOT / "evals" / "vidore_official_colqwen2_5_latest.json").read_text(encoding="utf-8")
    assert '"sota": false' in receipt
    assert '"reproduced_published_checkpoint": true' in receipt
    assert "not ViDoRe SOTA" in receipt
