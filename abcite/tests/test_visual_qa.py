import json
from pathlib import Path

import pytest

from abcite_core.figure_gallery import FigureGallery, FigureRec, load_figure_records
from abcite_core.holdout import query_contains_pdb, visual_query_ok
from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import HybridIndex, Hit
from abcite_core.vision import LocalImageEncoder
from abcite_core.visual_qa import (
    caption_keyword_overlap,
    evaluate_visual_qa,
    heuristic_vqa,
    load_visual_qa_gold,
    maxsim,
    maxsim_matrix,
    visual_qa_response,
)
from agents.graph import EvidenceGraph
from evals.run import _multimodal_snapshot_md

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "gold" / "visual_qa.jsonl"
FIX = ROOT / "evals" / "figures" / "fixture"


def _fixture_recs() -> list[FigureRec]:
    if not (FIX / "metadata.jsonl").exists():
        pytest.skip("fixture not downloaded yet")
    recs = load_figure_records(prefer_cache=False)
    if len(recs) < 8:
        pytest.skip("need 8+ fixture images")
    return recs


def test_visual_qa_gold_constraints():
    import json

    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 280
    golds: list[str] = []
    fail_n = 0
    ids: list[str] = []
    for row in rows:
        ids.append(row["id"])
        assert visual_query_ok(row["question"])
        assert not query_contains_pdb(row["question"])
        assert "1n8z" not in row["question"].lower()
        assert row.get("caption_bm25_should_fail") is True
        fail_n += 1
        if row.get("expected_status") != "abstain":
            assert row.get("gold_figure_ids")
            golds.append(row["gold_figure_ids"][0])
            assert row.get("must_look_at_image") is True
        else:
            assert not row.get("gold_figure_ids")
    assert len(ids) == len(set(ids))
    assert len(golds) == len(set(golds))
    assert fail_n == len(rows)
    abstain = [r for r in rows if r.get("expected_status") == "abstain"]
    assert len(abstain) >= 20
    assert len(golds) >= 250


def test_visual_qa_gold_caption_overlap_on_cache_metadata():
    meta = ROOT / "evals" / "figures" / "cache" / "metadata.jsonl"
    fix = ROOT / "evals" / "figures" / "fixture" / "metadata.jsonl"
    if not meta.exists() and not fix.exists():
        pytest.skip("no figure metadata")
    caps: dict[str, str] = {}
    for path in (meta, fix):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            caps[str(raw.get("figure_id") or "")] = str(raw.get("caption") or "")
    rows = [
        json.loads(line)
        for line in GOLD.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checked = 0
    for row in rows:
        if row.get("expected_status") == "abstain":
            continue
        cap = None
        for gid in row.get("gold_figure_ids") or []:
            if gid in caps:
                cap = caps[gid]
                break
        if cap is None:
            continue
        ov = caption_keyword_overlap(row["question"], cap)
        assert ov["should_fail"], (row["id"], ov["overlap_tokens"])
        checked += 1
    assert checked >= 20


def test_caption_bm25_should_fail_on_fixture_captions():
    recs = _fixture_recs()
    by_id = {r.figure_id: r for r in recs}
    items = load_visual_qa_gold(set(by_id), recs)
    flagged = [it for it in items if it.get("caption_bm25_should_fail") and it.get("gold_figure_ids")]
    assert len(flagged) >= 8
    for it in flagged:
        cap = by_id[it["gold_figure_ids"][0]].caption
        ov = caption_keyword_overlap(it["question"], cap)
        assert ov["should_fail"], (it["id"], ov["overlap_tokens"])


def test_visual_qa_eval_on_fixture():
    recs = _fixture_recs()
    gal = FigureGallery(recs, LocalImageEncoder())
    items = load_visual_qa_gold(set(gal.ids), recs)
    assert len([it for it in items if it.get("gold_figure_ids")]) >= 8
    report = evaluate_visual_qa(gal, items, rerank_k=5)
    assert report["protocol"] == "visual_qa"
    assert report["n_gallery"] == len(recs)
    for col in ("caption_bm25", "clip_image_only", "vision_rerank"):
        assert "recall@1" in report[col]
        assert "ndcg@5" in report[col]
    assert report["vqa"]["n"] >= 8
    assert "recall@1" not in report["vqa"]
    assert "end_to_end_accuracy" in report["vqa"]
    assert "oracle_figure_accuracy" in report["vqa"]
    assert "abstain_accuracy" in report["vqa"]
    assert "not the figure caption" in report["note"] or "not" in report["note"]


def test_vqa_abstains_when_figure_lacks_cues():
    recs = _fixture_recs()
    photo = next(r for r in recs if r.figure_id in {"fix-01", "pmc-00001"})
    miss_q = "rainbow circular dichroism polar plot with a magenta star on the origin"
    pred = heuristic_vqa(miss_q, photo, photo.path)
    assert pred.status == "abstain"
    assert pred.figure_id is None


def test_search_visual_and_graph_task():
    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha epitope cartoon",
            text="Figure caption: alpha-unique-caption membrane-proximal.",
            caption="alpha-unique-caption membrane-proximal domain IV",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
        Chunk(
            chunk_id="fig-beta",
            parent_id="txt-b",
            modality=Modality.FIGURE,
            title="Beta epitope cartoon",
            text="Figure caption: beta-unique-caption cryptic face.",
            caption="beta-unique-caption cryptic rbd ace2-distal",
            source_url="https://example.test/b",
            source_label="b",
            image_path="corpus/figures/fig-6w41.svg",
        ),
    ]
    idx = HybridIndex(chunks, vision=LocalImageEncoder())
    hits = idx.search_visual("emerald nuclear speckles against a dark field", top_k=2)
    assert hits
    assert hasattr(hits[0], "image")
    g = EvidenceGraph(idx, 0.12, False, "local")
    resp = g.invoke(
        "emerald nuclear speckles against a dark field",
        task="visual_qa",
    )
    assert resp.status.value in {"answered", "abstain"}
    evidence = g.invoke(
        "Does trastuzumab on HER2 share the pertuzumab domain II epitope?",
        task="evidence",
    )
    assert evidence.trace_id != resp.trace_id


class _LateToy:
    """MaxSim and mean-pool disagree on ranking — used to prove product path is MaxSim."""

    name = "late-toy"
    dim = 4
    late_interaction = True

    def encode_image(self, path: Path) -> list[float]:
        toks = self.encode_images_multi([path])[0]
        vec = toks.mean(axis=0)
        return vec.tolist()

    def encode_text(self, text: str) -> list[float]:
        toks = self.encode_queries_multi([text])[0]
        return toks.mean(axis=0).tolist()

    def caption(self, path: Path) -> str:
        return ""

    def encode_images_multi(self, paths: list[Path], batch_size: int = 1):
        import numpy as np

        out = []
        for path in paths:
            if "1n8z" in Path(path).name:
                out.append(np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1]], dtype=np.float32))
            else:
                out.append(np.array([[0.6, 0.6, 0.0, 0.0]], dtype=np.float32))
        return out

    def encode_queries_multi(self, texts: list[str], batch_size: int = 8):
        import numpy as np

        return [np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32) for _ in texts]


def test_search_visual_uses_maxsim_not_mean_pool():
    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha",
            text="alpha",
            caption="alpha",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
        Chunk(
            chunk_id="fig-beta",
            parent_id="txt-b",
            modality=Modality.FIGURE,
            title="Beta",
            text="beta",
            caption="beta",
            source_url="https://example.test/b",
            source_label="b",
            image_path="corpus/figures/fig-6w41.svg",
        ),
    ]
    toy = _LateToy()
    idx = HybridIndex(chunks, vision=toy)
    assert idx.visual_score_path == "maxsim"
    import numpy as np

    pooled = np.stack([np.asarray(toy.encode_image(Path(c.image_path))) for c in chunks])
    qv = np.asarray(toy.encode_text("token-match"), dtype=np.float32)
    pooled_order = [chunks[i].chunk_id for i in np.argsort(-(pooled @ qv))]
    hits = idx.search_visual("token-match", top_k=2)
    maxsim_order = [h.chunk.chunk_id for h in hits]
    assert pooled_order[0] == "fig-beta"
    assert maxsim_order[0] == "fig-alpha"
    local = HybridIndex(chunks, vision=LocalImageEncoder())
    assert local.visual_score_path == "pooled"


def test_search_by_image_uses_maxsim_not_mean_pool():
    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha",
            text="alpha",
            caption="alpha",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
        Chunk(
            chunk_id="fig-beta",
            parent_id="txt-b",
            modality=Modality.FIGURE,
            title="Beta",
            text="beta",
            caption="beta",
            source_url="https://example.test/b",
            source_label="b",
            image_path="corpus/figures/fig-6w41.svg",
        ),
    ]
    toy = _LateToy()
    idx = HybridIndex(chunks, vision=toy)
    qpath = Path("corpus/figures/fig-1n8z.svg")
    import numpy as np

    pooled = np.stack([np.asarray(toy.encode_image(Path(c.image_path))) for c in chunks])
    qv = np.asarray(toy.encode_image(qpath), dtype=np.float32)
    pooled_order = [chunks[i].chunk_id for i in np.argsort(-(pooled @ qv))]
    hits = idx.search_by_image(qpath, top_k=2, fuse_captions=False)
    maxsim_order = [h.chunk.chunk_id for h in hits]
    assert pooled_order[0] == "fig-beta"
    assert maxsim_order[0] == "fig-alpha"
    g = EvidenceGraph(idx, 0.12, False, "local")
    assert g.visual_gallery_kind() == "fixture"
    assert g.search_by_image(qpath, top_k=1, fuse_captions=False)[0].chunk.chunk_id == "fig-alpha"


def _write_toy_late_npz(tmp_path: Path) -> Path:
    import numpy as np

    npz = tmp_path / "embeddings-colqwen2.5-n2.npz"
    ids = ["pmc-alpha", "pmc-beta"]
    np.savez_compressed(
        npz,
        t0=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.1]], dtype=np.float16),
        t1=np.array([[0.6, 0.6, 0.0, 0.0]], dtype=np.float16),
    )
    npz.with_name(npz.stem + ".ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return npz


def test_pmc_gallery_stays_off_by_default(tmp_path, monkeypatch):
    from abcite_core.figure_gallery import resolve_pmc_gallery_npz

    monkeypatch.setenv("ABCITE_PMC_GALLERY", "0")
    npz = _write_toy_late_npz(tmp_path)
    monkeypatch.setenv("ABCITE_PMC_GALLERY_NPZ", str(npz))
    assert resolve_pmc_gallery_npz(_LateToy()) is None


def test_pmc_gallery_discovers_colqwen25_npz(tmp_path, monkeypatch):
    from abcite_core.figure_gallery import resolve_pmc_gallery_npz

    monkeypatch.setenv("ABCITE_PMC_GALLERY", "1")
    evals = tmp_path / "evals"
    evals.mkdir()
    src = _write_toy_late_npz(evals)
    toy = _LateToy()
    toy.name = "colqwen2.5"
    found = resolve_pmc_gallery_npz(toy, root=tmp_path)
    assert found == src


def test_search_visual_uses_pmc_npz_maxsim_when_flagged(tmp_path, monkeypatch):
    from abcite_core.figure_gallery import load_late_npz_gallery

    npz = _write_toy_late_npz(tmp_path)
    monkeypatch.setenv("ABCITE_PMC_GALLERY", str(npz))
    toy = _LateToy()
    chunks = [
        Chunk(
            chunk_id="fig-fixture",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Fixture",
            text="fixture",
            caption="fixture",
            source_url="https://example.test/f",
            source_label="f",
            image_path="corpus/figures/fig-1n8z.svg",
        )
    ]
    idx = HybridIndex(chunks, vision=toy)
    g = EvidenceGraph(idx, 0.12, False, "local")
    assert g.visual_gallery_kind().startswith("pmc-npz")
    hits = g.search_visual("token-match", top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["pmc-alpha", "pmc-beta"]
    gal = load_late_npz_gallery(npz, toy)
    assert gal.visual_score_path == "maxsim"
    g_off = EvidenceGraph(idx, 0.12, False, "local")
    monkeypatch.setenv("ABCITE_PMC_GALLERY", "0")
    g_off._pmc_gallery_loaded = False
    g_off.figure_gallery = None
    assert g_off.visual_gallery_kind() == "fixture"
    fixture_hits = g_off.search_visual("token-match", top_k=1)
    assert fixture_hits[0].chunk.chunk_id == "fig-fixture"


def test_search_by_image_uses_pmc_npz_when_flagged(tmp_path, monkeypatch):
    npz = _write_toy_late_npz(tmp_path)
    monkeypatch.setenv("ABCITE_PMC_GALLERY", str(npz))
    toy = _LateToy()
    chunks = [
        Chunk(
            chunk_id="fig-fixture",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Fixture",
            text="fixture",
            caption="fixture",
            source_url="https://example.test/f",
            source_label="f",
            image_path="corpus/figures/fig-6w41.svg",
        )
    ]
    idx = HybridIndex(chunks, vision=toy)
    g = EvidenceGraph(idx, 0.12, False, "local")
    qpath = Path("corpus/figures/fig-1n8z.svg")
    hits = g.search_by_image(qpath, top_k=2, fuse_captions=False)
    assert [h.chunk.chunk_id for h in hits] == ["pmc-alpha", "pmc-beta"]
    monkeypatch.setenv("ABCITE_PMC_GALLERY", "0")
    g_off = EvidenceGraph(idx, 0.12, False, "local")
    off = g_off.search_by_image(qpath, top_k=1, fuse_captions=False)
    assert off[0].chunk.chunk_id == "fig-fixture"


def _pmc_hit(fid: str, image_path: str | None, score: float = 1.0) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=fid,
            parent_id=fid,
            modality=Modality.FIGURE,
            title=fid,
            text=fid,
            caption="",
            source_url="",
            source_label="pmc",
            image_path=image_path,
        ),
        score=score,
        bm25=0.0,
        dense=0.0,
        image=score,
    )


def test_pmc_vqa_abstains_when_jpg_missing(monkeypatch):
    monkeypatch.delenv("ABCITE_PMC_IMAGE_DIR", raising=False)
    monkeypatch.delenv("ABCITE_FIGURES_CACHE", raising=False)
    monkeypatch.setenv("ABCITE_VLM", "0")
    hits = [_pmc_hit("pmc-missing-pixels", None)]
    resp = visual_qa_response(
        "emerald nuclear speckles against a dark field",
        hits,
        provider="local",
        trace_id="t-npz",
        latency_ms=1.0,
    )
    assert resp.status.value == "abstain"
    assert "no on-disk pixels" in resp.answer.lower() or "jpg cache" in resp.answer.lower()
    assert resp.citations == []


def test_pmc_vqa_heuristic_runs_on_cached_jpg(tmp_path, monkeypatch):
    from PIL import Image

    from abcite_core.figure_gallery import resolve_figure_pixels
    from abcite_core.models import AnswerStatus

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    jpg = img_dir / "pmc-alpha.jpg"
    Image.new("RGB", (16, 16), (20, 200, 40)).save(jpg, "JPEG")
    monkeypatch.setenv("ABCITE_PMC_IMAGE_DIR", str(img_dir))
    monkeypatch.setenv("ABCITE_VLM", "0")
    found = resolve_figure_pixels("pmc-alpha")
    assert found == jpg
    hits = [_pmc_hit("pmc-alpha", None)]
    resp = visual_qa_response(
        "green emerald nuclear speckles against a dark field",
        hits,
        provider="local",
        trace_id="t-jpg",
        latency_ms=1.0,
    )
    assert resp.status in {AnswerStatus.ANSWERED, AnswerStatus.ABSTAIN}
    assert resp.answer


def test_eval_snapshot_keeps_visual_qa_primary():
    md = _multimodal_snapshot_md()
    if not md:
        pytest.skip("no multimodal snapshot json")
    assert "Protocol A" in md and "Protocol B" in md
    assert "Appendix" in md
    assert "N=15949" in md
    assert "0.835" in md
    assert "Rerank helps BM25? **no**" in md


def test_align_recs_to_cached_late_ids(tmp_path, monkeypatch):
    from evals import run_visual_qa as rvq

    monkeypatch.setattr(rvq, "ROOT", tmp_path)
    cache = tmp_path / "evals" / "figures" / "cache"
    cache.mkdir(parents=True)
    (cache / "embeddings-colqwen2.5-n15949.ids.json").write_text(
        json.dumps(["a", "b"]), encoding="utf-8"
    )
    recs = [
        FigureRec("c", tmp_path / "c.jpg", "", None, "", "", ""),
        FigureRec("b", tmp_path / "b.jpg", "", None, "", "", ""),
        FigureRec("a", tmp_path / "a.jpg", "", None, "", "", ""),
    ]
    out = rvq._align_recs_to_cached_late_ids(recs, "vidore/colqwen2.5-v0.2")
    assert [r.figure_id for r in out] == ["a", "b"]


def test_maxsim_query_cap_defaults_to_64(monkeypatch):
    from abcite_core.visual_qa import maxsim_doc_cap, maxsim_query_cap

    monkeypatch.delenv("ABCITE_MAXSIM_MAX_Q", raising=False)
    monkeypatch.delenv("ABCITE_MAXSIM_MAX_D", raising=False)
    assert maxsim_query_cap() == 64
    assert maxsim_doc_cap() == 768
    monkeypatch.setenv("ABCITE_MAXSIM_MAX_Q", "128")
    monkeypatch.setenv("ABCITE_MAXSIM_MAX_D", "512")
    assert maxsim_query_cap() == 128
    assert maxsim_doc_cap() == 512


def test_maxsim_truncation_drops_late_query_tokens():
    q = __import__("numpy").zeros((80, 4), dtype="float32")
    q[70, 0] = 1.0
    d = __import__("numpy").zeros((3, 4), dtype="float32")
    d[0, 0] = 1.0
    full = maxsim(q, d)
    trunc = maxsim(q[:64], d)
    assert full > trunc
    assert abs(float(full) - 1.0) < 1e-5


def test_maxsim_matrix_shape_matches_scalar():
    rng = __import__("numpy").random.default_rng(0)
    queries = [rng.standard_normal((8, 128)).astype("float32") for _ in range(3)]
    docs = [rng.standard_normal((12, 128)).astype("float32") for _ in range(5)]
    mat = maxsim_matrix(queries, docs)
    assert mat.shape == (3, 5)
    for i in range(3):
        for j in range(5):
            assert abs(float(mat[i, j]) - maxsim(queries[i], docs[j])) < 0.05


def test_four_column_colpali_only_none_safe():
    from evals.run_visual_qa import _four_column

    col = {
        "caption_bm25": {"recall@1": 0.0, "ndcg@5": 0.0},
        "clip_image_only": {"recall@1": 0.08, "ndcg@5": 0.11},
        "vqa": {"n": 61},
    }
    four = _four_column(None, None, col)
    assert four["caption_bm25"]["recall@1"] == 0.0
    assert four["colpali_t2i"]["recall@1"] == 0.08
    assert four["openclip_t2i"] is None
    assert four["vision_beats_bm25_r@1"] is True


def test_merge_extra_column_keeps_colqwen2_four_column():
    from evals.run_visual_qa import _merge_extra_into_protocols

    prev = {
        "four_column": {
            "colpali_t2i": {"recall@1": 0.0733, "n": 273},
            "colqwen2_npz": "embeddings-colqwen2-n15949.npz hit",
        },
        "colpali": {"encoder": "colqwen2"},
    }
    extra = {
        "encoder": "colqwen2.5",
        "clip_image_only": {"recall@1": 0.1026, "n": 273},
        "npz": "embeddings-colqwen2.5-n15949.npz hit",
        "maxsim": "GPU n_q=273 n_d=15949",
        "device": "cuda",
        "cuda": True,
        "n_gallery": 15949,
        "n_answerable": 273,
    }
    out = _merge_extra_into_protocols(prev, extra)
    assert out["four_column"]["colpali_t2i"]["recall@1"] == 0.0733
    assert out["four_column"]["colqwen2_5_t2i"]["recall@1"] == 0.1026
    assert out["extra_columns"]["colqwen2.5"]["encoder"] == "colqwen2.5"
    assert out["colpali"]["encoder"] == "colqwen2"
