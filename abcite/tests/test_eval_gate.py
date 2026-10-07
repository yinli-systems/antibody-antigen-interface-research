import json
from pathlib import Path

from abcite_core.metrics import label_failure
from abcite_core.models import Chunk, Modality
from abcite_core.retrieval import HybridIndex, expand_query
from abcite_core.vision import LocalImageEncoder
from evals.clip_margin import clip_contrast
from evals.run import load_fixture, load_gold, load_human_labels, load_split, run_gate

ROOT = Path(__file__).resolve().parents[1]


def test_gold_splits():
    fixture = load_fixture()
    hard = load_split("hard")
    all_rows = load_gold()
    assert len(fixture) >= 150
    assert len(hard) >= 35
    assert len(all_rows) == len(fixture) + len(hard)
    unans = [g for g in fixture if g["expected_status"] == "abstain"]
    figs = [g for g in fixture if g.get("must_cite_image_id")]
    assert len(unans) >= 30
    assert len(figs) >= 15
    sources = {g["source"] for g in fixture} | {g["source"] for g in hard}
    assert sources <= {"public_pdb", "synthetic_exam"}
    assert all(g.get("split") == "fixture" for g in fixture)
    assert all(g.get("split") == "hard" for g in hard)
    hard_unans = [g for g in hard if g["expected_status"] == "abstain"]
    assert len(hard_unans) >= 8
    classes = {g["query_class"] for g in hard}
    assert "paraphrase" in classes and "multi_hop" in classes
    assert "out_of_index" in classes and "residue" in classes


def test_eval_gate_unanswerable_full():
    summary = run_gate()
    assert summary["abstain_on_unanswerable"] == 1.0
    assert summary["n_fixture"] >= 150
    assert summary["n_hard"] >= 35
    assert summary["n_unanswerable"] >= 30
    assert summary["hit_at_5"] is not None
    assert 0.0 <= summary["judge_score"] <= 1.0
    assert summary["hard"]["hit_at_5"] < summary["fixture"]["hit_at_5"]
    assert summary["hard"]["ndcg_at_10"] >= 0.70
    assert summary["hard"]["hit_at_5"] < 0.99
    assert "ndcg_at_10" in summary["fixture"]
    assert "public_pdb" in summary["fixture"]["by_source"]
    clip = summary["clip_contrast"]
    assert "mean_gap_rank1_minus_rank2" in clip
    assert "mean_gap_image_only" in clip
    assert clip["clip_sota_claim"] is False
    assert clip.get("caption_fusion") is True
    assert summary["judge_alignment"]["n_human_labeled"] >= 40
    assert summary["judge_alignment"]["n_annotators"] == 1
    assert summary["judge_alignment"]["second_annotator"] is False
    assert summary["judge_alignment"]["kappa_kind"] == "human_vs_machine"
    assert summary["judge_alignment"]["kappa_human_vs_human"] is None
    assert summary["judge_alignment"]["kappa_human_vs_lexical"] is not None
    assert summary["judge_alignment"]["kappa_human_vs_claim_nli"] is not None
    tax = json.loads((ROOT / "evals/traces/hard_taxonomy.json").read_text())
    assert tax["n"] == summary["n_hard"] == 40
    allowed = {"ok", "hallucinated_cite", "missed_abstain", "over_abstain", "miss_figure", "miss_multi_hop"}
    assert {row["class"] for row in tax["items"]} <= allowed
    assert summary["demo_1n8z"]["abstain"] is False
    assert any("1n8z" in i.lower() for i in summary["demo_1n8z"]["retrieved_ids"])


def test_clip_contrast_report_exists():
    report = clip_contrast()
    assert report["n_figure_queries"] >= 1
    assert "mean_gap_rank1_minus_rank2" in report
    assert "mean_gap_image_only" in report
    assert report["clip_sota_claim"] is False


def test_query_rewrite_herceptin():
    expanded = expand_query(
        "Does Herceptin occupy the same ERBB2 extracellular face as dimerization-blocking mAb 2C4?"
    )
    assert "trastuzumab" in expanded.lower()
    assert "her2" in expanded.lower()
    assert "pertuzumab" in expanded.lower()


def test_caption_fusion_changes_ranking_vs_image_only():
    """Same visual vector, different captions → fused rank ≠ image-only."""

    class CollapsedVision(LocalImageEncoder):
        name = "collapsed-local"

        def encode_image(self, path: Path) -> list[float]:
            return [1.0] + [0.0] * 255

        def caption(self, path: Path) -> str:
            if "1n8z" in path.name:
                return "trastuzumab domain IV membrane-proximal unique caption"
            return "cryptic rbd ace2-distal unique caption"

    chunks = [
        Chunk(
            chunk_id="fig-alpha",
            parent_id="txt-a",
            modality=Modality.FIGURE,
            title="Alpha epitope cartoon",
            text="Figure caption: alpha unique membrane-proximal domain IV.",
            caption="trastuzumab domain IV membrane-proximal unique caption",
            source_url="https://example.test/a",
            source_label="a",
            image_path="corpus/figures/fig-1n8z.svg",
        ),
        Chunk(
            chunk_id="fig-beta",
            parent_id="txt-b",
            modality=Modality.FIGURE,
            title="Beta epitope cartoon",
            text="Figure caption: beta cryptic RBD face.",
            caption="cryptic rbd ace2-distal unique caption",
            source_url="https://example.test/b",
            source_label="b",
            image_path="corpus/figures/fig-6w41.svg",
        ),
    ]
    idx = HybridIndex(chunks, vision=CollapsedVision())
    path = ROOT / "corpus" / "figures" / "fig-1n8z.svg"
    fused = idx.search_by_image(path, top_k=2, fuse_captions=True)
    only = idx.search_by_image(path, top_k=2, fuse_captions=False)
    assert fused[0].chunk.chunk_id == "fig-alpha"
    assert [h.chunk.chunk_id for h in fused] != [h.chunk.chunk_id for h in only] or (
        fused[0].score != only[0].score
    )
    gaps = [fused[0].score - fused[1].score]
    assert gaps[0] > 0


def test_human_labels_size_and_kappa_kind():
    human = load_human_labels()
    assert len(human["items"]) >= 40
    assert human["n_annotators"] == 1
    scores = [float(x["human_faithfulness"]) for x in human["items"]]
    assert min(scores) < 0.5
    assert max(scores) > 0.5


def test_taxonomy_ok_and_miss_figure():
    assert (
        label_failure(
            expected_status="answered",
            actual_status="answered",
            cited_ids=["txt-1n8z-parent"],
            retrieved_ids=["txt-1n8z-parent", "txt-pertuzumab"],
            required_ids=["txt-1n8z-parent"],
        )
        == "ok"
    )
    assert (
        label_failure(
            expected_status="answered",
            actual_status="answered",
            cited_ids=["txt-1n8z-parent"],
            retrieved_ids=["txt-1n8z-parent"],
            required_ids=["fig-1n8z"],
            required_image_id="fig-1n8z",
        )
        == "miss_figure"
    )
