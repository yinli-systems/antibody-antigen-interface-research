from evals.clip_margin import clip_contrast
from evals.run import load_fixture, load_gold, load_split, run_gate


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
    assert "ndcg_at_10" in summary["fixture"]
    assert "public_pdb" in summary["fixture"]["by_source"]
    clip = summary["clip_contrast"]
    assert "mean_gap_rank1_minus_rank2" in clip
    assert clip["clip_sota_claim"] is False
    assert summary["judge_alignment"]["n_human_labeled"] >= 20
    assert summary["judge_alignment"]["n_annotators"] == 1
    assert summary["judge_alignment"]["second_annotator"] is False
    assert summary["demo_1n8z"]["abstain"] is False
    assert any("1n8z" in i.lower() for i in summary["demo_1n8z"]["retrieved_ids"])


def test_clip_contrast_report_exists():
    report = clip_contrast()
    assert report["n_figure_queries"] >= 1
    assert "mean_gap_rank1_minus_rank2" in report
    assert report["clip_sota_claim"] is False
