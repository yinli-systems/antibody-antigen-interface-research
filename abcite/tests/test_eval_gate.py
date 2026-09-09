from evals.run import load_gold, run_gate


def test_gold_at_least_150():
    gold = load_gold()
    assert len(gold) >= 150
    unans = [g for g in gold if g["expected_status"] == "abstain"]
    figs = [g for g in gold if g.get("must_cite_image_id")]
    assert len(unans) >= 30
    assert len(figs) >= 15
    sources = {g["source"] for g in gold}
    assert sources <= {"public_pdb", "synthetic_exam"}


def test_eval_gate_unanswerable_full():
    summary = run_gate()
    assert summary["abstain_on_unanswerable"] == 1.0
    assert summary["n_unanswerable"] >= 30
    assert summary["n_gold"] >= 150
    assert summary["hit_at_5"] is not None
    assert 0.0 <= summary["judge_score"] <= 1.0
