from evals.run import run_gate


def test_eval_gate_unanswerable_full():
    summary = run_gate()
    assert summary["abstain_on_unanswerable"] == 1.0
    assert summary["n_unanswerable"] >= 8
    assert summary["hit_at_5"] is not None
    assert 0.0 <= summary["hit_at_5"] <= 1.0
    assert 0.0 <= summary["faithfulness"] <= 1.0
    assert 0.0 <= summary["multimodal_citation_id_match"] <= 1.0
