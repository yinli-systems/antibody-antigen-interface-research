from pathlib import Path

from abcite_core.judge import FakeLLMJudge, HeuristicJudge, LLMJudge, judge_prompt
from abcite_core.metrics import faithfulness


def test_heuristic_judge_abstain():
    r = HeuristicJudge().score("q", "Insufficient evidence. Abstaining.", [], "abstain")
    assert r.score == 1.0
    assert r.backend == "heuristic"


def test_fake_llm_judge_records_prompt_contract():
    FakeLLMJudge.last_prompt = None
    cites = ["PDB 1N8Z trastuzumab binds HER2 domain IV, not pertuzumab domain II."]
    j = FakeLLMJudge()
    r = j.score("Does 1N8Z share pertuzumab epitope?", "1N8Z is domain IV not domain II.", cites, "answered")
    assert r.backend == "fake-llm"
    assert FakeLLMJudge.last_prompt
    assert "CITED CHUNKS" in FakeLLMJudge.last_prompt
    assert "STATUS: answered" in FakeLLMJudge.last_prompt
    prompt = judge_prompt("q", "a", cites, "answered")
    assert "score" in prompt.lower() or "JSON" in prompt


def test_llm_judge_uses_complete_and_cache(tmp_path):
    calls = {"n": 0}

    def complete(prompt: str) -> str:
        calls["n"] += 1
        return '{"score": 0.8, "supported": true, "rationale": "cites domain IV"}'

    j = LLMJudge(complete, cache_dir=tmp_path)
    a = j.score("q", "domain IV", ["trastuzumab domain IV 1N8Z"], "answered")
    b = j.score("q", "domain IV", ["trastuzumab domain IV 1N8Z"], "answered")
    assert a.score == 0.8
    assert b.score == 0.8
    assert calls["n"] == 1
    assert list(tmp_path.glob("*.json"))


def test_lexical_faithfulness_sanity():
    assert faithfulness("abstain", [], "abstain") == 1.0
