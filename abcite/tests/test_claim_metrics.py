from abcite_core.clip_encoder import biomedclip_requested, clip_model_name
from abcite_core.metrics import (
    claim_faithfulness,
    cohens_kappa,
    label_failure,
    mrr,
    ndcg_at_k,
    split_claims,
)


def test_claim_faithfulness_abstain_no_claims():
    assert claim_faithfulness("Insufficient evidence. Abstaining.", [], "abstain") == 1.0


def test_claim_split_and_span():
    answer = "The epitope is on domain IV. It is distinct from domain II."
    cites = ["trastuzumab epitope lies on domain IV and is distinct from domain II."]
    score = claim_faithfulness(answer, cites, "answered")
    assert score == 1.0
    assert len(split_claims(answer)) >= 2
    unsupported = claim_faithfulness("The secret accession binds antigen Q exclusively.", cites, "answered")
    assert unsupported < 1.0


def test_ranking_metrics():
    retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert ndcg_at_k(retrieved, ["a"], 10) == 1.0
    assert mrr(retrieved, ["c"]) == 1.0 / 3
    assert ndcg_at_k(retrieved, ["z"], 10) == 0.0


def test_cohens_kappa_and_failures():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohens_kappa([], []) is None
    assert (
        label_failure(
            expected_status="abstain",
            actual_status="answered",
            cited_ids=["txt-x"],
            retrieved_ids=["txt-x"],
            required_ids=[],
        )
        == "missed_abstain"
    )
    assert (
        label_failure(
            expected_status="answered",
            actual_status="abstain",
            cited_ids=[],
            retrieved_ids=[],
            required_ids=["txt-1n8z-parent"],
        )
        == "over_abstain"
    )


def test_biomedclip_env_stub(monkeypatch):
    monkeypatch.setenv("ABCITE_CLIP_MODEL", "biomedclip")
    assert clip_model_name() == "biomedclip"
    assert biomedclip_requested() is True
