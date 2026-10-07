from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "remote_tools"
    / "abpairfm_distributed_contact_pretraining_v1"
    / "run_pinder_b_canary_v1.py"
)
SPEC = importlib.util.spec_from_file_location("pinder_b_canary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(index: int) -> dict[str, object]:
    return {
        "endpoint_cluster1": f"left-cluster-{index}",
        "endpoint_cluster2": f"right-cluster-{index}",
        "endpoint_exact1_sha256": f"left-exact-{index}",
        "endpoint_exact2_sha256": f"right-exact-{index}",
        "example_id": f"example-{index}",
        "pdb_id": f"p{index:03d}",
        "sequence1": "ACDEFGHIK",
        "sequence2": "LMNPQRSTV",
    }


def test_runner_resolves_repository_root() -> None:
    assert MODULE.REPOSITORY_ROOT == ROOT


def test_legal_groups_have_fail_closed_partner_derangements() -> None:
    records = [_record(index) for index in range(12)]
    known = {MODULE._known_pair(record) for record in records}
    groups = MODULE._legal_groups(records, group_count=2, known_pairs=known, seed=17)
    assert len(groups) == 2
    for group in groups:
        allowed = MODULE._swap_allowed(group, known)
        permutation = MODULE.build_partner_swap_permutation(allowed, seed=17)
        assert sorted(permutation.tolist()) == [0, 1, 2, 3]


def test_eight_rank_groups_have_fail_closed_partner_derangements() -> None:
    records = [_record(index) for index in range(24)]
    known = {MODULE._known_pair(record) for record in records}
    groups = MODULE._legal_groups(
        records,
        group_count=2,
        group_size=8,
        known_pairs=known,
        seed=29,
    )
    for group in groups:
        permutation = MODULE.build_partner_swap_permutation(
            MODULE._swap_allowed(group, known), seed=29
        )
        assert sorted(permutation.tolist()) == list(range(8))


def test_partner_groups_fail_closed_when_cross_pairs_exceed_token_budget() -> None:
    records = [_record(index) for index in range(8)]
    for index, record in enumerate(records):
        if index < 4:
            record["sequence1"] = "A" * 900
            record["sequence2"] = "C" * 100
        else:
            record["sequence1"] = "A" * 100
            record["sequence2"] = "C" * 900
    known = {MODULE._known_pair(record) for record in records}
    with pytest.raises(RuntimeError, match="legal 8-way partner swaps"):
        MODULE._legal_groups(
            records,
            group_count=1,
            group_size=8,
            known_pairs=known,
            seed=43,
            maximum_tokens=1024,
        )


def test_model_input_preserves_two_chain_residue_axes() -> None:
    tokens, chains, labels = MODULE._model_input("ACD", "EFGH", seed=3, mask=True)
    assert tokens.shape == chains.shape == labels.shape == (1, 11)
    assert chains[0].tolist() == [0] * 5 + [1] * 6
    assert int(labels.ne(-100).sum()) == 2


def test_warmup_cosine_schedule_hits_frozen_endpoints() -> None:
    values = [MODULE._learning_rate_scale(step, 10, 2, 0.1) for step in range(10)]
    assert values[0] == 0.5
    assert values[1] == 1.0
    assert values[2] == 1.0
    assert abs(values[-1] - 0.1) < 1e-12
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_input_dcp_verification_falls_back_before_process_group(monkeypatch) -> None:
    expected = {"content_sha256": "a" * 64}
    calls: list[tuple[Path, str]] = []

    def fake_verify(root: Path, generation: str) -> dict[str, str]:
        calls.append((root, generation))
        return expected

    monkeypatch.setattr(MODULE, "verify_dcp_generation", fake_verify)
    observed = MODULE._collective_verify_input_dcp(Path("/checkpoint"), "segment-1")
    assert observed == expected
    assert calls == [(Path("/checkpoint"), "segment-1")]


def test_pair_head_factory_builds_the_frozen_decoder_variant() -> None:
    low_rank = MODULE._build_pair_head(
        {"model": {"pair_decoder": "symmetric_low_rank", "pair_dimension": 8}}
    )
    bipartite = MODULE._build_pair_head(
        {
            "model": {
                "pair_decoder": "bipartite_edge_refinement",
                "pair_dimension": 8,
                "edge_dimension": 16,
                "refinement_steps": 2,
            }
        }
    )
    assert type(low_rank).__name__ == "SymmetricLowRankPairContactHead"
    assert type(bipartite).__name__ == "BipartiteEdgeRefinementContactHead"
