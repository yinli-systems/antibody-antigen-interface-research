"""Contract checks for the ABPairFM 3B development quality frontier."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "abpairfm_3b_quality_frontier_v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_quality_frontier_preserves_scale_and_test_boundaries() -> None:
    protocol = _protocol()
    scale = protocol["compute_optimality"]
    assert scale["minimum_stage_a_unique_pairs"] == 2_500_000
    assert scale["maximum_stage_a_unique_pairs"] == 25_000_000
    assert scale["minimum_stage_a_consumed_residue_tokens"] == 5_000_000_000
    assert scale["strict_scale_pool_minimum_cluster_pairs"] == 25_000_000
    assert protocol["authorization"]["production_pretraining"] is False
    assert protocol["authorization"]["test_access"] is False
    assert not protocol["test_access"]["asep_test_accessed"]
    assert not protocol["test_access"]["sabdab2_heldout_accessed"]


def test_quality_frontier_fails_closed_on_backbone_and_new_corpus_leakage() -> None:
    controls = _protocol()["strict_leakage_controls"]
    backbone = controls["historical_backbone_exposure"]
    new_corpus = controls["new_corpus_to_development"]
    assert backbone["esm2_uniref50_lineage_fully_reconstructable"] is False
    assert backbone["small_from_scratch_pair_model_control_required_before_sota_claim"] is True
    assert new_corpus["zero_exact_endpoint_overlap"] is True
    assert new_corpus["zero_exact_pair_overlap"] is True
    assert new_corpus["exclude_any_pair_touching_a_development_endpoint_cluster"] is True
    assert new_corpus["maximum_sequence_identity"] <= 0.3
    assert controls["test_sequences_labels_or_structure_bytes_permitted"] is False


def test_quality_frontier_requires_same_contract_small_and_large_baselines() -> None:
    protocol = _protocol()
    baselines = protocol["development_quality_frontier"]["baselines"]
    public = protocol["public_baselines"]
    assert "abpairfm_3b_stage_a" in baselines
    assert "pplm_650m_audited_checkpoint" in baselines
    assert "msa_pairformer_111m_audited_checkpoint" in baselines
    assert public["msa_pairformer_111m"]["model_parameters"] == 111_000_000
    assert public["pplm_650m"]["model_parameters"] == 650_000_000
    assert protocol["compute_optimality"]["quality_per_gpu_hour_is_required"] is True


def test_repeated_exposures_are_diverse_but_exactly_replayable() -> None:
    policy = _protocol()["compute_optimality"]["repeated_exposure_policy"]
    assert policy["augmentation_replay_key"] == [
        "global_seed",
        "canonical_pair_hash",
        "exposure_index",
    ]
    assert policy["deterministic_replay_required"] is True
    assert policy["pair_order_resampled_per_exposure"] is True
    assert policy["sequence_crop_resampled_per_exposure"] is True
    assert policy["token_mask_resampled_per_exposure"] is True


def test_objective_sequence_does_not_relabel_decoys_as_noninteractions() -> None:
    objectives = _protocol()["stage_a_objective_sequence"]
    retrieval = next(item for item in objectives if item["name"] == "cognate_retrieval_with_matched_partner_swaps")
    assert retrieval["negative_semantics"] == "ranking_decoy_not_measured_noninteraction"
    assert "development" not in objectives[-1]["entry_gate"]


def test_string_scale_tier_cannot_be_relabelled_as_contact_supervision() -> None:
    tiers = _protocol()["string_evidence_tiers"]
    authority = tiers["task_authority"]
    assert tiers["combined_high_confidence_threshold"] == 700
    assert tiers["direct_evidence_must_be_exhausted_before_transferred_only_fill"] is True
    assert tiers["physical_link_direct_binding_guaranteed"] is False
    assert authority["contact_supervision"] == "structural_sources_only"
    assert authority["transferred_only_string"] == [
        "pair_balanced_masked_language_modeling"
    ]
