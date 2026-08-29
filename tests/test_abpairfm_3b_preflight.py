"""Tests for the frozen ABPairFM 3B preflight gate."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/abpairfm_3b_pretraining_v1.json"
INVENTORY = ROOT / "protocols/abpairfm_3b_inventory_initial_v1.json"
SCRIPT = ROOT / "scripts/audit_abpairfm_3b_preflight.py"


def _module():
    spec = importlib.util.spec_from_file_location("abpairfm_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_protocol_freezes_requested_scale_and_test_boundary() -> None:
    protocol = _load(PROTOCOL)
    assert protocol["model"]["architecture"]["number_of_layers"] == 36
    assert protocol["model"]["architecture"]["embedding_dimension"] == 2560
    assert protocol["model"]["architecture"]["number_of_attention_heads"] == 40
    assert protocol["stage_a_pair_interface"]["example_count_minimum"] == 2_500_000
    assert protocol["stage_a_pair_interface"]["example_count_maximum"] == 25_000_000
    assert protocol["stage_a_pair_interface"]["consumed_residue_token_target"] == 5_000_000_000
    assert protocol["stage_b_cdr_specialization"]["minimum_native_vh_vl_pairs"] == 5_500_000
    assert protocol["stage_b_cdr_specialization"]["consumed_residue_token_target"] == 2_000_000_000
    assert protocol["stage_b_cdr_specialization"]["cdr_mask_probability"] == 0.5
    assert protocol["optimization"]["partition"] == "gpu_4090"
    assert protocol["optimization"]["slurm"]["hp_4090_permitted"] is False
    assert all(value is False for value in protocol["test_access"].values())
    assert protocol["evaluation"]["hard_decoy"]["binary_noninteraction_label_for_unmeasured_pairs"] is None


def test_initial_inventory_fails_closed_with_specific_real_blockers() -> None:
    payload = _module().audit(_load(PROTOCOL), _load(INVENTORY))
    assert payload["status"] == "BLOCKED"
    assert payload["authorization"] == {
        "development_evaluation": False,
        "gpu_smoke": False,
        "production_pretraining_stage_a": False,
        "production_pretraining_stage_b": False,
        "test_access": False,
    }
    assert "model_checkpoint_missing" in payload["blockers"]["model_and_stage_a_data"]
    assert "ppi_post_dedup_example_count_outside_frozen_range" in payload["blockers"]["model_and_stage_a_data"]
    assert "native_vh_vl_pair_count_below_5_5m" in payload["blockers"]["stage_b_data"]


def test_complete_inventory_authorizes_in_order_without_test_access() -> None:
    inventory = deepcopy(_load(INVENTORY))
    inventory["model"].update(
        {
            "checkpoint_present": True,
            "checkpoint_sha256": "a" * 64,
            "compatible_parameter_count": 3_000_000_000,
            "fsdp2_runner_present": True,
            "pair_aware_model_code_present": True,
        }
    )
    inventory["ppi"].update(
        {
            "normalized_training_examples": 3_300_000,
            "post_dedup_training_examples": 3_000_000,
            "unique_biological_pair_count_reported": True,
            "unique_interface_cluster_count_reported": True,
        }
    )
    for source in inventory["ppi"]["sources"].values():
        source["normalized"] = True
        source["decontamination_complete"] = True
        source["training_use_authorized"] = True
        if "raw_materialized" in source:
            source["raw_materialized"] = True
    inventory["ppi"]["sources"]["string_v12_physical"][
        "evidence_channel_filter_complete"
    ] = True
    inventory["ppi"]["sources"]["pinder_2024_02"].update(
        {"index_materialized": True, "metadata_materialized": True}
    )
    inventory["native_vh_vl"].update(
        {
            "admitted_native_pair_count": 5_500_000,
            "cdr_numbering_complete": True,
            "deduplication_complete": True,
            "donor_study_clone_split_complete": True,
            "native_pair_evidence_complete": True,
        }
    )
    inventory["development"].update(
        {
            "evaluation_harness_audited": True,
            "independent_leakage_audit_passed": True,
        }
    )
    inventory["systems"].update(
        {
            "eight_gpu_fixed_work_passed": True,
            "four_gpu_recovery_passed": True,
            "four_gpu_smoke_passed": True,
        }
    )
    payload = _module().audit(_load(PROTOCOL), inventory)
    assert payload["status"] == "READY"
    assert payload["authorization"] == {
        "development_evaluation": True,
        "gpu_smoke": True,
        "production_pretraining_stage_a": True,
        "production_pretraining_stage_b": True,
        "test_access": False,
    }
    assert payload["authorization"]["test_access"] is False
