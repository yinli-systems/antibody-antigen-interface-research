import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols" / "antibody_foundation_pretraining_v1.json"
RUNBOOK_PATH = ROOT / "protocols" / "ANTIBODY_FOUNDATION_PRETRAINING_PROTOCOL_V1.md"


def test_protocol_is_machine_readable_and_fail_closed() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))

    assert protocol["format_version"] == (
        "antibody_foundation_pretraining_protocol_v1"
    )
    assert protocol["authorization"]["user_authorized"] is True
    assert protocol["authorization"][
        "independent_of_existing_downstream_protocols"
    ] is True
    assert protocol["test_access"] == {
        "downstream_heldout_access_authorized": False,
        "pretraining_data_selection_may_use_downstream_metrics": False,
        "sabdab2_heldout_access_authorized": False,
    }
    assert protocol["data"]["source"] == "Observed Antibody Space"
    assert protocol["data"]["license"] == "CC-BY-4.0"
    assert protocol["data"]["exact_deduplication"] is True


def test_protocol_freezes_distributed_scale_gates() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))
    distributed = protocol["distributed"]

    assert distributed["fsdp2_primary"] is True
    assert distributed["deepspeed_zero_stages_to_audit"] == [2, 3]
    assert distributed["scale_gpus"] == [1, 8, 32, 64]
    assert distributed["strong_scaling_efficiency_minimum"] == {
        "32": 0.8,
        "64": 0.7,
    }
    assert distributed["offload_default"] == "disabled"


def test_protocol_does_not_overclaim_pretraining_evidence() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))
    forbidden = set(protocol["claim_boundary"]["forbidden"])

    assert "claim_that_oas_contains_antibody_antigen_pairs" in forbidden
    assert "claim_partner_specificity_from_pretraining_loss" in forbidden
    assert "claim_unrevealed_downstream_test_decontamination" in forbidden
    assert "sota_without_identical_benchmark_evidence" in forbidden
    assert "world_leading_without_identical_benchmark_evidence" in forbidden

    runbook = RUNBOOK_PATH.read_text(encoding="ascii")
    assert "It cannot by itself prove" in runbook
    assert "No model is described as best" in runbook
