import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="ascii"))


def test_v2_preserves_v1_failure_without_scale_authorization() -> None:
    protocol = _protocol()
    assert protocol["format_version"] == (
        "antibody_ddp_scientific_parity_gate_v2"
    )
    assert protocol["failure_preservation"] == {
        "attribution_result_sha256": (
            "7adc722d101adb1ad3e6871e291b117e66a0d5ce9ef40eda0f7b6fbe0e7dedfd"
        ),
        "failed_v1_gate_result_sha256": (
            "91befd50f2826065830e4cfbf0af3d09e8542218f4c930053d7d4e5db1cb76cb"
        ),
        "v1_gate_remains_failed": True,
        "v1_thresholds_are_not_reinterpreted": True,
    }
    assert protocol["authorization"] == {
        "exact_recovery_execution_after_gate_passes": True,
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }


def test_v2_uses_a_training_excluded_fixed_evaluation_panel() -> None:
    protocol = _protocol()
    split = protocol["pilot_split"]
    assert split["expected_record_count"] == 8112
    assert split["training_count"] == 7300
    assert split["evaluation_count"] == 812
    assert split["training_count"] + split["evaluation_count"] == 8112
    assert split["selection"] == "lowest_ranking_digest_without_ties"
    assert len(protocol["evaluation"]["mask_streams"]) == 4
    assert protocol["evaluation"]["stages"] == [
        "before_training",
        "after_update_500",
    ]


def test_v2_is_three_seed_counterbalanced_and_fail_closed() -> None:
    protocol = _protocol()
    benchmark = protocol["benchmark"]
    assert benchmark["training_seeds"] == [17, 42, 73]
    assert benchmark["updates_per_run"] == 500
    assert benchmark["candidate_warmup_updates"] == 50
    assert len(benchmark["run_order"]) == 9
    assert sorted(
        item.split("_", 1)[1] for item in benchmark["run_order"]
    ) == [
        "bf16_selected",
        "bf16_selected",
        "bf16_selected",
        "fp32_compiled",
        "fp32_compiled",
        "fp32_compiled",
        "safe_eager",
        "safe_eager",
        "safe_eager",
    ]
    acceptance = protocol["acceptance"]
    assert acceptance["short_horizon_numerical"] == {
        "maximum_gradient_norm_relative_difference": 0.01,
        "maximum_loss_absolute_difference": 0.005,
        "reference": "safe_eager",
        "required_updates": 20,
    }
    assert acceptance["evaluation_noninferiority"][
        "bootstrap_iterations"
    ] == 10000
    assert acceptance["evaluation_noninferiority"][
        "require_all_comparisons"
    ] is True


def test_v2_claim_boundary_forbids_biological_and_sota_claims() -> None:
    protocol = _protocol()
    forbidden = set(protocol["claim_boundary"]["forbidden"])
    assert {
        "biological_generalization",
        "downstream_model_quality",
        "oas_production_pretraining",
        "partner_specificity",
        "sixteen_or_thirty_two_gpu_scale_authorization",
        "state_of_the_art",
        "world_leading",
    } == forbidden
    assert protocol["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
